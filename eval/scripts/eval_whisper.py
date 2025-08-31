# eval_whisper.py
import os, json, argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import whisper
import torch
from jiwer import Compose, ToLowerCase, RemoveMultipleSpaces, Strip, RemovePunctuation, wer, cer

def norm_pipeline():
    # Simple, consistent normalization for both refs and hyps
    return Compose([
        ToLowerCase(),
        RemovePunctuation(),
        RemoveMultipleSpaces(),
        Strip(),
    ])

def load_items(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="data/manifests/test.jsonl",
                    help="Path to test.jsonl")
    ap.add_argument("--model", type=str, default="medium",
                    choices=["tiny","base","small","medium","large-v2"],
                    help="Whisper model size")
    ap.add_argument("--language", type=str, default="en",
                    help="Language hint to Whisper (e.g., 'en' for Indian English)")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto","cpu","cuda"],
                    help="Force device or auto-detect")
    ap.add_argument("--output_dir", type=str, default="eval/results",
                    help="Where to write outputs")
    ap.add_argument("--fp16", action="store_true",
                    help="Use FP16 if supported (ignored on CPU)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print Whisper internals")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device resolution
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # FP16 only makes sense on CUDA
    fp16_flag = bool(args.fp16 and device == "cuda")

    print(f"Loading whisper model: {args.model} on {device} (fp16={fp16_flag})")
    model = whisper.load_model(args.model, device=device)

    # jiwer normalization
    norm = norm_pipeline()

    # Decode
    rows = []
    refs = []
    hyps = []

    items = list(load_items(manifest_path))
    for ex in tqdm(items, desc=f"Transcribing {args.model}"):
        audio_path = ex["audio_filepath"]
        # Optional: if your manifest has relative paths, base them on repo root
        audio_path = str(Path(audio_path))

        # Whisper transcribe
        # Set language hint; for English-accented Indian speech, `language="en"` is a good baseline
        result = model.transcribe(
            audio_path,
            language=args.language,
            task="transcribe",
            fp16=fp16_flag,
            verbose=args.verbose
        )

        ref = (ex.get("text") or "").strip()
        hyp = (result.get("text") or "").strip()

        refs.append(ref)
        hyps.append(hyp)

        # Collect row for raw dump
        row = dict(ex)
        row["hyp"] = hyp
        rows.append(row)

    # Save raw hypotheses
    model_tag = args.model.replace("/", "_")
    csv_out = out_dir / f"whisper_{model_tag}_raw.csv"
    pd.DataFrame(rows).to_csv(csv_out, index=False, encoding="utf-8")
    print(f"Saved raw hyps -> {csv_out}")

    # Normalize for scoring
    refs_n = [norm(r) for r in refs]
    hyps_n = [norm(h) for h in hyps]

    # For WER/CER we want tokenized versions consistently
    # jiwer.wer/cer will tokenize internally if given strings, but we pre-normalize for consistency
    _wer = wer(refs_n, hyps_n)
    _cer = cer(refs_n, hyps_n)

    metrics = {
        "model": args.model,
        "language": args.language,
        "device": device,
        "n_items": len(items),
        "wer": _wer,    # 0.0 .. 1.0
        "cer": _cer,    # 0.0 .. 1.0
    }

    metrics_out = out_dir / f"whisper_{model_tag}_metrics.json"
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", json.dumps(metrics, indent=2))
    print(f"Saved metrics -> {metrics_out}")

if __name__ == "__main__":
    main()
