import os
from pathlib import Path
import io
import numpy as np
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio

# ---- SETTINGS ----
DATA_DIR = Path("data/raw/Svarah/data")  # path to the folder with parquet files
SPLIT = "train"  # change to "train" / "validation" if needed
OUT_DIR = Path("out_wavs")  # output folder
TARGET_SR = 16000  # set to None to keep original SR; set to 16000 to resample

# Columns in your schema
AUDIO_COL = "audio_filepath"  # feature type: 'audio'
META_COLS = [
    "text", "gender", "age-group", "primary_language",
    "native_place_state", "native_place_district",
    "highest_qualification", "job_category", "occupation_domain",
    "duration"  # we’ll also capture the dataset's provided duration if present
]

def ensure_mono(x: np.ndarray) -> np.ndarray:
    """Make audio mono by averaging channels if needed."""
    print(f"Before Mono: Channels={x.ndim}")
    x = np.asarray(x)
    if x.ndim == 2:  # if stereo
        if x.shape[0] < x.shape[1]:  # (channels, samples)
            x = x.mean(axis=0)
        else:  # (samples, channels)
            x = x.mean(axis=1)
    print(f"After Mono: Channels={x.ndim}")
    return x.astype(np.float32, copy=False)

def csv_escape(val):
    if val is None:
        return ""
    s = str(val)
    return '"' + s.replace('"', '""') + '"'

def maybe_resample(wav: np.ndarray, sr: int, target_sr: int | None):
    print(f"Before Resample: SR={sr}, Channels={wav.ndim if wav.ndim > 1 else 1}")
    if target_sr is None or sr == target_sr:
        return wav, sr
    # high-quality resample with librosa
    import librosa
    wav = librosa.resample(wav.astype(np.float32, copy=False), orig_sr=sr, target_sr=target_sr)
    print(f"After Resample: SR={target_sr}, Channels={wav.ndim if wav.ndim > 1 else 1}")
    return wav.astype(np.float32, copy=False), target_sr

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load from Hugging Face Hub, but from local Parquet files instead of online
    # List all .parquet files in the folder
    parquet_files = list(DATA_DIR.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {DATA_DIR}. Please check the directory.")
    
    # Load the dataset from the parquet files
    ds = load_dataset("parquet", data_files=str(DATA_DIR / "*.parquet"), split=SPLIT)
    
    if AUDIO_COL not in ds.column_names:
        raise ValueError(f"'{AUDIO_COL}' not in dataset columns: {ds.column_names}")

    # 2) Ask datasets NOT to auto-decode (so we can avoid torchcodec)
    ds = ds.cast_column(AUDIO_COL, Audio(decode=False))

    # 3) Prepare manifest
    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        # Keep "text" as the 3rd column, others after it if they exist
        header = ["path", "duration", "text"] + [c for c in META_COLS if c not in ("text",)]
        f.write(",".join(header) + "\n")

        # 4) Iterate and export
        for i in tqdm(range(len(ds)), desc=f"Exporting {SPLIT}"):
            row = ds[i]
            a = row[AUDIO_COL]  # dict; may contain 'path' and/or 'bytes'
            # Try local path first
            wav_data = None
            sr = None

            if isinstance(a, dict) and a.get("path") and os.path.exists(a["path"]):
                # Read from path on disk
                data, sr = sf.read(a["path"], always_2d=False)
                wav_data = np.asarray(data, dtype=np.float32)
            elif isinstance(a, dict) and a.get("bytes"):
                # Read from bytes (in-memory)
                bio = io.BytesIO(a["bytes"])
                data, sr = sf.read(bio, always_2d=False)
                wav_data = np.asarray(data, dtype=np.float32)
            else:
                raise FileNotFoundError(
                    f"Row {i}: audio has neither a valid local path nor bytes. Got: {a}"
                )

            # Make mono
            wav_data = ensure_mono(wav_data)

            # Optional: resample to 16 kHz for ASR pipelines
            wav_data, sr = maybe_resample(wav_data, sr, TARGET_SR)

            # Save WAV (16‑bit PCM)
            out_wav = OUT_DIR / f"{i:08d}.wav"
            sf.write(str(out_wav), wav_data, sr, subtype="PCM_16")

            # Duration from sample count
            duration_sec = len(wav_data) / float(sr)

            # Build manifest row (include text first, then other metadata if present)
            text_val = row.get("text", "")
            fields = [out_wav.as_posix(), f"{duration_sec:.3f}", csv_escape(text_val)]

            for c in META_COLS:
                if c == "text":
                    continue
                fields.append(csv_escape(row.get(c, "")))
            f.write(",".join(fields) + "\n")

    print(f"\nDone ✅  WAVs in: {OUT_DIR}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
