from pathlib import Path
import json, soundfile as sf, collections, itertools

MANI_DIR = Path("manifests")  # adjust if needed
SPLITS = ["train.jsonl","dev.jsonl","test.jsonl"]

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def audit():
    stats = {}
    speakers = {}
    combos = {}

    for split in SPLITS:
        path = MANI_DIR / split
        if not path.exists():
            print(f"Skip missing {split}")
            continue

        files_ok = 0
        sr_ok = 0
        mono_ok = 0
        n = 0

        spk = set()
        combo = collections.Counter()

        for item in read_jsonl(path):
            n += 1
            wav = Path(item["audio_filepath"])
            if wav.exists():
                files_ok += 1
                try:
                    data, sr = sf.read(str(wav), always_2d=False)
                    if sr == 16000: sr_ok += 1
                    ch = (data.shape[1] if data.ndim == 2 else 1)
                    if ch == 1: mono_ok += 1
                except Exception as e:
                    pass

            spk.add(item.get("speaker_id",""))
            combo[(item.get("l1",""), item.get("state",""))] += 1

        stats[split] = dict(
            total=n,
            files_exist=files_ok,
            sr16k=sr_ok,
            mono=mono_ok
        )
        speakers[split] = spk
        combos[split] = combo

    # speaker leakage check
    leaks = {}
    for a, b in itertools.combinations([s for s in SPLITS if s in speakers], 2):
        leaks[(a,b)] = len(speakers[a].intersection(speakers[b]))

    # print summary
    print("\n=== DATA INTEGRITY ===")
    for s, v in stats.items():
        print(s, v)

    print("\n=== SPEAKER OVERLAP (should be 0) ===")
    for k, v in leaks.items():
        print(f"{k}: {v}")

    print("\n=== (l1,state) counts per split ===")
    for s, c in combos.items():
        print(f"\n{s}")
        for (l1, st), cnt in c.most_common(20):
            print(f"  {l1 or 'UNK'} | {st or 'UNK'} : {cnt}")

if __name__ == "__main__":
    audit()
