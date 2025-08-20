# build_manifests.py
import json
import os
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split

# INPUT: the CSV we created earlier OR build directly from your dataset iterator.
# Here we’ll read the CSV the export script wrote: out_wavs/manifest.csv
IN_CSV = Path("data\manifests\manifest.csv")
OUT_DIR = Path("manifests")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Column mapping from your earlier schema -> JSONL fields
# CSV header we wrote: path,duration,text,gender,age-group,primary_language,native_place_state,
#                      native_place_district,highest_qualification,job_category,occupation_domain
CSV_COLS = [
    "path","duration","text","gender","age-group","primary_language",
    "native_place_state","native_place_district","highest_qualification",
    "job_category","occupation_domain"
]

# Choose which column to use as "domain"
DOMAIN_SOURCE_PRIORITY = ["occupation_domain", "job_category"]

def parse_csv():
    rows = []
    with open(IN_CSV, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        for line in f:
            # simple CSV with quoted text; minimal robust parsing:
            # if your text contains commas, our earlier writer quoted it,
            # so a quick-and-dirty split will fail. If that happens, switch to csv.reader.
            import csv
            reader = csv.reader([line])
            parsed = next(reader)

            rec = {col: (parsed[idx[col]] if col in idx and idx[col] < len(parsed) else "") for col in CSV_COLS}
            rows.append(rec)
    return rows

def to_jsonl_item(r, speaker_id):
    # domain fallback
    domain = ""
    for c in DOMAIN_SOURCE_PRIORITY:
        if r.get(c):
            domain = r[c]
            break
    if not domain:
        domain = "general"

    # map fields
    item = {
        "audio_filepath": r["path"],
        "duration": float(r["duration"]) if r["duration"] else None,
        "text": r.get("text", ""),
        "speaker_id": speaker_id,
        "l1": r.get("primary_language", ""),
        "state": r.get("native_place_state", ""),
        "gender": r.get("gender", ""),
        "domain": domain,
    }
    return item

def main():
    data = parse_csv()

    # Create a stable speaker_id.
    # If you have a real speaker column, use it. Otherwise derive from (state, gender, l1)
    # plus a running counter per combo to avoid merging distinct people.
    counter = defaultdict(int)
    json_items = []
    for r in data:
        key = (r.get("primary_language",""), r.get("native_place_state",""), r.get("gender",""))
        counter[key] += 1
        spk = f"SPK_{(r.get('primary_language') or 'UNK')}_{(r.get('native_place_state') or 'UNK')}_{counter[key]:04d}"
        json_items.append(to_jsonl_item(r, spk))

    # Build stratify labels by (l1,state)
    y = [f"{it.get('l1','') }|{ it.get('state','')}" for it in json_items]

    # If you only need test:
    # with open(OUT_DIR / "test.jsonl", "w", encoding="utf-8") as w:
    #     for it in json_items:
    #         w.write(json.dumps(it, ensure_ascii=False) + "\n")
    #     return

    # Otherwise do train/dev/test with stratification
    # First split off test (e.g., 15%), then split dev from the remainder (e.g., 10% of rest)
    try:
        rest, test = train_test_split(json_items, test_size=0.15, random_state=42, stratify=y)
        y_rest = [f"{it.get('l1','')}|{it.get('state','')}" for it in rest]
        train, dev = train_test_split(rest, test_size=0.10, random_state=42, stratify=y_rest)
    except ValueError:
        # Fallback: not enough samples per stratum; do random splits without stratify
        rest, test = train_test_split(json_items, test_size=0.15, random_state=42)
        train, dev = train_test_split(rest, test_size=0.10, random_state=42)

    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as w:
            for it in items:
                w.write(json.dumps(it, ensure_ascii=False) + "\n")

    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "dev.jsonl",   dev)
    write_jsonl(OUT_DIR / "test.jsonl",  test)

    print(f"Done ✅  Wrote {len(train)} train, {len(dev)} dev, {len(test)} test to {OUT_DIR}")

if __name__ == "__main__":
    main()
