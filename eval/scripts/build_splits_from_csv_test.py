from pathlib import Path
import csv, json
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ====== CONFIG ======
CSV_PATH = Path(r"C:\Users\johan\accent-benchmark\out_wavs\manifest.csv")
OUT_DIR  = Path(r"C:\Users\johan\accent-benchmark\manifests")
MAKE_SPLITS = True   # True -> write train/dev/test; False -> write a single test.jsonl with all rows

# Split sizes when MAKE_SPLITS=True
TEST_SIZE = 0.15          # 15% test
DEV_FRACTION_OF_REST = 0.10  # 10% of remaining -> ~75/10/15 overall

# Which CSV columns map to your JSONL schema
# Your CSV header (from the earlier export) should be:
# path,duration,text,gender,age-group,primary_language,native_place_state,native_place_district,highest_qualification,job_category,occupation_domain
DOMAIN_PRIORITY = ["occupation_domain", "job_category"]  # use whichever exists/non-empty

def read_csv_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def make_speaker_ids(rows):
    """
    If you don't have a true speaker column, synthesize stable speaker ids
    per (l1,state,gender) with a counter to avoid collisions.
    """
    ctr = defaultdict(int)
    spk_list = []
    for r in rows:
        l1 = (r.get("primary_language") or "").strip() or "UNK"
        st = (r.get("native_place_state") or "").strip() or "UNK"
        gd = (r.get("gender") or "").strip() or "U"
        key = (l1, st, gd)
        ctr[key] += 1
        spk = f"SPK_{l1}_{st}_{ctr[key]:04d}"
        spk_list.append(spk)
    return spk_list

def pick_domain(r):
    for c in DOMAIN_PRIORITY:
        v = r.get(c, "")
        if v:
            return v
    return "general"

def to_jsonl_item(r, speaker_id):
    return {
        "audio_filepath": r["path"],
        "duration": float(r["duration"]) if r.get("duration") else None,
        "text": r.get("text", ""),
        "speaker_id": speaker_id,
        "l1": r.get("primary_language", ""),
        "state": r.get("native_place_state", ""),
        "gender": r.get("gender", ""),
        "domain": pick_domain(r),
    }

def write_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for it in items:
            w.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    rows = read_csv_rows(CSV_PATH)
    if not rows:
        raise SystemExit(f"No rows found in {CSV_PATH}")

    spk_ids = make_speaker_ids(rows)
    items = [to_jsonl_item(r, spk) for r, spk in zip(rows, spk_ids)]

    if not MAKE_SPLITS:
        # Single test.jsonl with all rows
        write_jsonl(OUT_DIR / "test.jsonl", items)
        print(f"✅ Wrote {len(items)} items to {OUT_DIR/'test.jsonl'}")
        return

    # ===== Stratify by (l1,state) at *speaker* level to avoid leakage =====
    # Group items by synthetic speaker id
    by_spk = defaultdict(list)
    for it in items:
        by_spk[it["speaker_id"]].append(it)

    spk_groups = list(by_spk.values())

    # label for stratification (first item in each group is fine)
    def label(group):
        it = group[0]
        return f"{it.get('l1','')}|{it.get('state','')}"

    y = [label(g) for g in spk_groups]

    # Split speakers -> test
    try:
        rest_groups, test_groups = train_test_split(
            spk_groups, test_size=TEST_SIZE, random_state=42, stratify=y
        )
    except ValueError:
        # not enough per stratum -> fallback to random split
        rest_groups, test_groups = train_test_split(
            spk_groups, test_size=TEST_SIZE, random_state=42
        )

    # Split rest -> train/dev
    y_rest = [label(g) for g in rest_groups]
    try:
        train_groups, dev_groups = train_test_split(
            rest_groups,
            test_size=DEV_FRACTION_OF_REST,
            random_state=42,
            stratify=y_rest if len(set(y_rest)) > 1 else None,
        )
    except ValueError:
        train_groups, dev_groups = train_test_split(
            rest_groups, test_size=DEV_FRACTION_OF_REST, random_state=42
        )

    # Flatten
    def flatten(groups):
        for g in groups:
            for it in g:
                yield it

    train_items = list(flatten(train_groups))
    dev_items   = list(flatten(dev_groups))
    test_items  = list(flatten(test_groups))

    # Write
    write_jsonl(OUT_DIR / "train.jsonl", train_items)
    write_jsonl(OUT_DIR / "dev.jsonl",   dev_items)
    write_jsonl(OUT_DIR / "test.jsonl",  test_items)

    # Report
    def counts(items):
        from collections import Counter
        c = Counter((it.get("l1",""), it.get("state","")) for it in items)
        return sum(1 for _ in items), dict(c.most_common(10))

    ntr, _ = counts(train_items)
    ndv, _ = counts(dev_items)
    nts, _ = counts(test_items)

    print(f"✅ Wrote splits to {OUT_DIR}")
    print(f"  train: {ntr}  | dev: {ndv} | test: {nts}")

if __name__ == "__main__":
    main()
