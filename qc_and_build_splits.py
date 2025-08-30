from pathlib import Path
import csv, json
from collections import defaultdict, Counter
import soundfile as sf
from sklearn.model_selection import train_test_split

# ====== PATHS ======
ROOT = Path(r"C:\Users\johan\accent-benchmark")
CSV_IN = ROOT / "out_wavs" / "manifest.csv"
CLEAN_CSV = ROOT / "out_wavs" / "clean_manifest.csv"
REJECT_CSV = ROOT / "out_wavs" / "rejected.csv"
OUT_JSONL_DIR = ROOT / "manifests"

# ====== QC PARAMS ======
MIN_DUR = 0.3     # seconds
MAX_DUR = 30.0     # seconds
DUR_ABS_TOL = 0.10 # seconds allowed diff between CSV duration and file duration
# (you can also use relative tol; this absolute tolerance is usually enough)

REQUIRE_MIN_TOKENS = 1

# ====== SPLIT PARAMS ======
MAKE_SPLITS = True
TEST_SIZE = 0.15
DEV_FRACTION_OF_REST = 0.10

# ====== DOMAIN SELECTION ======
DOMAIN_PRIORITY = ["occupation_domain", "job_category"]

def file_duration_sec(wav_path: Path) -> float | None:
    try:
        info = sf.info(str(wav_path))
        if info.samplerate and info.frames:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    return None

def read_manifest(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def qc_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    kept, rejected = [], []
    for r in rows:
        reasons = []
        wav = Path(r["path"])
        # 1) file must exist
        if not wav.exists():
            reasons.append("file_missing")
        # 2) optional: min token rule
        text = (r.get("text") or "").strip()
        if REQUIRE_MIN_TOKENS and len(text.split()) < REQUIRE_MIN_TOKENS:
            reasons.append("text_too_short")

        # 3) duration limits
        try:
            dur_manifest = float(r.get("duration", "") or "nan")
        except Exception:
            dur_manifest = None

        dur_file = file_duration_sec(wav)
        # prefer dur_file if available
        duration = dur_file if dur_file is not None else dur_manifest

        if duration is None:
            reasons.append("no_duration")
        else:
            if duration < MIN_DUR:
                reasons.append("too_short")
            if duration > MAX_DUR:
                reasons.append("too_long")

        # 4) duration consistency
        if dur_manifest is not None and dur_file is not None:
            if abs(dur_manifest - dur_file) > DUR_ABS_TOL:
                reasons.append("duration_mismatch")

        if reasons:
            r2 = dict(r)
            r2["reject_reasons"] = ";".join(reasons)
            r2["dur_file"] = f"{dur_file:.3f}" if dur_file is not None else ""
            r2["dur_manifest"] = f"{dur_manifest:.3f}" if dur_manifest is not None else ""
            rejected.append(r2)
        else:
            # keep + normalize duration to file duration if we have it
            r2 = dict(r)
            if dur_file is not None:
                r2["duration"] = f"{dur_file:.3f}"
            kept.append(r2)
    return kept, rejected

def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")  # empty file
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def make_speaker_ids(rows: list[dict]) -> list[str]:
    """
    Synthesize stable speaker_ids using (l1,state,gender) + a counter.
    Replace with true speaker id if you have it.
    """
    ctr = defaultdict(int)
    spk_ids = []
    for r in rows:
        l1 = (r.get("primary_language") or "").strip() or "UNK"
        st = (r.get("native_place_state") or "").strip() or "UNK"
        gd = (r.get("gender") or "").strip() or "U"
        key = (l1, st, gd)
        ctr[key] += 1
        spk_ids.append(f"SPK_{l1}_{st}_{ctr[key]:04d}")
    return spk_ids

def pick_domain(r: dict) -> str:
    for c in DOMAIN_PRIORITY:
        v = r.get(c, "")
        if v:
            return v
    return "general"

def to_jsonl_item(r: dict, spk_id: str) -> dict:
    return {
        "audio_filepath": r["path"],
        "duration": float(r["duration"]) if r.get("duration") else None,
        "text": r.get("text", ""),
        "speaker_id": spk_id,
        "l1": r.get("primary_language", ""),
        "state": r.get("native_place_state", ""),
        "gender": r.get("gender", ""),
        "domain": pick_domain(r),
    }

def write_jsonl(path: Path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for it in items:
            w.write(json.dumps(it, ensure_ascii=False) + "\n")

def stratified_speaker_splits(items: list[dict]):
    # group by speaker
    by_spk = defaultdict(list)
    for it in items:
        by_spk[it["speaker_id"]].append(it)
    groups = list(by_spk.values())

    # label per group for stratification
    def label(g): 
        it = g[0]
        return f"{it.get('l1','')}|{it.get('state','')}"
    y = [label(g) for g in groups]

    # split test
    try:
        rest, test = train_test_split(groups, test_size=TEST_SIZE, random_state=42, stratify=y)
    except ValueError:
        rest, test = train_test_split(groups, test_size=TEST_SIZE, random_state=42)

    # split dev from rest
    y_rest = [label(g) for g in rest]
    try:
        train, dev = train_test_split(
            rest, test_size=DEV_FRACTION_OF_REST, random_state=42,
            stratify=y_rest if len(set(y_rest)) > 1 else None
        )
    except ValueError:
        train, dev = train_test_split(rest, test_size=DEV_FRACTION_OF_REST, random_state=42)

    def flatten(gs):
        for g in gs:
            for it in g:
                yield it
    return list(flatten(train)), list(flatten(dev)), list(flatten(test))

def main():
    # 1) Read
    rows = read_manifest(CSV_IN)
    if not rows:
        raise SystemExit(f"No rows in {CSV_IN}")

    # 2) QC
    kept, rejected = qc_rows(rows)
    write_csv(CLEAN_CSV, kept)
    write_csv(REJECT_CSV, rejected)

    print("=== QC SUMMARY ===")
    print(f"Input rows      : {len(rows)}")
    print(f"Kept            : {len(kept)}")
    print(f"Rejected        : {len(rejected)}")
    if rejected:
        reasons = Counter(reason for r in rejected for reason in (r["reject_reasons"].split(";")))
        print("Top reject reasons:", dict(reasons.most_common(10)))
        print(f"Rejected CSV -> {REJECT_CSV}")

    # 3) Build JSONL(s)
    # map kept rows -> jsonl items
    spk_ids = make_speaker_ids(kept)
    items = [to_jsonl_item(r, s) for r, s in zip(kept, spk_ids)]

    OUT_JSONL_DIR.mkdir(parents=True, exist_ok=True)

    if not MAKE_SPLITS:
        write_jsonl(OUT_JSONL_DIR / "test.jsonl", items)
        print(f"✅ Wrote single test.jsonl with {len(items)} items to {OUT_JSONL_DIR}")
        return

    train_items, dev_items, test_items = stratified_speaker_splits(items)

    write_jsonl(OUT_JSONL_DIR / "train.jsonl", train_items)
    write_jsonl(OUT_JSONL_DIR / "dev.jsonl",   dev_items)
    write_jsonl(OUT_JSONL_DIR / "test.jsonl",  test_items)

    print("\n=== SPLIT SUMMARY ===")
    print(f"train: {len(train_items)}")
    print(f"dev  : {len(dev_items)}")
    print(f"test : {len(test_items)}")

if __name__ == "__main__":
    main()
