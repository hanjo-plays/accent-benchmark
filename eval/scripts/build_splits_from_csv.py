import os
from pathlib import Path
import csv
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ====== CONFIGURATION ======
# Define the input and output paths. 
# The input CSV contains metadata for each audio sample (like file paths, transcriptions, etc.)
CSV_PATH = Path(r"C:\Users\johan\accent-benchmark\out_wavs\manifest.csv")

# The output directory will store the splits: train.jsonl, dev.jsonl, and test.jsonl
OUT_DIR = Path(r"C:\Users\johan\accent-benchmark\manifests")

# Flag to indicate whether we want to create train, dev, and test splits or just a single test file
Make_Splits = True   # If True, splits the data; False creates a single test.jsonl

# Split configuration: 15% for testing, 10% of the rest for development (dev)
Test_size = 0.15          # 15% data will go into the test set
Validation_Rest = 0.10  # 10% of the remaining data will go into the dev set

# Columns in the CSV file that will be mapped to the JSONL format
# These are the metadata fields available for each audio sample in the CSV
DataCols = [
    "text", "gender", "age-group", "primary_language",
    "native_place_state", "native_place_district",
    "highest_qualification", "job_category", "occupation_domain",
    "duration"  # Duration of the audio file in seconds
]

# Domain selection priority: we'll pick the first non-empty field from this list as the 'domain'
DOMAIN_PRIORITY = ["occupation_domain", "job_category"]

def read_csv_rows(csv_path: Path):
    """
    Read the CSV file at `csv_path` and return its rows as a list of dictionaries.
    Each dictionary contains metadata for one audio file, where the keys are the column names.
    """
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)  # csv.DictReader reads each row as a dictionary
        for r in reader:
            rows.append(r)  # Append each row to the list of rows
    return rows

def make_speaker_ids(rows):
    """
    Generate synthetic speaker IDs based on the combination of language (l1), state, and gender.
    This is useful when the dataset does not include a true speaker ID, but we want stable IDs for each unique speaker.
    """
    ctr = defaultdict(int)  # A defaultdict to count occurrences of each (l1, state, gender) combination
    spk_list = []  # List to store generated speaker IDs

    for r in rows:
        # Extract primary language, state, and gender for each row
        l1 = (r.get("primary_language") or "").strip() or "UNK"  # Default to "UNK" if missing
        st = (r.get("native_place_state") or "").strip() or "UNK"
        gd = (r.get("gender") or "").strip() or "U"  # Default gender to "U" (Unknown)

        # Create a key based on (language, state, gender)
        key = (l1, st, gd)
        ctr[key] += 1  # Increment the counter for this combination

        # Create a unique speaker ID in the format "SPK_<language>_<state>_<counter>"
        spk = f"SPK_{l1}_{st}_{ctr[key]:04d}"
        spk_list.append(spk)  # Append the speaker ID to the list

    return spk_list

def pick_domain(r):
    """
    Pick the domain for each sample based on the first available field in DOMAIN_PRIORITY.
    If both fields are empty, it returns "general".
    """
    for c in DOMAIN_PRIORITY:
        v = r.get(c, "")  # Check if the field is not empty
        if v:
            return v  # Return the first non-empty domain
    return "general"  # Return "general" if no domain is found

def to_jsonl_item(r, speaker_id):
    """
    Convert the metadata of each row into a format suitable for JSONL.
    Each dictionary corresponds to a single audio sample with relevant metadata (file path, speaker ID, transcription, etc.)
    """
    return {
        "audio_filepath": r["path"],  # Path to the audio file
        "duration": float(r["duration"]) if r.get("duration") else None,  # Duration of the audio sample
        "text": r.get("text", ""),  # The transcription of the audio
        "speaker_id": speaker_id,  # The unique speaker ID
        "l1": r.get("primary_language", ""),  # The primary language of the speaker
        "state": r.get("native_place_state", ""),  # The state of the speaker
        "gender": r.get("gender", ""),  # The gender of the speaker
        "domain": pick_domain(r),  # The domain (occupation/job category) of the speaker
    }

def write_jsonl(path: Path, items):
    """
    Write a list of JSONL items (dictionaries) to a `.jsonl` file. Each item is written on a new line in the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    with path.open("w", encoding="utf-8") as w:
        for it in items:
            w.write(json.dumps(it, ensure_ascii=False) + "\n")  # Write each item as a JSON line

def main():
    # 1) Read data from the CSV file and convert it into a list of dictionaries
    rows = read_csv_rows(CSV_PATH)
    if not rows:
        raise SystemExit(f"No rows found in {CSV_PATH}")  # Exit if the CSV is empty

    # 2) Generate synthetic speaker IDs for each row
    spk_ids = make_speaker_ids(rows)

    # 3) Convert the data into a JSONL-compatible format
    items = [to_jsonl_item(r, spk) for r, spk in zip(rows, spk_ids)]

    # 4) If we only want a single test.jsonl file, write it
    if not Make_Splits:
        write_jsonl(OUT_DIR / "test.jsonl", items)  # Write all data to a single test.jsonl file
        print(f"✅ Wrote {len(items)} items to {OUT_DIR/'test.jsonl'}")
        return

    # ===== Stratify the splits based on speaker IDs =====
    # Group the items by speaker ID to prevent data leakage (no overlap of speakers between splits)
    by_spk = defaultdict(list)
    for it in items:
        by_spk[it["speaker_id"]].append(it)

    spk_groups = list(by_spk.values())  # List of groups of items by speaker ID

    # Define the stratification label based on (language, state)
    def label(group):
        it = group[0]  # Use the first item in the group as the label
        return f"{it.get('l1','')}|{it.get('state','')}"

    y = [label(g) for g in spk_groups]  # List of labels for each group

    # Split the data into test and rest sets using stratification
    try:
        rest_groups, test_groups = train_test_split(
            spk_groups, test_size=Test_size, random_state=42, stratify=y
        )
    except ValueError:
        rest_groups, test_groups = train_test_split(
            spk_groups, test_size=Test_size, random_state=42
        )

    # Split the remaining data into train and dev sets
    y_rest = [label(g) for g in rest_groups]
    try:
        train_groups, dev_groups = train_test_split(
            rest_groups,
            test_size=Validation_Rest,
            random_state=42,
            stratify=y_rest if len(set(y_rest)) > 1 else None,
        )
    except ValueError:
        train_groups, dev_groups = train_test_split(
            rest_groups, test_size=Validation_Rest, random_state=42
        )

    # Flatten the groups into individual items (for train, dev, and test)
    def flatten(groups):
        for g in groups:
            for it in g:
                yield it

    train_items = list(flatten(train_groups))
    dev_items   = list(flatten(dev_groups))
    test_items  = list(flatten(test_groups))

    # 5) Write the split data to JSONL files
    write_jsonl(OUT_DIR / "train.jsonl", train_items)
    write_jsonl(OUT_DIR / "dev.jsonl", dev_items)
    write_jsonl(OUT_DIR / "test.jsonl", test_items)

    # 6) Report the counts of items in each split
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
