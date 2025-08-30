import os
from pathlib import Path
import io
import numpy as np
import soundfile as sf
from tqdm import tqdm
from datasets import load_dataset, Audio

# Set the path to the needed directories
ParquetDataDir = Path("data/raw/Svarah/data")
OutputDataDir= Path("out_wavs_johan")

DataCols = [
    "text", "gender", "age-group", "primary_language",
    "native_place_state", "native_place_district",
    "highest_qualification", "job_category", "occupation_domain",
    "duration"  
]

def main():
    