import os
from pathlib import Path
import io # for in-memory byte streams
import numpy as np 
import soundfile as sf # for reading/writing audio files
from tqdm import tqdm # progress bar
from datasets import load_dataset, Audio # for loading datasets and audio
import librosa # for high-quality audio resampling
# librosa is good for resqampling and is used in many ASR projects

# Set the path to the needed directories
ParquetDataDir = Path("data/raw/Svarah/data")
OutputDataDir= Path("out_wavs_johan")

# Settings

# Set to 16000 to resample to 16kHz, or None to keep original Sameple Rate
TARGET_SR = 16000  

# Does not matter here because there is only one split in the Parquet files for Svarah (Hugging Face Datasets)
SPLIT = "train"  

# Columns in the file 
DataCols = [
    "text", "gender", "age-group", "primary_language",
    "native_place_state", "native_place_district",
    "highest_qualification", "job_category", "occupation_domain",
    "duration"  
]


def make_mono(x: np.ndarray) -> np.ndarray:
    """Make audio mono by averaging channels if needed."""
    
    # tqdm.write because the progress bar is then not messed up
    tqdm.write(f"Before Mono: Channels={x.ndim}")
    
    # Ensure x is a NumPy array
    # Because soundfile may return lists or other types
    x = np.asarray(x)
    if x.ndim == 2:  # if stereo
        
        # After reading audio, a stereo (or multi‑channel) signal can come in either shape:
        # (channels, samples) e.g. (2, 48000) (library returns channel-first)
        # or
        # (samples, channels) e.g. (48000, 2) (library returns channel-last)
        
        # We check which dimension is smaller to determine which is channels and which is samples
        # Channels are usually fewer than samples
        if x.shape[0] < x.shape[1]:  # (channels, samples)
        # mean gives (L + R) / 2
            
            # Average across channels (axis 0) to make mono
            # For example, if shape is (2, 48000), mean(axis=0) gives (48000,)
            x = x.mean(axis=0)
            
        else:  # (samples, channels)
            
            # Average across channels (axis 1) to make mono
            # For example, if shape is (48000, 2), mean(axis=1) gives (48000,)
            x = x.mean(axis=1)
            
    # tqdm.write because the progress bar is then not messed up                
    tqdm.write(f"After Mono: Channels={x.ndim}")
    return x.astype(np.float32, copy=False)

def csv_escape(val):
    """Escape a value for CSV output, handling quotes and None."""
    
    # If the value is None, return an empty string
    if val is None:
        return ""
    
    # Convert the value to string
    s = str(val)
    
    # Escape double quotes by doubling them, and wrap the whole string in double quotes
    return '"' + s.replace('"', '""') + '"'


def main():
    
    # Check and create output directory if it doesn't exist
    OutputDataDir.mkdir(parents=True, exist_ok=True)
    
    # List all .parquet files in the folder
    parquet_files = list(ParquetDataDir.glob("*.parquet"))
    
    # If no parquet files found, raise an error
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {ParquetDataDir}. Please check the directory.")
    
    # Load dataset from the Parquet files
    ds = load_dataset("parquet", data_files=str(ParquetDataDir / "*.parquet"), split=SPLIT)
    
    # audio_filepath is checked if it exists in the dataset
    if "audio_filepath" not in ds.column_names:
        raise ValueError(f"'audio_filepath' column not found in the dataset.")

    # Force the dataset to not decode audio immediately
    # So we can handle resampling and mono conversion ourselves
    # Avoiding torchcodec. which is automatically used by Audio()
    ds = ds.cast_column("audio_filepath", Audio(decode=False))
    
    # Prepare the manifest file for CSV output
    manifest_path = OutputDataDir / "manifest.csv"
    
    # Write the header to the manifest file
    # 'w' mode to overwrite if it exists
    # utf-8 ensures all Unicode characters in later text (e.g. accents) are preserved.
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        
        # Name the header rows in the manifest file
        # Keep the order: path, duration, text, then other metadata columns
        # Guarantees "text" is always 3rd—many ASR loaders expect that position
        header = ["path", "duration", "text"]+ [c for c in DataCols if c not in ("text",)]
        manifest_file.write(",".join(header) + "\n")
        
        
        # tqdm progress bar for visual feedback during processing
        # Pattern: tqdm(iterable, desc="label")
        for i in tqdm(range(len(ds)), desc=f"Exporting {SPLIT}"):
            
            # Get the current row from the dataset
            # row = the whole record (all columns) for item i.
            # Access audio data from the row, for example: text, gender, etc.
            row = ds[i]
            
            # Access the audio data from the row 
            # Sepeated into variable 'a' for clarity and easier debugging
            a = row["audio_filepath"]  
            
            # wav_data will hold the actual audio samples as a NumPy array
            wav_data = None
            
            # sr will hold the sample rate of the audio
            sr = None  
            
            # "a" is expected dict that contains metadata about the audio
            # It may contain 'path' (where it is stored) 
            # and/or 'bytes' (the actual in-memory audio data)
            # Checks if "a" is a dict 
            # Checks if "a" contains a key named 'path'
            # and if the path exists on disk
            if isinstance(a, dict) and a.get("path") and os.path.exists(a["path"]):
               
                # Read from path on disk
                # False so that audio data is not forced into two dimensions when reading stereo files
                # Because we handle mono/stereo ourselves
                data, sr = sf.read(a["path"], always_2d=False)
                
                # Convert to NumPy array of type float32 for consistency
                wav_data = np.asarray(data, dtype=np.float32)
                
            # If 'a' is a dict and contains 'bytes' (the actual in-memory audio data), read from in-memory bytes
            elif isinstance(a, dict) and a.get("bytes"):
                
                # Read from bytes (in-memory)
                # BytesIO is a Python class that allows binary data (in-memory bytes) to be treated like a file. 
                # By wrapping the audio bytes in BytesIO, we can pass it to soundfile.read as if it were a file.
                bio = io.BytesIO(a["bytes"])
                
                # Read the audio data and sample rate from the in-memory bytes
                # False so that audio data is not forced into two dimensions when reading stereo files
                # Because we handle mono/stereo ourselves   
                data, sr = sf.read(bio, always_2d=False)
                
                # Convert to NumPy array of type float32 for consistency
                wav_data = np.asarray(data, dtype=np.float32)
            
            # If neither is found, raise an error indicating that no audio source was provided.
            else:
                raise FileNotFoundError(
                    f"Row {i}: audio has neither a valid local path nor bytes. Got: {a}"
                )
            
            # Make mono if needed
            wav_data = make_mono(wav_data)
            
            # Resample to TARGET_SR if specified
            if TARGET_SR is not None and sr != TARGET_SR:
                wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR
            # tqdm.write because the progress bar is then not messed up    
            tqdm.write(f"Resampled to {sr} Hz")
            
            # Save the processed audio as a WAV file in the output directory
            # Use 16-bit PCM format for compatibility with most ASR tools
            # i:08d gives zero-padded filenames like 00000001.wav. Here, 8 because we may have many files.
            out_wav = OutputDataDir / f"{i:08d}.wav"
            sf.write(str(out_wav), wav_data, sr, subtype="PCM_16")
            
            # Duration from sample count
            duration_sec = len(wav_data) / float(sr)
            
            # Build the manifest row with required fields
            # Start with path, duration, and text
            text_val = row.get("text", "")
            
            # as_posix() gives a consistent path format with forward slashes
            # which is generally preferred in CSVs and cross-platform
            # .3f formats duration to 3 decimal places
            fields = [out_wav.as_posix(), f"{duration_sec:.3f}", csv_escape(text_val)]
            
            # Add other metadata columns if they exist in the row
            for c in DataCols:
                if c == "text":
                    continue              
                # 
                fields.append(csv_escape(row.get(c, "")))
            # Write the row to the manifest file
            manifest_file.write(",".join(fields) + "\n")
    
    print(f"\nDone!!!  \nWAVs in: {OutputDataDir}")
    print(f"Manifest: {manifest_path}")
    
if __name__ == "__main__":
    main()