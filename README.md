# Accent Benchmark

This project benchmarks Automatic Speech Recognition (ASR) models on Indian English speech.  
It uses the Svarah dataset and computes metrics such as Word Error Rate (WER) and Character Error Rate (CER) for models like OpenAI Whisper.

## Structure

data/raw/         → original parquet files    

data/out_wavs/    → exported 16kHz mono wavs + manifest.csv  

data/manifests/   → train/dev/test jsonl manifests  

eval/             → results and metrics  

scripts/          → data prep + evaluation scripts  

## Pipeline

### Data preparation

Convert audio to 16kHz mono wav

Export metadata to manifest.csv

Build train/dev/test jsonl splits

### Evaluation

Run Whisper (small/medium/large-v2)

Compute WER and CER using jiwer

Save outputs to eval/results/

## Initial Result (Whisper Medium, Svarah test split)

| Model                        | **WER** | **CER** | **Version**               | **Additional Information**                                                  |
|------------------------------|---------|---------|---------------------------|------------------------------------------------------------------------------|
| **Whisper Medium (This Project)** | 7.8%    | 3.6%    | 20250625 (June 2025)       | Compare with Svarah paper's results.                                          |
| **Svarah Paper (2023)**       | 8.3%    | N/A | N/A                       | [Link to Svarah Paper Results](https://github.com/AI4Bharat/Svarah?tab=readme-ov-file#table-1-wer-comparison) |

### Key Observations:
- **WER for Whisper Medium** (7.8%) shows a **slight improvement** over the **Svarah paper's 8.3%**.
- This indicates that a resource intensive model like Whisper Medium has not had a significant improvement over the last 2 years in terms of WER on Indian English speech.
- This suggests that further research and development is needed to achieve substantial gains in ASR performance for Indian English.

