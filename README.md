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

## Example Result (Whisper Medium, Svarah test split)
WER = 7.8%
CER = 3.6%
