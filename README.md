# Accent Benchmark

**Benchmarking Automatic Speech Recognition (ASR) Models on Indian English Speech**

This project evaluates state-of-the-art ASR models, such as **OpenAI Whisper**, on the **Svarah dataset**, computing performance metrics like **Word Error Rate (WER)** and **Character Error Rate (CER)**.



##  Project Structure

| Folder                  | Description |
|-------------------------|-------------|
| `data/raw/`             | Original Parquet files containing audio data |
| `data/out_wavs/`        | Exported 16kHz mono WAV files and `manifest.csv` |
| `data/manifests/`       | Train/dev/test JSONL manifests |
| `eval/results`          | Evaluation results, WER/CER metrics |
| `eval/scripts/`         | Data preparation and evaluation scripts |



##  Pipeline Overview

### 1. Data Preparation
- Convert audio to **16kHz mono WAV** format for consistency.
- Export metadata to **`manifest.csv`**.
- Build **train/dev/test JSONL splits** for model evaluation.

### 2️. Model Evaluation
- Run currently **Whisper models** (medium/) on test data.
- Compute **WER** and **CER** using **`jiwer`**.
- Save outputs to **`eval/results/`** for analysis.



##  Key Features
- Handles **audio preprocessing** (resampling, mono conversion) for ASR pipelines.
- Supports **benchmarking multiple models** in a consistent manner.
- Produces **quantitative performance metrics** for comparison.



## Initial Result (Whisper Medium, Svarah test split)

| Model                        | **WER** | **CER** | **Version**               | **Additional Information**                                                  |
|------------------------------|---------|---------|---------------------------|------------------------------------------------------------------------------|
| **Whisper Medium (This Project)** | 7.8%    | 3.6%    | 20250625 (June 2025)       | Compare with Svarah paper's results.                                          |
| **Svarah Paper (2023)**       | 8.3%    | N/A | N/A                       | [Link to Svarah Paper Results](https://github.com/AI4Bharat/Svarah?tab=readme-ov-file#table-1-wer-comparison) |

### Key Observations:
- **WER for Whisper Medium** (7.8%) shows a **slight improvement** over the **Svarah paper's 8.3%**.
- This indicates that a resource intensive model like Whisper Medium has not had a significant improvement over the last 2 years in terms of WER on Indian English speech.
- This suggests that further research and development is needed to achieve substantial gains in ASR performance for Indian English.

---

## Student Notes / Reflection

- **Project Type**: Personal project 
- **Objective**: Benchmark state-of-the-art ASR models on open-source datasets to understand performance on accented English.  

### Key Learnings:
- Gained practical experience in **data preprocessing for audio** (resampling, mono conversion, handling parquet datasets).  
- Learned to work with **pre-trained ML models** (Whisper) and evaluate their performance using **WER and CER** metrics.  
- Understood challenges in **ASR for non-native or accented speech**, including **stagnation in model performance** over multiple years.  

### Challenges Overcome:
- Managing **large audio datasets** efficiently.  
- Converting **heterogeneous audio formats** into a uniform pipeline-ready format.  
- Integrating multiple **ML libraries and frameworks** (Hugging Face, PyTorch, librosa, soundfile).  

### Skills Developed:
- **Programming**: Python (NumPy, Pandas, PyTorch), scripting, file handling.  
- **Machine Learning**: Model evaluation, benchmarking, metric computation.  
- **Data Engineering**: Handling Parquet files, creating manifests, splitting datasets for train/dev/test.  
- **Analytical Thinking**: Interpreting model performance trends, comparing with literature (Svarah paper).  

---

## Acknowledgements

This project uses the **Svarah dataset** (AI4Bharat, 2023) for benchmarking ASR models.  
I thank the authors for making the dataset publicly available and providing detailed evaluation metrics.

- **Dataset Repository**: [https://huggingface.co/datasets/ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)  [https://github.com/AI4Bharat/Svarah](https://github.com/AI4Bharat/Svarah)  
- **Paper / Reference**: Javed et al., *Interspeech 2023*, [Link to Paper](https://www.isca-speech.org/archive/Interspeech_2023/abstracts/xxx.html)
