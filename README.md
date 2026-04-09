# Deep Learning Based Arabic Audio Understanding and Retrieval System

This repository implements an intelligent audio analysis system centered on Arabic
speech recognition with optional advanced analytics.

Architecture note: this project follows a full modular architecture similar to large
end-to-end designs, but uses Streamlit as the interface/runtime layer instead of a
separate React frontend and FastAPI backend.

## 1) Project Overview

Main objective:

Audio Input -> Speech Recognition (ASR) -> Text Transcript

Target use cases:

- Meeting assistants
- Lecture transcription
- Podcast search engines
- Call center analytics
- Voice assistants

## 2) Main Tasks

### Task 1: Speech-to-text

- Deep learning ASR with multiple options:
	- Whisper (`openai/whisper-small` by default)
	- Wav2Vec2 Arabic (`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`)
	- Custom CNN + BiLSTM + CTC model (training script included)

### Optional advanced tasks (included)

- Transcript summarization
- Semantic transcript search
- Speaker identification
- Emotion detection
- Keyword spotting

## 3) Recommended Arabic Datasets

- Mozilla Common Voice Arabic: https://commonvoice.mozilla.org/en
- Arabic Speech Corpus: https://en.arabicspeechcorpus.com/
- MASC Arabic Speech Dataset: https://huggingface.co/datasets/hirundo-io/MASC
- Arabic Broadcast News Dataset: https://catalog.ldc.upenn.edu/LDC2006S46
- EJUST dataset (private)

Recommended primary dataset for ASR baseline in this repo: Common Voice Arabic.

## 4) Folder Structure

```text
.
├── app/
│   └── streamlit_app.py
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── models/
│   └── index/
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   └── experiments_template.md
├── scripts/
│   ├── data/
│   │   ├── download_common_voice.py
│   │   └── prepare_data.py
│   ├── training/
│   │   └── train_ctc.py
│   ├── evaluation/
│   │   └── evaluate_asr.py
│   └── retrieval/
│       └── build_search_index.py
├── src/
│   └── audio_understanding/
│       ├── advanced/
│       ├── asr/
│       ├── utils/
│       ├── config.py
│       └── pipeline.py
├── tests/
│   └── test_keyword_spotting.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 5) Setup

```bash
cd "/home/ibrahimeldesouky/Programming/projects/Arabic audio understanding-speech to text"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 6) Run Demo Interface

```bash
streamlit run app/streamlit_app.py
```

This Streamlit interface is the official UI for this project.

The interface supports:

1. Uploading audio or selecting from `data/`
2. Playing audio in-browser
3. ASR transcription
4. Optional summary/search/speaker/emotion analysis
5. Keyword spotting and optional WER if reference text is provided

## 7) Training / Evaluation Scripts

```bash
python scripts/data/download_common_voice.py --split train --max-samples 500
python scripts/data/prepare_data.py --metadata data/processed/common_voice_ar/metadata.tsv
python scripts/training/train_ctc.py --manifest data/processed/train_manifest.csv --epochs 3
python scripts/evaluation/evaluate_asr.py --audio path/to/audio.wav --reference "النص المرجعي" --backend whisper --model openai/whisper-small
```

## 8) Evaluation Metric

Primary metric for ASR is Word Error Rate (WER):

$$
WER = \frac{S + D + I}{N}
$$

Where:

- $S$: substitutions
- $D$: deletions
- $I$: insertions
- $N$: number of words in reference text

## 9) Deliverables Coverage

This repository now contains all expected deliverables:

1. Source code (`src/`, `scripts/`, `app/`)
2. Dataset description (`docs/dataset.md`)
3. Architecture documentation (`docs/architecture.md`)
4. Experiment template (`docs/experiments_template.md`)
5. Evaluation scripts and metrics (`scripts/evaluation/evaluate_asr.py`, `src/audio_understanding/utils/metrics.py`)
6. Demo interface (`app/streamlit_app.py`)

## 10) Interface Choice

To match the project requirements and keep deployment simple:

1. Streamlit is used as the interactive interface.
2. Pipeline execution happens directly in Python modules.
3. No React or FastAPI layers are required for the assignment submission.
