# Deep Learning Based Arabic Audio Understanding and Retrieval

End-to-end intelligent audio analysis system:
- Speech to text (Arabic ASR)
- Transcript summarization
- Semantic transcript search
- Speaker identification
- Emotion detection
- Keyword spotting
- Streamlit demo interface

## 1) Project Overview
This project implements the assignment pipeline:

Audio Input -> ASR -> Text Transcript

Then extends it with advanced tasks:
- Speech -> Text -> Summary -> Search
- Speaker analysis
- Emotion recognition
- Keyword detection

Typical use cases:
- Meeting assistants
- Lecture transcription and search
- Podcast indexing and summarization
- Call-center analytics
- Voice assistants

## 2) Models Included
### ASR options
- Whisper (recommended strong baseline): `openai/whisper-small`
- Wav2Vec2 Arabic: `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`
- Custom model for coursework: CNN + BiLSTM + CTC (`scripts/train_ctc.py`)

### Advanced tasks
- Summarization: multilingual mT5 XLSum
- Search: multilingual sentence-transformer embeddings
- Speaker ID: SpeechBrain ECAPA embeddings + clustering
- Emotion detection: transformer audio classifier (configurable)
- Keyword spotting: normalized Arabic text matching

## 3) Folder Structure
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
│   ├── download_common_voice.py
│   ├── prepare_data.py
│   ├── train_ctc.py
│   ├── evaluate_asr.py
│   └── build_search_index.py
├── src/
│   └── audio_understanding/
│       ├── asr/
│       ├── advanced/
│       ├── utils/
│       ├── config.py
│       └── pipeline.py
├── tests/
│   └── test_keyword_spotting.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 4) Installation
```bash
cd "/home/ibrahimeldesouky/Programming/projects/Arabic audio understanding-speech to text"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 5) Quick Start
### Run Streamlit demo
```bash
streamlit run app/streamlit_app.py
```

### Download sample Arabic data (Common Voice)
```bash
python scripts/download_common_voice.py --split train --max-samples 200
python scripts/prepare_data.py --metadata data/processed/common_voice_ar/metadata.tsv
```

### Train custom CNN+LSTM+CTC ASR
```bash
python scripts/train_ctc.py --manifest data/processed/train_manifest.csv --epochs 3
```

### Evaluate ASR with WER
```bash
python scripts/evaluate_asr.py \
  --audio path/to/audio.wav \
  --reference "النص المرجعي هنا" \
  --backend whisper \
  --model openai/whisper-small
```

## 6) Evaluation Metric
Primary metric for ASR: Word Error Rate (WER)

$$
WER = \frac{S + D + I}{N}
$$

Where:
- $S$: substitutions
- $D$: deletions
- $I$: insertions
- $N$: words in reference transcript

Lower WER is better.

## 7) Deliverables Mapping
This repository already contains what you need to submit:
1. Source code: `src/`, `scripts/`, `app/`
2. Dataset description: `docs/dataset.md`
3. System architecture diagram: `docs/architecture.md`
4. Experiments template: `docs/experiments_template.md`
5. Evaluation scripts and metrics: `scripts/evaluate_asr.py`, `src/audio_understanding/utils/metrics.py`
6. Demo interface: `app/streamlit_app.py`

## 8) Notes on Arabic SOTA
For Arabic ASR, practical strong choices are Whisper variants and Arabic-adapted Wav2Vec2 models. In many real setups:
- Whisper is very robust for noisy/mixed audio.
- Arabic-specific Wav2Vec2 can be strong on clean/read speech.
- A custom CNN+LSTM+CTC model is excellent for educational understanding and controlled experiments.

Use your own benchmark experiments (WER on your test split) to decide the final best model for your report.
