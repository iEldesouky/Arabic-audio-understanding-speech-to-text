# System Architecture

## 1. High-Level Objective

Build an intelligent Arabic audio understanding pipeline where speech is converted to text,
then optional downstream analytics are applied.

Core path:

Audio Input -> ASR -> Transcript

Extended path:

Transcript -> Summary -> Search
Audio -> Speaker Identification
Audio -> Emotion Detection
Transcript -> Keyword Spotting

## 2. Modular Architecture

### 2.1 Interface Layer (Streamlit)

- File: app/streamlit_app.py
- Responsibilities:
   - Accept uploaded audio or select from data folder
   - Play audio in browser
   - Expose model/task settings
   - Trigger pipeline execution
   - Render transcript, WER, and advanced outputs

### 2.2 Orchestration Layer

- File: src/audio_understanding/pipeline.py
- Main class: AudioUnderstandingPipeline
- Responsibilities:
   - Select ASR backend (Whisper or Wav2Vec2)
   - Execute optional advanced modules based on configuration
   - Return unified output object for UI rendering

### 2.3 ASR Layer

- Files:
   - src/audio_understanding/asr/whisper_asr.py
   - src/audio_understanding/asr/wav2vec_asr.py
   - src/audio_understanding/asr/cnn_lstm_ctc.py
- Responsibilities:
   - Inference wrappers for pretrained models
   - Trainable custom CNN + BiLSTM + CTC model

### 2.4 Advanced Analysis Layer

- Files under: src/audio_understanding/advanced/
   - summarization.py
   - search.py
   - speaker_id.py
   - emotion.py
   - keyword_spotting.py

### 2.5 Utilities and Evaluation Layer

- Files:
   - src/audio_understanding/utils/text.py
   - src/audio_understanding/utils/metrics.py
- Responsibilities:
   - Arabic text normalization/chunking
   - WER computation

### 2.6 Data and Experiment Layer

- Data directories:
   - data/raw
   - data/processed
   - data/models
   - data/index
- Scripts:
   - scripts/data/download_common_voice.py
   - scripts/data/prepare_data.py
   - scripts/training/train_ctc.py
   - scripts/evaluation/evaluate_asr.py
   - scripts/retrieval/build_search_index.py

## 3. Module Dependency Graph

1. Streamlit UI calls AudioUnderstandingPipeline
2. Pipeline calls one ASR backend
3. Pipeline optionally calls advanced modules
4. Advanced modules use shared text utilities
5. UI computes WER through metrics utility

## 4. Interface Decision

This project intentionally uses Streamlit as both:

1. Presentation layer (frontend UI)
2. Execution trigger layer (instead of separate FastAPI service)

Why:

1. Faster iteration for deep learning experiments
2. Lower integration overhead for student project scope
3. Easier reproducibility with a single Python stack

## 5. Training and Inference Strategy

### 5.1 Inference (demo path)

1. User selects audio in Streamlit
2. Pipeline runs ASR
3. Optional modules run in sequence
4. Results displayed interactively

### 5.2 Training (offline path)

1. Download/prep datasets (Common Voice Arabic baseline)
2. Build manifest CSV with audio-text pairs
3. Train CNN+BiLSTM+CTC model
4. Save checkpoint under data/models
5. Evaluate with WER

## 6. Deliverables Mapping

1. Source code: app/, src/, scripts/
2. Dataset description: docs/dataset.md
3. Architecture: docs/architecture.md
4. Experiments: docs/experiments_template.md
5. Evaluation: scripts/evaluate_asr.py and utils/metrics.py
6. Demo interface: app/streamlit_app.py
