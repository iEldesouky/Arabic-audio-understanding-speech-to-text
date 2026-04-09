# Experiments Template

## 1. Experiment metadata

- Date:
- Research question:
- Dataset version:
- Model:
- Hyperparameters:

## 2. Pipeline scope

- Core ASR only or full pipeline:
- Modules enabled:
	- Summarization:
	- Search:
	- Speaker ID:
	- Emotion:
	- Keyword spotting:

## 3. Training details

- Epochs:
- Batch size:
- Learning rate:
- Optimizer:
- Hardware:

## 4. Data setup

- Train split size:
- Validation split size:
- Test split size:
- Audio sampling rate:
- Preprocessing steps:
- Augmentation used:

## 5. Evaluation

- WER:
- CER (optional):
- Inference latency (sec/sample):
- Memory usage (optional):
- Notes on failure cases:

## 6. Qualitative analysis

- Good examples:
- Bad examples:
- Arabic dialect/noise observations:

## 7. Ablation and comparisons

- ASR model comparison (Whisper vs Wav2Vec2 vs CNN+BiLSTM+CTC):
- Data-size scaling impact:
- Augmentation impact:
- Optional module impact on latency:

## 8. Conclusion

- Best model so far:
- Next change to try:
