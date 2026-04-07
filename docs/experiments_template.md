# Experiments and Results Template

## Experiment Matrix
| Exp ID | ASR Model | Dataset Split | Advanced Tasks Enabled | WER | Notes |
|---|---|---|---|---:|---|
| E1 | Whisper Small | CV Arabic test | None |  | Baseline ASR only |
| E2 | Wav2Vec2 Arabic | CV Arabic test | None |  | Compare ASR backbone |
| E3 | CNN+LSTM+CTC | CV Arabic test | None |  | Custom model |
| E4 | Best ASR | CV Arabic test | Summary + Search + Speaker + Emotion + Keywords |  | Full pipeline |

## Evaluation Metrics
- Main metric: Word Error Rate (WER)
- Optional metrics:
  - CER (Character Error Rate)
  - Latency per audio minute
  - Throughput (samples/sec)

## Analysis Prompts
- Which ASR backbone is most accurate for Arabic in your setup?
- How robust are predictions on noisy audio?
- Which advanced tasks add value for your target application (lecture, podcast, call center)?
