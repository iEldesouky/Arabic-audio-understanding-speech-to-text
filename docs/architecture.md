# System Architecture

```mermaid
flowchart TD
    A[Audio Input] --> B[ASR: Whisper or Wav2Vec2 or CNN+LSTM+CTC]
    B --> C[Text Transcript]
    C --> D[Summarization]
    C --> E[Semantic Search Index]
    C --> F[Keyword Spotting]
    A --> G[Speaker Identification]
    A --> H[Emotion Detection]

    D --> I[Streamlit UI]
    E --> I
    F --> I
    G --> I
    H --> I
    C --> I
```

## Core Notes
- The system separates speech recognition from downstream NLP/audio analytics.
- This allows benchmarking multiple ASR engines using the same evaluation and UI.
- Advanced tasks can be turned on/off independently for ablation experiments.
