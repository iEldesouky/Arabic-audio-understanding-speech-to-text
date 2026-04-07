from __future__ import annotations

from dataclasses import dataclass

import librosa
from transformers import pipeline


@dataclass
class EmotionPrediction:
    label: str
    score: float


class EmotionDetector:
    """Audio emotion classifier using a pretrained transformer model."""

    def __init__(self, model_name: str) -> None:
        self._pipe = pipeline("audio-classification", model=model_name)

    def predict(self, audio_path: str, top_k: int = 3) -> list[EmotionPrediction]:
        audio, sr = librosa.load(audio_path, sr=16_000)
        results = self._pipe({"array": audio, "sampling_rate": sr}, top_k=top_k)
        return [
            EmotionPrediction(label=str(item["label"]), score=float(item["score"]))
            for item in results
        ]
