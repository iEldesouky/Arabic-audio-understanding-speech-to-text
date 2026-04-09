from __future__ import annotations

from dataclasses import dataclass

from transformers import pipeline


@dataclass
class EmotionPrediction:
    label: str
    score: float


class EmotionDetector:
    """Voice emotion classification wrapper."""

    def __init__(self, model_name: str) -> None:
        self._pipe = pipeline(task="audio-classification", model=model_name)

    def predict(self, audio_path: str, top_k: int = 4) -> list[EmotionPrediction]:
        raw = self._pipe(audio_path, top_k=top_k)
        return [EmotionPrediction(label=str(item["label"]), score=float(item["score"])) for item in raw]
