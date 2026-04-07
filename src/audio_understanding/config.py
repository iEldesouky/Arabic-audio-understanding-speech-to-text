from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    asr_model: str
    asr_backend: str
    summary_model: str
    search_embedding_model: str
    emotion_model: str
    speaker_model: str


@dataclass
class PipelineConfig:
    chunk_seconds: float
    speaker_count: int
    default_keywords: list[str]


@dataclass
class AppConfig:
    models: ModelConfig
    pipeline: PipelineConfig


_DEFAULT_CONFIG = {
    "models": {
        "asr_model": "openai/whisper-small",
        "asr_backend": "whisper",
        "summary_model": "csebuetnlp/mT5_multilingual_XLSum",
        "search_embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "emotion_model": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "speaker_model": "speechbrain/spkrec-ecapa-voxceleb",
    },
    "pipeline": {
        "chunk_seconds": 2.0,
        "speaker_count": 2,
        "default_keywords": ["طوارئ", "موعد", "امتحان", "deadline", "emergency", "exam"],
    },
}


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load YAML config or fallback to defaults."""
    config_data: dict[str, Any] = _DEFAULT_CONFIG

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

    models = ModelConfig(**config_data["models"])
    pipeline = PipelineConfig(**config_data["pipeline"])
    return AppConfig(models=models, pipeline=pipeline)
