from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    asr_backend: str = "whisper"
    whisper_model: str = "openai/whisper-small"
    wav2vec_model: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    summary_model: str = "csebuetnlp/mT5_multilingual_XLSum"
    emotion_model: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"


@dataclass
class PipelineConfig:
    enable_summary: bool = True
    enable_search: bool = True
    enable_speaker_id: bool = True
    enable_emotion: bool = True
    speaker_count: int = 2
    chunk_seconds: float = 2.0
    default_keywords: list[str] = field(default_factory=lambda: ["emergency", "deadline", "exam", "طوارئ", "امتحان"])


@dataclass
class AppConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def load_config(path: str | Path) -> AppConfig:
    file_path = Path(path)
    if not file_path.exists():
        return AppConfig()

    with file_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    models = ModelConfig(**raw.get("models", {}))
    pipeline = PipelineConfig(**raw.get("pipeline", {}))
    return AppConfig(models=models, pipeline=pipeline)
