from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import librosa

from audio_understanding.asr.whisper_asr import WhisperASR
from audio_understanding.utils.text import normalize_arabic_text


@dataclass
class SimpleAnalysisOutput:
    transcript: str
    duration_sec: float
    word_count: int
    unique_word_count: int
    top_words: list[tuple[str, int]]
    keyword_hits: dict[str, int]


class SimpleAudioPipeline:
    """Minimal flow: audio -> transcript -> basic text analysis."""

    def __init__(self, model_name: str = "openai/whisper-small") -> None:
        self.asr = WhisperASR(model_name=model_name)

    def process(self, audio_path: str, keywords: list[str] | None = None) -> SimpleAnalysisOutput:
        transcript = self.asr.transcribe(audio_path)
        duration = self._get_duration(audio_path)

        normalized_text = normalize_arabic_text(transcript)
        words = [w for w in normalized_text.split(" ") if w]
        counts = Counter(words)

        keyword_hits: dict[str, int] = {}
        for keyword in keywords or []:
            normalized_keyword = normalize_arabic_text(keyword)
            if not normalized_keyword:
                continue
            hit_count = normalized_text.count(normalized_keyword)
            if hit_count > 0:
                keyword_hits[keyword] = hit_count

        return SimpleAnalysisOutput(
            transcript=transcript,
            duration_sec=duration,
            word_count=len(words),
            unique_word_count=len(counts),
            top_words=counts.most_common(10),
            keyword_hits=keyword_hits,
        )

    @staticmethod
    def _get_duration(audio_path: str) -> float:
        return float(librosa.get_duration(path=audio_path))