from __future__ import annotations

from transformers import pipeline


class Summarizer:
    """Text summarization wrapper for Arabic/multilingual transcripts."""

    def __init__(self, model_name: str) -> None:
        self._pipe = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_length: int = 120, min_length: int = 30) -> str:
        if not text.strip():
            return ""
        result = self._pipe(text, max_length=max_length, min_length=min_length, do_sample=False)
        return str(result[0]["summary_text"]).strip()
