from __future__ import annotations

from transformers import pipeline


class Summarizer:
    """Light wrapper for transcript summarization."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipe = None

    def summarize(self, text: str, max_length: int = 120, min_length: int = 30) -> str:
        if not text.strip():
            return ""

        if self._pipe is None:
            try:
                self._pipe = pipeline("summarization", model=self.model_name)
            except Exception:
                # Fallback keeps the app working if model download fails.
                words = text.split()
                return " ".join(words[: min(80, len(words))])

        result = self._pipe(text, max_length=max_length, min_length=min_length, do_sample=False)
        return str(result[0]["summary_text"]).strip()
