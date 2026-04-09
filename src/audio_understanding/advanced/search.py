from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from audio_understanding.utils.text import normalize_arabic_text


@dataclass
class SearchResult:
    text: str
    score: float


class TranscriptSearchEngine:
    """Semantic retrieval on transcript chunks."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, chunks: list[str]) -> None:
        self.chunks = [c for c in chunks if c.strip()]
        if not self.chunks:
            self.embeddings = None
            return
        normalized = [normalize_arabic_text(c) for c in self.chunks]
        self.embeddings = self.model.encode(normalized, convert_to_numpy=True, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.embeddings is None or not self.chunks:
            return []

        query_vec = self.model.encode([normalize_arabic_text(query)], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec
        order = np.argsort(-scores)[:top_k]
        return [SearchResult(text=self.chunks[i], score=float(scores[i])) for i in order]
