from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    text: str
    score: float


class TranscriptSearchEngine:
    """Embedding-based semantic search over transcript chunks."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)
        self._chunks: list[str] = []
        self._embeddings: np.ndarray | None = None

    def build_index(self, chunks: list[str]) -> None:
        self._chunks = chunks
        if not chunks:
            self._embeddings = None
            return
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        self._embeddings = np.array(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self._embeddings is None or len(self._chunks) == 0:
            return []
        query_vec = np.array(self.model.encode([query], normalize_embeddings=True))[0]
        scores = self._embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(text=self._chunks[i], score=float(scores[i])) for i in top_indices]
