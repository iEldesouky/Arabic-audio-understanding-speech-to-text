from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier


@dataclass
class SpeakerSegment:
    start_sec: float
    end_sec: float
    speaker_label: str


class SpeakerIdentifier:
    """Chunk-level speaker grouping using speaker embeddings + clustering."""

    def __init__(self, model_name: str) -> None:
        self.encoder = EncoderClassifier.from_hparams(source=model_name)

    def detect_speakers(
        self,
        audio_path: str,
        chunk_seconds: float = 2.0,
        n_speakers: int = 2,
    ) -> list[SpeakerSegment]:
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        chunk_size = int(chunk_seconds * sample_rate)

        chunks: list[np.ndarray] = []
        times: list[tuple[float, float]] = []
        for start in range(0, len(waveform), chunk_size):
            end = min(start + chunk_size, len(waveform))
            chunk = waveform[start:end]
            if len(chunk) < sample_rate:
                continue
            chunks.append(chunk)
            times.append((start / sample_rate, end / sample_rate))

        if not chunks:
            return []

        embeddings: list[np.ndarray] = []
        for chunk in chunks:
            tensor = torch.tensor(chunk).unsqueeze(0)
            emb = self.encoder.encode_batch(tensor).squeeze().detach().cpu().numpy()
            embeddings.append(emb)

        if len(embeddings) < n_speakers:
            n_speakers = max(1, len(embeddings))

        clustering = AgglomerativeClustering(n_clusters=n_speakers)
        labels = clustering.fit_predict(np.array(embeddings))

        segments: list[SpeakerSegment] = []
        for (start, end), label in zip(times, labels, strict=True):
            segments.append(
                SpeakerSegment(
                    start_sec=float(start),
                    end_sec=float(end),
                    speaker_label=f"Speaker {int(label) + 1}",
                )
            )
        return segments
