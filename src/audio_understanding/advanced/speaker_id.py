from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class SpeakerSegment:
    speaker_label: str
    start_sec: float
    end_sec: float


class SpeakerIdentifier:
    """Chunk-level unsupervised speaker segmentation using MFCC + KMeans."""

    def detect_speakers(self, audio_path: str, chunk_seconds: float = 2.0, n_speakers: int = 2) -> list[SpeakerSegment]:
        audio, sr = librosa.load(audio_path, sr=16_000)
        chunk_size = int(chunk_seconds * sr)
        if chunk_size <= 0 or len(audio) < chunk_size:
            return []

        feats: list[np.ndarray] = []
        bounds: list[tuple[float, float]] = []
        for start in range(0, len(audio) - chunk_size + 1, chunk_size):
            end = start + chunk_size
            chunk = audio[start:end]
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            feats.append(mfcc.mean(axis=1))
            bounds.append((start / sr, end / sr))

        if len(feats) < n_speakers:
            return []

        labels = KMeans(n_clusters=n_speakers, random_state=42, n_init=10).fit_predict(np.asarray(feats))

        segments: list[SpeakerSegment] = []
        for (start_sec, end_sec), label in zip(bounds, labels):
            segments.append(SpeakerSegment(speaker_label=f"speaker_{int(label) + 1}", start_sec=start_sec, end_sec=end_sec))
        return segments
