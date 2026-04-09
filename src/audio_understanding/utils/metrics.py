from __future__ import annotations

from jiwer import wer


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate between two transcripts."""
    if not reference.strip() and not hypothesis.strip():
        return 0.0
    if not reference.strip():
        return 1.0
    return float(wer(reference, hypothesis))
