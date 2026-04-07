from __future__ import annotations

from jiwer import wer


def compute_wer(reference_text: str, predicted_text: str) -> float:
    """Compute Word Error Rate (WER). Lower is better."""
    return float(wer(reference_text, predicted_text))
