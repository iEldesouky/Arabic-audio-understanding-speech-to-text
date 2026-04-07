from __future__ import annotations

import re


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_PUNCTUATION_SPLIT_RE = re.compile(r"[\.!?؟\n]+")


def normalize_arabic_text(text: str) -> str:
    """Normalize Arabic text for robust matching and retrieval."""
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def split_sentences(text: str) -> list[str]:
    """Split transcript into sentence-like chunks for indexing."""
    parts = [p.strip() for p in _PUNCTUATION_SPLIT_RE.split(text)]
    return [p for p in parts if p]
