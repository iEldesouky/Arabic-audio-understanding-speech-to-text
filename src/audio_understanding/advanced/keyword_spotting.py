from __future__ import annotations

from dataclasses import dataclass

from audio_understanding.utils.text import normalize_arabic_text


@dataclass
class KeywordHit:
    keyword: str
    count: int


class KeywordSpotter:
    """Keyword spotting by normalized transcript matching."""

    def find_keywords(self, transcript: str, keywords: list[str]) -> list[KeywordHit]:
        normalized_transcript = normalize_arabic_text(transcript)
        hits: list[KeywordHit] = []
        for keyword in keywords:
            normalized_keyword = normalize_arabic_text(keyword)
            if not normalized_keyword:
                continue
            count = normalized_transcript.count(normalized_keyword)
            if count > 0:
                hits.append(KeywordHit(keyword=keyword, count=count))
        return hits
