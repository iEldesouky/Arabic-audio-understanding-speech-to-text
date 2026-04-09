"""Optional advanced audio understanding modules."""

from .emotion import EmotionDetector
from .keyword_spotting import KeywordHit, KeywordSpotter
from .search import SearchResult, TranscriptSearchEngine
from .speaker_id import SpeakerIdentifier, SpeakerSegment
from .summarization import Summarizer

__all__ = [
    "EmotionDetector",
    "KeywordHit",
    "KeywordSpotter",
    "SearchResult",
    "SpeakerIdentifier",
    "SpeakerSegment",
    "Summarizer",
    "TranscriptSearchEngine",
]
