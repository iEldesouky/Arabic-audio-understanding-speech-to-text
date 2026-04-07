"""Advanced downstream analytics for transcript and audio."""

from .emotion import EmotionDetector
from .emotion import EmotionPrediction
from .keyword_spotting import KeywordHit, KeywordSpotter
from .search import SearchResult, TranscriptSearchEngine
from .speaker_id import SpeakerIdentifier, SpeakerSegment
from .summarization import Summarizer

__all__ = [
    "Summarizer",
    "TranscriptSearchEngine",
    "SearchResult",
    "SpeakerIdentifier",
    "SpeakerSegment",
    "EmotionDetector",
    "EmotionPrediction",
    "KeywordSpotter",
    "KeywordHit",
]
