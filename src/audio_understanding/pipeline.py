from __future__ import annotations

from dataclasses import dataclass

from audio_understanding.advanced import (
    EmotionDetector,
    KeywordHit,
    KeywordSpotter,
    SearchResult,
    SpeakerIdentifier,
    SpeakerSegment,
    Summarizer,
    TranscriptSearchEngine,
)
from audio_understanding.asr import Wav2Vec2ASR, WhisperASR
from audio_understanding.config import AppConfig
from audio_understanding.utils.text import split_sentences


@dataclass
class PipelineOutput:
    transcript: str
    summary: str | None
    search_results: list[SearchResult]
    speaker_segments: list[SpeakerSegment]
    emotions: list
    keyword_hits: list[KeywordHit]


class AudioUnderstandingPipeline:
    """End-to-end system: Speech -> Text -> Summary/Search/Insights."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        if config.models.asr_backend.lower() == "wav2vec2":
            self.asr = Wav2Vec2ASR(config.models.asr_model)
        else:
            self.asr = WhisperASR(config.models.asr_model)

        self.summarizer: Summarizer | None = None
        self.search_engine: TranscriptSearchEngine | None = None
        self.speaker_identifier: SpeakerIdentifier | None = None
        self.emotion_detector: EmotionDetector | None = None
        self.keyword_spotter = KeywordSpotter()

    def _get_summarizer(self) -> Summarizer:
        if self.summarizer is None:
            self.summarizer = Summarizer(self.config.models.summary_model)
        return self.summarizer

    def _get_search_engine(self) -> TranscriptSearchEngine:
        if self.search_engine is None:
            self.search_engine = TranscriptSearchEngine(self.config.models.search_embedding_model)
        return self.search_engine

    def _get_speaker_identifier(self) -> SpeakerIdentifier:
        if self.speaker_identifier is None:
            self.speaker_identifier = SpeakerIdentifier(self.config.models.speaker_model)
        return self.speaker_identifier

    def _get_emotion_detector(self) -> EmotionDetector:
        if self.emotion_detector is None:
            self.emotion_detector = EmotionDetector(self.config.models.emotion_model)
        return self.emotion_detector

    def run(
        self,
        audio_path: str,
        enable_summary: bool = True,
        enable_search: bool = True,
        enable_speaker_id: bool = True,
        enable_emotion: bool = True,
        custom_keywords: list[str] | None = None,
    ) -> PipelineOutput:
        transcript = self.asr.transcribe(audio_path)

        summary: str | None = None
        if enable_summary:
            summary = self._get_summarizer().summarize(transcript)

        search_results: list[SearchResult] = []
        if enable_search:
            chunks = split_sentences(transcript)
            search_engine = self._get_search_engine()
            search_engine.build_index(chunks)
            if chunks:
                search_results = search_engine.search(chunks[0], top_k=min(3, len(chunks)))

        speaker_segments: list[SpeakerSegment] = []
        if enable_speaker_id:
            speaker_segments = self._get_speaker_identifier().detect_speakers(
                audio_path=audio_path,
                chunk_seconds=self.config.pipeline.chunk_seconds,
                n_speakers=self.config.pipeline.speaker_count,
            )

        emotions: list = []
        if enable_emotion:
            emotions = self._get_emotion_detector().predict(audio_path)

        keywords = custom_keywords or self.config.pipeline.default_keywords
        keyword_hits = self.keyword_spotter.find_keywords(transcript, keywords)

        return PipelineOutput(
            transcript=transcript,
            summary=summary,
            search_results=search_results,
            speaker_segments=speaker_segments,
            emotions=emotions,
            keyword_hits=keyword_hits,
        )
