from __future__ import annotations

import librosa
from transformers import pipeline


class WhisperASR:
    """Wrapper around Hugging Face Whisper ASR pipeline."""

    def __init__(self, model_name: str = "openai/whisper-small") -> None:
        self._pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            chunk_length_s=20,
            return_timestamps=False,
        )

    def transcribe(self, audio_path: str) -> str:
        audio, sr = librosa.load(audio_path, sr=16_000)
        result = self._pipe({"array": audio, "sampling_rate": sr})
        return str(result["text"]).strip()
