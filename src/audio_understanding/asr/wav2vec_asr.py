from __future__ import annotations

import librosa
from transformers import pipeline


class Wav2Vec2ASR:
    """Wrapper around an Arabic-compatible Wav2Vec2 CTC model."""

    def __init__(
        self,
        model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    ) -> None:
        self._pipe = pipeline(task="automatic-speech-recognition", model=model_name)

    def transcribe(self, audio_path: str) -> str:
        audio, sr = librosa.load(audio_path, sr=16_000)
        result = self._pipe({"array": audio, "sampling_rate": sr})
        return str(result["text"]).strip()
