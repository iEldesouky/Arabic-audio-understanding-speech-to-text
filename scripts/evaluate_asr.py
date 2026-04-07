from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from audio_understanding.asr import Wav2Vec2ASR, WhisperASR
from audio_understanding.utils.metrics import compute_wer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR using WER")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--reference", required=True, help="Reference transcript text")
    parser.add_argument("--backend", choices=["whisper", "wav2vec2"], default="whisper")
    parser.add_argument("--model", default="openai/whisper-small")
    args = parser.parse_args()

    if args.backend == "wav2vec2":
        asr = Wav2Vec2ASR(args.model)
    else:
        asr = WhisperASR(args.model)

    prediction = asr.transcribe(args.audio)
    wer_value = compute_wer(args.reference, prediction)

    print("Predicted transcript:")
    print(prediction)
    print("\nWER:", f"{wer_value:.4f}")


if __name__ == "__main__":
    main()
