from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from audio_understanding.asr import Wav2Vec2ASR, WhisperASR
from audio_understanding.utils.metrics import compute_wer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR output and compute WER for one sample.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--backend", default="whisper", choices=["whisper", "wav2vec2"])
    parser.add_argument("--model", default="openai/whisper-small")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend == "wav2vec2":
        asr = Wav2Vec2ASR(args.model)
    else:
        asr = WhisperASR(args.model)

    hyp = asr.transcribe(args.audio)
    score = compute_wer(args.reference, hyp)

    print("Reference:", args.reference)
    print("Hypothesis:", hyp)
    print(f"WER: {score:.4f}")


if __name__ == "__main__":
    main()
