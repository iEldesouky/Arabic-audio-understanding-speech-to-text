from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from audio_understanding.advanced import TranscriptSearchEngine
from audio_understanding.utils.text import split_sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and save semantic search index from transcript text.")
    parser.add_argument("--transcript-file", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("data/index/transcript_chunks.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transcript = args.transcript_file.read_text(encoding="utf-8")
    chunks = split_sentences(transcript)

    engine = TranscriptSearchEngine()
    engine.build_index(chunks)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"chunks": chunks}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {args.out_json}")


if __name__ == "__main__":
    main()
