from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from audio_understanding.advanced.search import TranscriptSearchEngine
from audio_understanding.utils.text import split_sentences


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query transcript search index")
    parser.add_argument("--transcript", required=True, help="Path to transcript text file")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    args = parser.parse_args()

    transcript_text = Path(args.transcript).read_text(encoding="utf-8")
    chunks = split_sentences(transcript_text)

    engine = TranscriptSearchEngine(args.model)
    engine.build_index(chunks)

    for idx, hit in enumerate(engine.search(args.query, top_k=5), start=1):
        print(f"{idx}. score={hit.score:.4f} | text={hit.text}")


if __name__ == "__main__":
    main()
