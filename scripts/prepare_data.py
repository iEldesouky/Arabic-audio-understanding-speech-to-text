from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare metadata for ASR training")
    parser.add_argument("--metadata", required=True, help="Path to metadata TSV")
    parser.add_argument("--output", default="data/processed/train_manifest.csv")
    args = parser.parse_args()

    records = []
    for line in Path(args.metadata).read_text(encoding="utf-8").splitlines():
        audio_path, text = line.split("\t", maxsplit=1)
        records.append({"audio_path": audio_path, "text": text.strip()})

    df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Prepared {len(df)} records into {args.output}")


if __name__ == "__main__":
    main()
