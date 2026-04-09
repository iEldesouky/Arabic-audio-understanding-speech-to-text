from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ASR training manifest from metadata TSV.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to TSV with audio_path and text columns")
    parser.add_argument("--out-manifest", type=Path, default=Path("data/processed/train_manifest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metadata, sep="\t")

    required = {"audio_path", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metadata: {missing}")

    df = df.dropna(subset=["audio_path", "text"]).copy()
    df["audio_path"] = df["audio_path"].astype(str)
    df["text"] = df["text"].astype(str)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_manifest, index=False)
    print(f"Saved manifest with {len(df)} rows to {args.out_manifest}")


if __name__ == "__main__":
    main()
