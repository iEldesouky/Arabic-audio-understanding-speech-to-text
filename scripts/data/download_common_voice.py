from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Mozilla Common Voice Arabic subset.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/common_voice_ar"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("mozilla-foundation/common_voice_17_0", "ar", split=args.split)
    ds = ds.select(range(min(args.max_samples, len(ds))))

    rows: list[dict[str, str]] = []
    for idx, item in enumerate(ds):
        audio = item["audio"]
        wav_path = args.out_dir / f"cv_ar_{args.split}_{idx:06d}.wav"
        audio_array = audio["array"]
        sample_rate = audio["sampling_rate"]

        import soundfile as sf

        sf.write(wav_path, audio_array, sample_rate)
        rows.append({"audio_path": str(wav_path), "text": str(item.get("sentence", "")).strip()})

    pd.DataFrame(rows).to_csv(args.out_dir / "metadata.tsv", sep="\t", index=False)
    print(f"Saved {len(rows)} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
