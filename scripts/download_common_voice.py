from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download


def download_common_voice(
    split: str,
    max_samples: int,
    output: Path,
    dataset_id: str,
    lang: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(dataset_id, lang, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.select(range(min(max_samples, len(ds))))

    rows = []
    for i, item in enumerate(ds):
        audio_path = output / f"sample_{i:06d}.wav"
        waveform = item["audio"]["array"]
        sample_rate = item["audio"]["sampling_rate"]

        import soundfile as sf

        sf.write(audio_path, waveform, sample_rate)
        rows.append(f"{audio_path}\t{item['sentence']}\n")

    (output / "metadata.tsv").write_text("".join(rows), encoding="utf-8")
    print(f"Saved {len(rows)} Common Voice samples to {output}")


def build_masc_metadata(output: Path) -> Path:
    meta_path = output / "meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    df = pd.read_csv(meta_path)
    required = {"audio_path", "transcript"}
    if not required.issubset(set(df.columns)):
        raise ValueError("MASC metadata is missing required columns: audio_path, transcript")

    rows = []
    for _, row in df.iterrows():
        wav_rel = str(row["audio_path"])
        transcript = str(row["transcript"]).strip()
        wav_abs = (output / wav_rel).resolve()
        if wav_abs.exists() and transcript:
            rows.append(f"{wav_abs}\t{transcript}\n")

    metadata_tsv = output / "metadata.tsv"
    metadata_tsv.write_text("".join(rows), encoding="utf-8")
    print(f"Built metadata file with {len(rows)} segments: {metadata_tsv}")
    return metadata_tsv


def download_masc_full(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    print("Downloading full MASC dataset snapshot...")
    snapshot_download(
        repo_id="hirundo-io/MASC",
        repo_type="dataset",
        local_dir=str(output),
        local_dir_use_symlinks=False,
    )
    build_masc_metadata(output)
    print(f"Full MASC dataset is available at {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Arabic speech datasets")
    parser.add_argument("--source", choices=["common_voice", "masc"], default="masc")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output", default="data/raw/masc_full")
    parser.add_argument("--dataset-id", default="mozilla-foundation/common_voice_17_0")
    parser.add_argument("--lang", default="ar")
    args = parser.parse_args()

    out_dir = Path(args.output)
    if args.source == "masc":
        download_masc_full(out_dir)
        return

    download_common_voice(
        split=args.split,
        max_samples=args.max_samples,
        output=out_dir,
        dataset_id=args.dataset_id,
        lang=args.lang,
    )


if __name__ == "__main__":
    main()
