from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import librosa
import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from audio_understanding.asr import CNNLSTMCTC


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: dict[int, str]


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, vocab: Vocab, sample_rate: int = 16000) -> None:
        self.df = pd.read_csv(manifest_path)
        self.vocab = vocab
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=80)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Use librosa loading to avoid backend-specific torchaudio codec issues.
        audio, _ = librosa.load(str(row["audio_path"]), sr=self.sample_rate, mono=True)
        wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        feat = self.mel(wav).squeeze(0).transpose(0, 1)
        text = str(row["text"])
        labels = torch.tensor([self.vocab.stoi[c] for c in text if c in self.vocab.stoi], dtype=torch.long)
        return feat, labels


def build_vocab(texts: list[str]) -> Vocab:
    chars = sorted({c for t in texts for c in str(t)})
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return Vocab(stoi=stoi, itos=itos)


def collate(batch):
    feats, labels = zip(*batch)
    feat_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    label_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    max_t = int(feat_lengths.max())
    n_mels = feats[0].shape[1]
    padded_feats = torch.zeros(len(feats), max_t, n_mels)
    for i, f in enumerate(feats):
        padded_feats[i, : f.shape[0], :] = f

    flat_labels = torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)
    return padded_feats, feat_lengths, flat_labels, label_lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple CNN+BiLSTM+CTC Arabic ASR model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("data/models/cnn_lstm_ctc_sanity.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    vocab = build_vocab(df["text"].astype(str).tolist())

    ds = ManifestDataset(args.manifest, vocab)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTMCTC(n_mels=80, vocab_size=len(vocab.stoi) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for feats, feat_lens, labels, label_lens in dl:
            feats = feats.to(device)
            labels = labels.to(device)

            logits = model(feats)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(log_probs, labels, feat_lens, label_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        avg = running / max(1, len(dl))
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "vocab": vocab.stoi}, args.out)
    print(f"Saved model checkpoint to {args.out}")


if __name__ == "__main__":
    main()
