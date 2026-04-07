from __future__ import annotations

import argparse
from pathlib import Path
import sys

import librosa
import pandas as pd
import torch
from torch import optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from audio_understanding.asr.cnn_lstm_ctc import CNNLSTMCTC, ctc_loss


class CharTokenizer:
    def __init__(self, texts: list[str]) -> None:
        vocab = sorted({ch for text in texts for ch in text})
        self.blank_id = 0
        self.stoi = {"<blank>": self.blank_id, **{ch: i + 1 for i, ch in enumerate(vocab)}}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]


def load_features(audio_path: str, n_mels: int = 80) -> torch.Tensor:
    y, sr = librosa.load(audio_path, sr=16_000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel).T
    return torch.tensor(mel_db, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN+LSTM+CTC model")
    parser.add_argument("--manifest", required=True, help="CSV with columns audio_path,text")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", default="data/models/cnn_lstm_ctc.pt")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    tokenizer = CharTokenizer(df["text"].astype(str).tolist())

    model = CNNLSTMCTC(n_mels=80, hidden_size=256, vocab_size=tokenizer.vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for _, row in df.iterrows():
            feats = load_features(row["audio_path"]).unsqueeze(0)
            target_ids = tokenizer.encode(str(row["text"]))
            if not target_ids:
                continue

            targets = torch.tensor(target_ids, dtype=torch.long)
            logits = model(feats)

            input_lengths = torch.tensor([logits.shape[1]], dtype=torch.long)
            target_lengths = torch.tensor([len(target_ids)], dtype=torch.long)

            loss = ctc_loss(logits, targets, input_lengths, target_lengths, blank_id=tokenizer.blank_id)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        print(f"Epoch {epoch + 1}/{args.epochs} - loss={epoch_loss:.4f}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "stoi": tokenizer.stoi}, args.save)
    print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
