from __future__ import annotations

import torch
from torch import nn


class CNNLSTMCTC(nn.Module):
    """Simple CNN + BiLSTM acoustic model with CTC output."""

    def __init__(self, n_mels: int, vocab_size: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.encoder = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [batch, time, n_mels] -> logits: [batch, time, vocab_size]"""
        x = features.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.encoder(x)
        return self.classifier(x)
