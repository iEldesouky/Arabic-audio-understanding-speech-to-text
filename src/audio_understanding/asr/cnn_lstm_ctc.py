from __future__ import annotations

import torch
from torch import nn


class CNNLSTMCTC(nn.Module):
    """Simple CNN + BiLSTM + CTC head for speech recognition experiments.

    Expected input shape: (batch, time, mel_bins)
    """

    def __init__(self, n_mels: int, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M) -> (B, M, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Back to (B, T, C) for LSTM
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        logits = self.classifier(x)
        return logits


def ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int = 0,
) -> torch.Tensor:
    """CTC loss helper. Logits shape must be (B, T, V)."""
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    return loss_fn(log_probs, targets, input_lengths, target_lengths)


def greedy_decode(logits: torch.Tensor, blank_id: int = 0) -> list[list[int]]:
    """Greedy decode token ids from CTC logits."""
    token_ids = logits.argmax(dim=-1)
    decoded: list[list[int]] = []

    for seq in token_ids:
        output = []
        prev = None
        for token in seq.tolist():
            if token != blank_id and token != prev:
                output.append(token)
            prev = token
        decoded.append(output)

    return decoded
