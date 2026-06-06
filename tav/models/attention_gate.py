from __future__ import annotations

import torch
from torch import nn


class AttentionGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        weights = self.gate(features + attention)
        return weights * features + features
