from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LightweightTextEncoder(nn.Module):
    """Runnable text encoder with the same shape contract as BERT-style encoders."""

    def __init__(self, vocab_size: int, text_dim: int, max_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, text_dim))
        self.encoder = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Linear(text_dim * 2, text_dim),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embedding(token_ids)
        x = x + self.position_embedding[:, : x.shape[1]]
        x = self.encoder(x)
        mask = (token_ids != 0).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return x.transpose(1, 2), F.normalize(pooled, dim=-1)


class TextFeatureProjector(nn.Module):
    def __init__(self, text_dim: int, channels: tuple[int, ...]):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv1d(text_dim, channel, 1) for channel in channels])

    def forward(self, text_features: torch.Tensor) -> list[torch.Tensor]:
        return [projection(text_features) for projection in self.projections]
