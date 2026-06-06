from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class TGAM(nn.Module):
    """Tri-Guided Attention Module: visual-visual, language-language, language-visual."""

    def __init__(self, channels: int, lambda_visual: float = 0.6, lambda_cross: float = 0.4):
        super().__init__()
        self.lambda_visual = lambda_visual
        self.lambda_cross = lambda_cross
        self.v_q = nn.Conv2d(channels, channels, 1)
        self.v_k = nn.Conv2d(channels, channels, 1)
        self.v_v = nn.Conv2d(channels, channels, 1)
        self.t_q = nn.Conv1d(channels, channels, 1)
        self.t_k = nn.Conv1d(channels, channels, 1)
        self.t_v = nn.Conv1d(channels, channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def _visual_attention(self, visual: torch.Tensor) -> torch.Tensor:
        b, c, h, w = visual.shape
        q = self.v_q(visual).flatten(2).transpose(1, 2)
        k = self.v_k(visual).flatten(2)
        v = self.v_v(visual).flatten(2).transpose(1, 2)
        attn = torch.softmax((q @ k) / math.sqrt(c), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return out * visual

    def _language_attention(self, text: torch.Tensor) -> torch.Tensor:
        c = text.shape[1]
        q = self.t_q(text).transpose(1, 2)
        k = self.t_k(text)
        v = self.t_v(text).transpose(1, 2)
        attn = torch.softmax((q @ k) / math.sqrt(c), dim=-1)
        return (attn @ v).transpose(1, 2) * text

    def _language_visual_attention(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        b, c, h, w = visual.shape
        q = self.t_q(text).transpose(1, 2)
        k = self.v_k(visual).flatten(2)
        v = self.v_v(visual).flatten(2).transpose(1, 2)
        attn = torch.softmax((q @ k) / math.sqrt(c), dim=-1)
        out = attn.transpose(1, 2) @ q
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out * visual

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        visual_attention = self._visual_attention(visual)
        language_attention = self._language_attention(text).mean(dim=-1, keepdim=True).unsqueeze(-1)
        cross_attention = self._language_visual_attention(visual, text)
        combined = self.lambda_visual * visual_attention + self.lambda_cross * cross_attention
        return self.output(combined * torch.sigmoid(language_attention))
