from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: tuple[int, ...]):
        super().__init__()
        stages = []
        previous = in_channels
        for index, channel in enumerate(channels):
            stages.append(ConvBlock(previous, channel, stride=1 if index == 0 else 2))
            previous = channel
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
