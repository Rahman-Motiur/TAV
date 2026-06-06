from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .attention_gate import AttentionGate
from .decoder import DecoderBlock
from .text_encoder import LightweightTextEncoder, TextFeatureProjector
from .tgam import TGAM
from .vision_encoder import VisionEncoder


@dataclass
class TAVConfig:
    image_channels: int = 1
    num_classes: int = 2
    vocab_size: int = 4096
    max_text_length: int = 32
    text_dim: int = 128
    channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    lambda_visual: float = 0.6
    lambda_cross: float = 0.4


class TAV(nn.Module):
    def __init__(self, config: TAVConfig):
        super().__init__()
        self.config = config
        c1, c2, c3, c4 = config.channels
        self.text_encoder = LightweightTextEncoder(config.vocab_size, config.text_dim, config.max_text_length)
        self.text_projector = TextFeatureProjector(config.text_dim, config.channels)
        self.vision_encoder = VisionEncoder(config.image_channels, config.channels)
        self.text_to_image = nn.ModuleList([nn.Linear(config.text_dim, c) for c in config.channels])
        self.tgam = nn.ModuleList([TGAM(c, config.lambda_visual, config.lambda_cross) for c in config.channels[:3]])
        self.gates = nn.ModuleList([AttentionGate(c) for c in config.channels[:3]])
        self.bottleneck = nn.Sequential(nn.Conv2d(c4, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.GELU())
        self.decoder3 = DecoderBlock(c4, c3, c3)
        self.decoder2 = DecoderBlock(c3, c2, c2)
        self.decoder1 = DecoderBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, config.num_classes, 1)
        self.image_projection = nn.Linear(c4, config.text_dim)
        self.text_projection = nn.Linear(config.text_dim, config.text_dim)

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        text_features, text_embedding = self.text_encoder(text_tokens)
        projected_text = self.text_projector(text_features)
        text_global = text_embedding

        features = []
        x = images
        for i, stage in enumerate(self.vision_encoder.stages):
            x = stage(x)
            text_bias = self.text_to_image[i](text_global)[:, :, None, None]
            x = x + text_bias
            if i < 3:
                attention = self.tgam[i](x, projected_text[i])
                x = self.gates[i](x, attention)
            features.append(x)

        x = self.bottleneck(features[3])
        x = self.decoder3(x, features[2])
        x = self.decoder2(x, features[1])
        x = self.decoder1(x, features[0])
        logits = F.interpolate(self.head(x), size=images.shape[-2:], mode="bilinear", align_corners=False)
        image_embedding = F.normalize(self.image_projection(features[3].mean(dim=(2, 3))), dim=-1)
        text_embedding = F.normalize(self.text_projection(text_embedding), dim=-1)
        return {"logits": logits, "image_embedding": image_embedding, "text_embedding": text_embedding, "features": features}
