from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TAVLossConfig:
    dice_weight: float = 0.6
    ce_weight: float = 0.4
    contrastive_weight: float = 1.0
    contrastive_margin: float = 0.2


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]
    target_1h = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_1h, dim=dims)
    denominator = torch.sum(probs + target_1h, dim=dims)
    return 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()


def contrastive_alignment_loss(image_embedding: torch.Tensor, text_embedding: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    image_embedding = F.normalize(image_embedding, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)
    sim = image_embedding @ text_embedding.T
    positive = 1.0 - sim.diag()
    negative_mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    negative = torch.relu(sim[negative_mask] - margin).pow(2)
    if negative.numel() == 0:
        return positive.mean()
    return positive.mean() + negative.mean()


class TAVLoss(nn.Module):
    def __init__(self, config: TAVLossConfig = TAVLossConfig()):
        super().__init__()
        self.config = config

    def forward(self, outputs: dict, target: torch.Tensor) -> dict[str, torch.Tensor]:
        dice = dice_loss(outputs["logits"], target)
        ce = F.cross_entropy(outputs["logits"], target.long())
        contrastive = contrastive_alignment_loss(
            outputs["image_embedding"], outputs["text_embedding"], self.config.contrastive_margin
        )
        total = self.config.dice_weight * dice + self.config.ce_weight * ce + self.config.contrastive_weight * contrastive
        return {"loss": total, "dice_loss": dice.detach(), "ce_loss": ce.detach(), "contrastive_loss": contrastive.detach()}
