from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from tav import TAV, TAVConfig
from tav.training import TAVLoss, TAVLossConfig


def main() -> None:
    config = TAVConfig(
        image_channels=1,
        num_classes=2,
        vocab_size=1024,
        max_text_length=16,
        text_dim=64,
        channels=(16, 32, 64, 128),
    )
    model = TAV(config)
    images = torch.randn(2, 1, 64, 64)
    masks = torch.randint(0, 2, (2, 64, 64))
    text_tokens = torch.randint(1, 1024, (2, 16))
    outputs = model(images, text_tokens)
    losses = TAVLoss(TAVLossConfig())(outputs, masks)
    print("logits:", tuple(outputs["logits"].shape))
    print("image embedding:", tuple(outputs["image_embedding"].shape))
    print("text embedding:", tuple(outputs["text_embedding"].shape))
    print("loss:", float(losses["loss"].detach()))


if __name__ == "__main__":
    main()
