import torch

from tav import TAV, TAVConfig
from tav.training import TAVLoss


def test_tav_forward_and_loss():
    model = TAV(TAVConfig(image_channels=1, num_classes=2, vocab_size=256, max_text_length=8, text_dim=32, channels=(8, 16, 32, 64)))
    images = torch.randn(1, 1, 32, 32)
    masks = torch.randint(0, 2, (1, 32, 32))
    text_tokens = torch.randint(1, 256, (1, 8))
    outputs = model(images, text_tokens)
    assert outputs["logits"].shape == (1, 2, 32, 32)
    losses = TAVLoss()(outputs, masks)
    assert torch.isfinite(losses["loss"])
