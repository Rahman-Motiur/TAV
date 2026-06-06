from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from tav import TAV, TAVConfig
from tav.data import TextSegmentationCsvDataset
from tav.training import TAVLoss, TAVLossConfig, dice_score, mean_iou
from tav.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TextSegmentationCsvDataset(
        cfg["data"]["train_csv"],
        tuple(cfg["data"]["image_size"]),
        cfg["data"]["in_channels"],
        cfg["data"]["vocab_size"],
        cfg["data"]["max_text_length"],
    )
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"])
    model_cfg = cfg["model"].copy()
    model_cfg["channels"] = tuple(model_cfg["channels"])
    model = TAV(TAVConfig(**model_cfg)).to(device)
    loss_keys = TAVLossConfig.__dataclass_fields__.keys()
    criterion = TAVLoss(TAVLossConfig(**{k: cfg["training"][k] for k in loss_keys}))
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            outputs = model(images, text_tokens)
            losses = criterion(outputs, masks)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            total_loss += float(losses["loss"].detach())
            total_dice += dice_score(outputs["logits"].detach(), masks)
            total_iou += mean_iou(outputs["logits"].detach(), masks)
        print(f"epoch={epoch:03d} loss={total_loss / len(loader):.4f} dice={total_dice / len(loader):.4f} iou={total_iou / len(loader):.4f}")
        torch.save({"model": model.state_dict(), "config": cfg}, run_dir / "last.pt")


if __name__ == "__main__":
    main()
