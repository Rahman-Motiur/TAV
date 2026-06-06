from __future__ import annotations

import numpy as np
import torch


def dice_score(logits: torch.Tensor, target: torch.Tensor, include_background: bool = False, smooth: float = 1e-6) -> float:
    pred = torch.argmax(logits, dim=1)
    scores = []
    start = 0 if include_background else 1
    for cls in range(start, logits.shape[1]):
        p = pred == cls
        t = target == cls
        denom = p.sum() + t.sum()
        if denom == 0:
            continue
        scores.append(float((2.0 * (p & t).sum().float() + smooth) / (denom.float() + smooth)))
    return float(np.mean(scores)) if scores else 1.0


def mean_iou(logits: torch.Tensor, target: torch.Tensor, include_background: bool = False, smooth: float = 1e-6) -> float:
    pred = torch.argmax(logits, dim=1)
    scores = []
    start = 0 if include_background else 1
    for cls in range(start, logits.shape[1]):
        p = pred == cls
        t = target == cls
        union = (p | t).sum()
        if union == 0:
            continue
        scores.append(float(((p & t).sum().float() + smooth) / (union.float() + smooth)))
    return float(np.mean(scores)) if scores else 1.0
