from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = (1 - pt) ** self.gamma
        return (focal * bce).mean()


def estimate_pos_weight(data: torch.Tensor) -> torch.Tensor:
    total = data.numel()
    positives = data.sum().item()
    negatives = total - positives
    pos_weight = negatives / max(positives, 1.0)
    pos_weight = max(10.0, min(30.0, pos_weight))
    return torch.tensor(pos_weight, dtype=torch.float32)


def save_loss_plot(train_losses: list[float], val_losses: list[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
