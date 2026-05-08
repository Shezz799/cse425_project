from __future__ import annotations

import numpy as np
import torch

from src.config import WINDOW_LEN


def sample_latent(model, num_samples: int, device: torch.device) -> torch.Tensor:
    if hasattr(model, "latent_fc"):
        latent_dim = model.latent_fc.out_features
    elif hasattr(model, "mu_fc"):
        latent_dim = model.mu_fc.out_features
    else:
        raise ValueError("Model does not expose a latent dimension.")
    return torch.randn(num_samples, latent_dim, device=device)


def decode_latent(model, z: torch.Tensor, threshold: float = 0.3) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model.decode(z, WINDOW_LEN)
        probs = torch.sigmoid(logits)
        samples = (probs > threshold).float().cpu().numpy()
    return samples
