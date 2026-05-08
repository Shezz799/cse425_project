from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src.config import OUTPUT_ROOT, PROCESSED_DIR, SEED
from src.generation.midi_io import save_pianoroll_midi
from src.models.lstm_ae import LSTMVAE


def interpolate_latent(mu_a: torch.Tensor, mu_b: torch.Tensor, steps: int) -> torch.Tensor:
    alphas = torch.linspace(0.0, 1.0, steps, device=mu_a.device).unsqueeze(1)
    return (1 - alphas) * mu_a + alphas * mu_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = PROCESSED_DIR / "pianoroll_validation.npz"
    payload = np.load(data_path)
    windows = payload["data"].astype(np.float32)

    if len(windows) < 2:
        raise ValueError("Need at least two windows for interpolation.")

    idx_a, idx_b = rng.sample(range(len(windows)), 2)
    x_a = torch.from_numpy(windows[idx_a]).unsqueeze(0).to(device)
    x_b = torch.from_numpy(windows[idx_b]).unsqueeze(0).to(device)

    model = LSTMVAE().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    mu_a, _ = model.encode(x_a)
    mu_b, _ = model.encode(x_b)

    z = interpolate_latent(mu_a, mu_b, steps=args.num_samples)
    decoded = model.decode(z, seq_len=x_a.size(1))
    probs = torch.sigmoid(decoded)
    samples = (probs > args.threshold).float().cpu().numpy()

    out_dir = OUTPUT_ROOT / "generated_midis"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        save_pianoroll_midi(sample, out_dir / f"task2_interp_{i+1}.mid")


if __name__ == "__main__":
    main()
