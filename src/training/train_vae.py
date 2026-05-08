from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import OUTPUT_ROOT, PROCESSED_DIR, SEED
from src.data.datasets import PianoRollDataset
from src.models.lstm_ae import LSTMVAE
from src.training.utils import FocalLoss, estimate_pos_weight, save_loss_plot, set_seed


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def train_epoch(model, loader, loss_fn, optimizer, device, beta: float):
    model.train()
    total_loss = 0.0
    recon_total = 0.0
    kl_total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, mu, logvar = model(batch)
        recon = loss_fn(logits, batch)
        kl = kl_divergence(mu, logvar)
        loss = recon + beta * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        recon_total += recon.item()
        kl_total += kl.item()
    return (
        total_loss / max(len(loader), 1),
        recon_total / max(len(loader), 1),
        kl_total / max(len(loader), 1),
    )


def eval_epoch(model, loader, loss_fn, device, beta: float):
    model.eval()
    total_loss = 0.0
    recon_total = 0.0
    kl_total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, mu, logvar = model(batch)
            recon = loss_fn(logits, batch)
            kl = kl_divergence(mu, logvar)
            loss = recon + beta * kl
            total_loss += loss.item()
            recon_total += recon.item()
            kl_total += kl.item()
    return (
        total_loss / max(len(loader), 1),
        recon_total / max(len(loader), 1),
        kl_total / max(len(loader), 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--use-focal", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = PianoRollDataset(PROCESSED_DIR / "pianoroll_train.npz")
    val_data = PianoRollDataset(PROCESSED_DIR / "pianoroll_validation.npz")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = LSTMVAE().to(device)
    pos_weight = estimate_pos_weight(torch.from_numpy(train_data.data))
    pos_weight = pos_weight.to(device)

    if args.use_focal:
        loss_fn = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    recon_losses: list[float] = []
    kl_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        beta = min(1.0, epoch / max(args.warmup_epochs, 1))
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, loss_fn, optimizer, device, beta
        )
        val_loss, val_recon, val_kl = eval_epoch(model, val_loader, loss_fn, device, beta)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        recon_losses.append(val_recon)
        kl_losses.append(val_kl)
        print(
            f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f} beta={beta:.2f}"
        )

    ckpt_dir = OUTPUT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "lstm_vae.pt")

    save_loss_plot(train_losses, val_losses, OUTPUT_ROOT / "plots" / "task2_vae_loss.png")
    save_loss_plot(recon_losses, kl_losses, OUTPUT_ROOT / "plots" / "task2_recon_kl.png")


if __name__ == "__main__":
    main()
