from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import OUTPUT_ROOT, PROCESSED_DIR, SEED
from src.data.datasets import PianoRollDataset
from src.models.lstm_ae import LSTMAutoencoder
from src.training.utils import FocalLoss, estimate_pos_weight, save_loss_plot, set_seed


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = loss_fn(logits, batch)
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-focal", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = PianoRollDataset(PROCESSED_DIR / "pianoroll_train.npz")
    val_data = PianoRollDataset(PROCESSED_DIR / "pianoroll_validation.npz")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = LSTMAutoencoder().to(device)
    pos_weight = estimate_pos_weight(torch.from_numpy(train_data.data))
    pos_weight = pos_weight.to(device)

    if args.use_focal:
        loss_fn = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

    ckpt_dir = OUTPUT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "lstm_ae.pt")

    save_loss_plot(train_losses, val_losses, OUTPUT_ROOT / "plots" / "task1_ae_loss.png")


if __name__ == "__main__":
    main()
