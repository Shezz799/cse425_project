from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import OUTPUT_ROOT, SEED, TOKENIZED_DIR
from src.data.datasets import TokenSequenceDataset, collate_token_batch
from src.models.transformer import TransformerDecoderModel
from src.training.utils import save_loss_plot, set_seed


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        composer_ids = batch["composer_ids"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attention_mask, composer_ids=composer_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
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
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            composer_ids = batch["composer_ids"].to(device)
            logits = model(input_ids, attention_mask=attention_mask, composer_ids=composer_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--num-genres", type=int, default=1)
    parser.add_argument("--pad-token-id", type=int, default=-1)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = TokenSequenceDataset(TOKENIZED_DIR / "tokens_train.pkl")
    val_data = TokenSequenceDataset(TOKENIZED_DIR / "tokens_validation.pkl")
    pad_token_id = train_data.pad_token_id if args.pad_token_id < 0 else args.pad_token_id

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_token_batch(b, pad_token_id=pad_token_id),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_token_batch(b, pad_token_id=pad_token_id),
    )

    model = TransformerDecoderModel(
        vocab_size=args.vocab_size,
        num_genres=max(args.num_genres, 1),
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        perplexity = math.exp(val_loss)
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} ppl={perplexity:.2f}")

    ckpt_dir = OUTPUT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "transformer.pt")

    save_loss_plot(train_losses, val_losses, OUTPUT_ROOT / "plots" / "task3_transformer_loss.png")


if __name__ == "__main__":
    main()
