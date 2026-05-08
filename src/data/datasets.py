from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class PianoRollDataset(Dataset):
    def __init__(self, npz_path: Path):
        payload = np.load(npz_path)
        self.data = payload["data"].astype(np.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])


class TokenSequenceDataset(Dataset):
    def __init__(self, pkl_path: Path):
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        self.sequences = payload["sequences"]
        self.composer_ids = payload.get("composer_ids", [0] * len(self.sequences))
        self.pad_token_id = payload.get("pad_token_id", 0)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(self.sequences[idx], dtype=torch.long), int(self.composer_ids[idx])


def collate_token_batch(batch: list[tuple[torch.Tensor, int]], pad_token_id: int = 0) -> dict:
    sequences, composer_ids = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    if max_len < 2:
        raise ValueError("Sequences must contain at least 2 tokens for autoregressive training.")

    input_ids = torch.full((len(sequences), max_len - 1), pad_token_id, dtype=torch.long)
    target_ids = torch.full((len(sequences), max_len - 1), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len - 1), dtype=torch.bool)

    for i, seq in enumerate(sequences):
        inp = seq[:-1]
        tgt = seq[1:]
        length = len(inp)
        input_ids[i, :length] = inp
        target_ids[i, :length] = tgt
        attention_mask[i, :length] = True

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
        "composer_ids": torch.tensor(composer_ids, dtype=torch.long),
    }
