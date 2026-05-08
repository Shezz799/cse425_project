from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi

from src.config import FS, PITCH_MAX, PITCH_MIN, PROCESSED_DIR, SPARSE_THRESHOLD, WINDOW_LEN
from src.preprocessing.maestro import iter_midi_paths


def safe_load_midi(path: Path) -> pretty_midi.PrettyMIDI | None:
    try:
        return pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None


def midi_to_pianoroll(pm: pretty_midi.PrettyMIDI, fs: int = FS) -> np.ndarray:
    roll = pm.get_piano_roll(fs=fs)
    roll = roll[PITCH_MIN : PITCH_MAX + 1]
    roll = roll.T
    roll = (roll > 0).astype(np.uint8)
    return roll


def window_pianoroll(roll: np.ndarray, window_len: int = WINDOW_LEN, min_active_ratio: float = SPARSE_THRESHOLD) -> list[np.ndarray]:
    total_steps = roll.shape[0]
    windows: list[np.ndarray] = []
    for start in range(0, total_steps - window_len + 1, window_len):
        window = roll[start : start + window_len]
        active_ratio = window.mean()
        if active_ratio >= min_active_ratio:
            windows.append(window)
    return windows


def preprocess_split(split: str, limit: int | None = None) -> np.ndarray:
    windows: list[np.ndarray] = []
    for midi_path in iter_midi_paths(split, limit=limit):
        pm = safe_load_midi(midi_path)
        if pm is None:
            continue
        roll = midi_to_pianoroll(pm, fs=FS)
        windows.extend(window_pianoroll(roll))
    if not windows:
        return np.empty((0, WINDOW_LEN, PITCH_MAX - PITCH_MIN + 1), dtype=np.uint8)
    return np.stack(windows, axis=0)


def save_split(split: str, data: np.ndarray) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"pianoroll_{split}.npz"
    np.savez_compressed(out_path, data=data)
    return out_path


def run_preprocessing(limit: int | None = None) -> None:
    for split in ["train", "validation", "test"]:
        data = preprocess_split(split, limit=limit)
        save_split(split, data)
        print(f"{split}: {data.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional file limit for quick runs")
    args = parser.parse_args()
    run_preprocessing(limit=args.limit)


if __name__ == "__main__":
    main()
