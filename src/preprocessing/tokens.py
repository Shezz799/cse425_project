from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import miditoolkit
from miditok import REMI, TokenizerConfig

from src.config import RAW_MIDI_ROOT, TOKENIZED_DIR, VELOCITY_BINS
from src.preprocessing.maestro import get_split_df


def build_tokenizer() -> REMI:
    config = TokenizerConfig(
        num_velocities=VELOCITY_BINS,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=False,
    )
    return REMI(config)


def _extract_token_ids(encoded: Any) -> list[int]:
    if encoded is None:
        return []

    ids = getattr(encoded, "ids", None)
    if ids is not None:
        if isinstance(ids, (list, tuple)):
            if not ids:
                return []
            if all(isinstance(x, int) for x in ids):
                return list(ids)
        if hasattr(ids, "tolist"):
            maybe = ids.tolist()
            if isinstance(maybe, list) and (not maybe or all(isinstance(x, int) for x in maybe)):
                return list(maybe)

    if isinstance(encoded, list):
        if not encoded:
            return []
        if all(isinstance(x, int) for x in encoded):
            return list(encoded)

        combined: list[int] = []
        for item in encoded:
            combined.extend(_extract_token_ids(item))
        return combined

    raise TypeError(f"Unexpected encode() return type: {type(encoded)}")


def tokenize_split(split: str, limit: int | None = None) -> dict:
    df = get_split_df(split)
    if limit is not None:
        df = df.head(limit)

    tokenizer = build_tokenizer()
    sequences: list[list[int]] = []
    skipped = 0
    printed_exceptions = 0
    min_len = 2

    for row in df.itertuples(index=False):
        midi_path = RAW_MIDI_ROOT / row.midi_filename
        try:
            midi = miditoolkit.MidiFile(str(midi_path))
            encoded = tokenizer.encode(midi)
            token_ids = _extract_token_ids(encoded)

            if len(token_ids) < min_len:
                skipped += 1
                continue

            sequences.append(token_ids)
        except Exception as e:
            skipped += 1
            if printed_exceptions < 5:
                printed_exceptions += 1
                print(f"[skip-exception] {midi_path}: {type(e).__name__}: {e}")
            continue

    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    return {
        "sequences": sequences,
        "pad_token_id": pad_token_id,
    }


def save_split(split: str, payload: dict) -> Path:
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TOKENIZED_DIR / f"tokens_{split}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return out_path


def run_tokenization(limit: int | None = None) -> None:
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer()
    tokenizer.save(TOKENIZED_DIR / "tokenizer.json")
    for split in ["train", "validation", "test"]:
        payload = tokenize_split(split, limit=limit)
        save_split(split, payload)
        print(f"saved {split}: {len(payload['sequences'])} sequences")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional file limit for quick runs")
    args = parser.parse_args()
    run_tokenization(limit=args.limit)


if __name__ == "__main__":
    main()
