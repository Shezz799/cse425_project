from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from src.config import RAW_MIDI_ROOT


def load_metadata() -> pd.DataFrame:
    csv_path = RAW_MIDI_ROOT / "maestro-v3.0.0.csv"
    return pd.read_csv(csv_path)


def get_split_df(split: str) -> pd.DataFrame:
    df = load_metadata()
    return df[df["split"] == split].copy()


def iter_midi_paths(split: str, limit: int | None = None) -> Iterator[Path]:
    df = get_split_df(split)
    if limit is not None:
        df = df.head(limit)
    for row in df.itertuples(index=False):
        yield RAW_MIDI_ROOT / row.midi_filename


def build_composer_map(df: pd.DataFrame) -> dict[str, int]:
    composers = sorted(df["canonical_composer"].dropna().unique().tolist())
    return {name: idx for idx, name in enumerate(composers)}


def get_composer_map() -> dict[str, int]:
    return build_composer_map(load_metadata())


def attach_composer_ids(df: pd.DataFrame, composer_map: dict[str, int] | None = None) -> pd.DataFrame:
    composer_map = composer_map or get_composer_map()
    df = df.copy()
    df["composer_id"] = df["canonical_composer"].map(composer_map)
    df["composer_id"] = df["composer_id"].fillna(0).astype(int)
    return df
