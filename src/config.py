from __future__ import annotations

import os
from pathlib import Path

SEED = 42

FS = 16
WINDOW_LEN = 128
PITCH_MIN = 21
PITCH_MAX = 108
SPARSE_THRESHOLD = 0.02
VELOCITY_BINS = 32

ROOT = Path(__file__).resolve().parents[1]


def _candidate_data_roots() -> list[Path]:
    candidates: list[Path] = []
    env_root = os.getenv("DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            ROOT / "data" / "raw_midi" / "maestro-v3.0.0",
            Path("/kaggle/input/maestro-v3-0-0/maestro-v3.0.0"),
            Path("/kaggle/input/maestro-v3.0.0/maestro-v3.0.0"),
            Path("/kaggle/input/maestro-v3-0-0"),
            Path("/kaggle/input/maestro-v3.0.0"),
        ]
    )
    return candidates


def resolve_maestro_root() -> Path:
    for candidate in _candidate_data_roots():
        if (candidate / "maestro-v3.0.0.csv").exists():
            return candidate
        nested = candidate / "maestro-v3.0.0"
        if (nested / "maestro-v3.0.0.csv").exists():
            return nested
    raise FileNotFoundError(
        "MAESTRO dataset not found. Set DATA_ROOT or place data under data/raw_midi/maestro-v3.0.0."
    )


RAW_MIDI_ROOT = resolve_maestro_root()
PROCESSED_DIR = ROOT / "data" / "processed"
TOKENIZED_DIR = ROOT / "data" / "tokenized"

if Path("/kaggle/working").exists():
    DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/outputs")
else:
    DEFAULT_OUTPUT_ROOT = ROOT / "outputs"

OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, TOKENIZED_DIR, OUTPUT_ROOT, OUTPUT_ROOT / "generated_midis", OUTPUT_ROOT / "plots"]:
        path.mkdir(parents=True, exist_ok=True)


ensure_dirs()
