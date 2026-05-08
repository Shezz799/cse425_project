from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
from miditok import REMI, TokenizerConfig

from src.config import FS, PITCH_MIN
from src.preprocessing.tokens import build_tokenizer


def piano_roll_to_midi(piano_roll: np.ndarray, fs: int = FS, velocity: int = 80) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    time_steps, pitches = piano_roll.shape

    for pitch_idx in range(pitches):
        active = piano_roll[:, pitch_idx].astype(bool)
        start = None
        for t, is_on in enumerate(active):
            if is_on and start is None:
                start = t
            elif not is_on and start is not None:
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=PITCH_MIN + pitch_idx,
                    start=start / fs,
                    end=t / fs,
                )
                instrument.notes.append(note)
                start = None
        if start is not None:
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=PITCH_MIN + pitch_idx,
                start=start / fs,
                end=time_steps / fs,
            )
            instrument.notes.append(note)

    pm.instruments.append(instrument)
    return pm


def save_pianoroll_midi(piano_roll: np.ndarray, out_path: Path, fs: int = FS) -> None:
    midi = piano_roll_to_midi(piano_roll, fs=fs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out_path))


def tokens_to_midi(tokens: list[int], out_path: Path) -> None:
    tokenizer = build_tokenizer()
    midi = tokenizer.tokens_to_midi(tokens)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(out_path))
