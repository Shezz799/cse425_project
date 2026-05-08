from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi

from src.config import OUTPUT_ROOT, PITCH_MAX, PITCH_MIN
from src.evaluation.metrics import extract_notes, list_midi_files


def random_note_generator(
    num_notes: int = 200,
    duration_choices: list[float] | None = None,
    total_time: float = 30.0,
    seed: int = 42,
) -> pretty_midi.PrettyMIDI:
    rng = random.Random(seed)
    duration_choices = duration_choices or [0.25, 0.5, 1.0]
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for _ in range(num_notes):
        pitch = rng.randint(PITCH_MIN, PITCH_MAX)
        duration = rng.choice(duration_choices)
        start = rng.uniform(0.0, max(0.0, total_time - duration))
        note = pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=start + duration)
        instrument.notes.append(note)

    pm.instruments.append(instrument)
    return pm


def build_markov_chain(midi_paths: Iterable[Path]) -> tuple[np.ndarray, list[float]]:
    transition = np.zeros((PITCH_MAX - PITCH_MIN + 1, PITCH_MAX - PITCH_MIN + 1), dtype=np.float32)
    durations: list[float] = []

    for path in midi_paths:
        notes = extract_notes(path)
        if len(notes) < 2:
            continue
        notes_sorted = sorted(notes, key=lambda n: n.start)
        pitches = [note.pitch for note in notes_sorted]
        for i in range(len(pitches) - 1):
            src = pitches[i] - PITCH_MIN
            dst = pitches[i + 1] - PITCH_MIN
            if 0 <= src < transition.shape[0] and 0 <= dst < transition.shape[1]:
                transition[src, dst] += 1
        durations.extend([note.end - note.start for note in notes_sorted])

    row_sums = transition.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition = transition / row_sums
    return transition, durations


def markov_chain_generator(
    transition: np.ndarray,
    durations: list[float],
    num_notes: int = 200,
    seed: int = 42,
) -> pretty_midi.PrettyMIDI:
    rng = random.Random(seed)
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    pitch = rng.randint(PITCH_MIN, PITCH_MAX)
    start = 0.0
    duration_choices = durations or [0.25, 0.5, 1.0]

    for _ in range(num_notes):
        duration = rng.choice(duration_choices)
        note = pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=start + duration)
        instrument.notes.append(note)
        start += duration

        row = transition[pitch - PITCH_MIN]
        if row.sum() == 0:
            pitch = rng.randint(PITCH_MIN, PITCH_MAX)
        else:
            pitch = PITCH_MIN + int(rng.choices(range(len(row)), weights=row, k=1)[0])

    pm.instruments.append(instrument)
    return pm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--num-notes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = OUTPUT_ROOT / "generated_midis"
    out_dir.mkdir(parents=True, exist_ok=True)

    random_midi = random_note_generator(num_notes=args.num_notes, seed=args.seed)
    random_midi.write(str(out_dir / "baseline_random.mid"))

    ref_paths = list_midi_files(args.reference_dir)
    transition, durations = build_markov_chain(ref_paths)
    markov_midi = markov_chain_generator(transition, durations, num_notes=args.num_notes, seed=args.seed)
    markov_midi.write(str(out_dir / "baseline_markov.mid"))


if __name__ == "__main__":
    main()
