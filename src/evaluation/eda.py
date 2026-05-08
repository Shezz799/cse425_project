from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi

from src.config import OUTPUT_ROOT
from src.preprocessing.maestro import load_metadata, iter_midi_paths


def collect_notes(limit: int | None = None) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for midi_path in iter_midi_paths("train", limit=limit):
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception:
            continue
        for inst in midi.instruments:
            notes.extend(inst.notes)
    return notes


def plot_duration_histogram(df, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(df["duration"], bins=40, color="#3c4f76", alpha=0.8)
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pitch_histogram(notes: list[pretty_midi.Note], out_path: Path) -> None:
    pitches = [note.pitch for note in notes]
    if not pitches:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(pitches, bins=88, color="#4f7a57", alpha=0.8)
    plt.xlabel("MIDI Pitch")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_velocity_histogram(notes: list[pretty_midi.Note], out_path: Path) -> None:
    velocities = [note.velocity for note in notes]
    if not velocities:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(velocities, bins=32, color="#7a4f4f", alpha=0.8)
    plt.xlabel("Velocity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note-limit", type=int, default=200)
    args = parser.parse_args()

    out_dir = OUTPUT_ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata()
    plot_duration_histogram(df, out_dir / "eda_duration_hist.png")

    notes = collect_notes(limit=args.note_limit)
    plot_pitch_histogram(notes, out_dir / "eda_pitch_hist.png")
    plot_velocity_histogram(notes, out_dir / "eda_velocity_hist.png")


if __name__ == "__main__":
    main()
