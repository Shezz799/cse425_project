from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi


def extract_notes(midi_path: Path) -> list[pretty_midi.Note]:
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return []
    notes: list[pretty_midi.Note] = []
    for inst in midi.instruments:
        notes.extend(inst.notes)
    return notes


def pitch_histogram(notes: list[pretty_midi.Note]) -> np.ndarray:
    if not notes:
        return np.zeros(12, dtype=np.float32)
    hist = np.zeros(12, dtype=np.float32)
    for note in notes:
        hist[note.pitch % 12] += 1
    hist /= hist.sum()
    return hist


def pitch_histogram_similarity(gen_notes: list[pretty_midi.Note], ref_hist: np.ndarray) -> float:
    gen_hist = pitch_histogram(gen_notes)
    return float(np.abs(gen_hist - ref_hist).sum())


def rhythm_diversity(notes: list[pretty_midi.Note]) -> float:
    if not notes:
        return 0.0
    durations = [max(0.0, note.end - note.start) for note in notes]
    quantized = [round(d / 0.05) * 0.05 for d in durations]
    return len(set(quantized)) / max(len(quantized), 1)


def repetition_ratio(notes: list[pretty_midi.Note], n: int = 4) -> float:
    if len(notes) < n:
        return 0.0
    notes_sorted = sorted(notes, key=lambda n: n.start)
    pitches = [note.pitch for note in notes_sorted]
    ngrams = [tuple(pitches[i : i + n]) for i in range(len(pitches) - n + 1)]
    if not ngrams:
        return 0.0
    counts = {}
    for ng in ngrams:
        counts[ng] = counts.get(ng, 0) + 1
    repeated = sum(1 for count in counts.values() if count > 1)
    return repeated / max(len(ngrams), 1)


def build_reference_histogram(ref_paths: Iterable[Path]) -> np.ndarray:
    hist = np.zeros(12, dtype=np.float32)
    total_notes = 0
    for path in ref_paths:
        notes = extract_notes(path)
        for note in notes:
            hist[note.pitch % 12] += 1
        total_notes += len(notes)
    if total_notes == 0:
        return hist
    return hist / total_notes


def list_midi_files(folder: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(folder.rglob("*.mid")) + sorted(folder.rglob("*.midi"))
    if limit is not None:
        paths = paths[:limit]
    return paths


def compute_metrics(gen_dir: Path, ref_dir: Path, limit: int | None = None) -> dict:
    gen_paths = list_midi_files(gen_dir, limit)
    ref_paths = list_midi_files(ref_dir, limit)
    ref_hist = build_reference_histogram(ref_paths)

    pitch_scores = []
    rhythm_scores = []
    repetition_scores = []

    for path in gen_paths:
        notes = extract_notes(path)
        pitch_scores.append(pitch_histogram_similarity(notes, ref_hist))
        rhythm_scores.append(rhythm_diversity(notes))
        repetition_scores.append(repetition_ratio(notes))

    return {
        "pitch_histogram_similarity": float(np.mean(pitch_scores)) if pitch_scores else 0.0,
        "rhythm_diversity": float(np.mean(rhythm_scores)) if rhythm_scores else 0.0,
        "repetition_ratio": float(np.mean(repetition_scores)) if repetition_scores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    metrics = compute_metrics(args.generated_dir, args.reference_dir, limit=args.limit)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
