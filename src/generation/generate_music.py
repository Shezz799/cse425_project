from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import OUTPUT_ROOT
from src.generation.midi_io import save_pianoroll_midi, tokens_to_midi
from src.generation.sample_latent import decode_latent
from src.models.lstm_ae import LSTMAutoencoder, LSTMVAE
from src.models.transformer import TransformerDecoderModel


def generate_from_autoencoder(checkpoint: Path, out_dir: Path, num_samples: int = 5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    z = torch.randn(num_samples, model.latent_fc.out_features, device=device)
    samples = decode_latent(model, z)
    for i, sample in enumerate(samples):
        save_pianoroll_midi(sample, out_dir / f"task1_sample_{i+1}.mid")


def generate_from_vae(checkpoint: Path, out_dir: Path, num_samples: int = 8) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    z = torch.randn(num_samples, model.mu_fc.out_features, device=device)
    samples = decode_latent(model, z)
    for i, sample in enumerate(samples):
        save_pianoroll_midi(sample, out_dir / f"task2_sample_{i+1}.mid")


def generate_from_transformer(
    checkpoint: Path,
    out_dir: Path,
    vocab_size: int,
    start_token_id: int,
    num_samples: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 50,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerDecoderModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    for i in range(num_samples):
        input_ids = torch.tensor([[start_token_id]], device=device)
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        tokens_to_midi(generated[0].tolist(), out_dir / f"task3_sample_{i+1}.mid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ae", "vae", "transformer"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--start-token-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    out_dir = OUTPUT_ROOT / "generated_midis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "ae":
        generate_from_autoencoder(args.checkpoint, out_dir, num_samples=args.num_samples)
    elif args.task == "vae":
        generate_from_vae(args.checkpoint, out_dir, num_samples=args.num_samples)
    else:
        if args.vocab_size <= 0:
            raise ValueError("--vocab-size is required for transformer generation.")
        generate_from_transformer(
            args.checkpoint,
            out_dir,
            vocab_size=args.vocab_size,
            start_token_id=args.start_token_id,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
