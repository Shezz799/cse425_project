from __future__ import annotations

import math
import torch
from torch import nn


class TransformerDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
        num_genres: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.genre_emb = nn.Embedding(num_genres, d_model) if num_genres > 1 else None

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        composer_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        if self.genre_emb is not None and composer_ids is not None:
            x = x + self.genre_emb(composer_ids).unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask

        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(encoded)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        pad_token_id: int = 0,
        eos_token_id: int | None = None,
        composer_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.max_len:
                input_ids = input_ids[:, -self.max_len :]
            logits = self.forward(input_ids, attention_mask=None, composer_ids=composer_ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k > 0:
                top_vals, top_idx = torch.topk(next_logits, top_k)
                probs = torch.softmax(top_vals, dim=-1)
                next_token = top_idx.gather(-1, torch.multinomial(probs, 1))
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids
