from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 88,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.latent_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        z = self.latent_fc(h_n[-1])
        return z

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(z_repeated)
        logits = self.output_fc(decoded)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z, x.size(1))


class LSTMVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 88,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.mu_fc = nn.Linear(hidden_dim, latent_dim)
        self.logvar_fc = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]
        mu = self.mu_fc(h)
        logvar = self.logvar_fc(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(z_repeated)
        logits = self.output_fc(decoded)
        return logits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x.size(1))
        return logits, mu, logvar
