import torch
import torch.nn as nn

from .base import Encoder


class MLPEncoder(Encoder):
    """Nonlinear autoencoder: input -> hidden -> latent -> hidden -> input."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__(input_dim, latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.forward(x))
