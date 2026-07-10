import torch
import torch.nn as nn

from .base import Encoder


class LinearEncoder(Encoder):
    """A single linear projection, no bias or nonlinearity.

    Trained via reconstruction MSE, a linear autoencoder's optimal weights
    span the same subspace as PCA's top components (Baldi & Hornik, 1989) --
    so in the frozen/finetune regimes this behaves like PCA, while still
    being an nn.Module so the same frozen/finetune/end-to-end training code
    path applies to it as to the MLP and attention encoders.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__(input_dim, latent_dim)
        self.encode_layer = nn.Linear(input_dim, latent_dim, bias=False)
        self.decode_layer = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_layer(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_layer(self.forward(x))
