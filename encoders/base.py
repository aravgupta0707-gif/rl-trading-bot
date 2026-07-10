"""Common interface for all preprocessing encoders.

Every encoder maps the raw flat cross-sectional feature vector to a
compressed latent state. Each also implements `reconstruct`, used only for
the unsupervised pretraining objective (reconstruction MSE) -- there is no
future-return label anywhere in this module. What differs between the
frozen / finetune / end-to-end training regimes is not the architecture
here, but whether pretraining runs at all and whether gradients are allowed
to flow into these weights during PPO training (see training/train_ppo.py).
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Encoder(nn.Module, ABC):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor: ...

    def reconstruction_target(self, x: torch.Tensor) -> torch.Tensor:
        """Defaults to reconstructing the full input; override if an encoder
        only reconstructs part of it."""
        return x
