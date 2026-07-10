"""Unsupervised pretraining for the frozen/finetune encoder regimes.

Trains purely on reconstruction MSE over the train-split feature panel --
no future-return label enters anywhere in this file.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from encoders import Encoder


def pretrain_encoder(
    encoder: Encoder,
    train_features: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-5,
    verbose: bool = True,
) -> Encoder:
    x = torch.tensor(train_features, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    encoder.train()
    log_every = max(1, epochs // 10)
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = encoder.reconstruct(batch)
            target = encoder.reconstruction_target(batch)
            loss = torch.nn.functional.mse_loss(recon, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        avg_loss = total_loss / len(x)
        if verbose and (epoch + 1) % log_every == 0:
            print(f"  pretrain epoch {epoch + 1}/{epochs}  recon_mse={avg_loss:.5f}")

    encoder.eval()
    return encoder


def freeze(encoder: Encoder) -> Encoder:
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder
