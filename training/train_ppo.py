"""Wires an encoder (any architecture, any training regime) into a PPO
policy's feature extractor, and runs training.

The env always emits the same observation (raw normalized market features +
current weights); what changes across the 3x3 grid is only:
  - which Encoder subclass is used (linear / mlp / attention)
  - whether it's pretrained via reconstruction first (frozen / finetune),
    and if so whether its weights are then frozen or left trainable
  - or skipped entirely and trained purely by the PPO loss (e2e)
"""
import json
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from encoders import Encoder, build_encoder
from envs import align_features_and_returns, compute_forward_returns, make_env
from features import etf_feature_columns, macro_feature_columns
from training.pretrain_encoder import freeze, pretrain_encoder


class EncoderFeaturesExtractor(BaseFeaturesExtractor):
    """Splits the env observation into (market features, current weights),
    runs only the market features through the encoder, and concatenates the
    latent with the (untouched) weights before SB3's policy/value heads."""

    def __init__(self, observation_space, encoder: Encoder, n_weight_dims: int):
        features_dim = encoder.latent_dim + n_weight_dims
        super().__init__(observation_space, features_dim=features_dim)
        self.encoder = encoder
        self.market_dim = encoder.input_dim
        self.n_weight_dims = n_weight_dims

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        market = observations[:, : self.market_dim]
        weights = observations[:, self.market_dim :]
        latent = self.encoder(market)
        return torch.cat([latent, weights], dim=1)


def build_encoder_for_config(encoder_cfg: dict, tickers: list[str], panel: pd.DataFrame) -> Encoder:
    encoder_type = encoder_cfg["type"]
    latent_dim = encoder_cfg["latent_dim"]

    if encoder_type == "attention":
        per_asset_dim = len(etf_feature_columns(panel, tickers[0]))
        macro_dim = len(macro_feature_columns(panel))
        return build_encoder(
            "attention",
            n_assets=len(tickers),
            per_asset_dim=per_asset_dim,
            macro_dim=macro_dim,
            latent_dim=latent_dim,
        )

    input_dim = panel.shape[1]
    return build_encoder(encoder_type, input_dim=input_dim, latent_dim=latent_dim)


def prepare_encoder(
    encoder_cfg: dict,
    encoder: Encoder,
    train_features: pd.DataFrame,
) -> Encoder:
    regime = encoder_cfg["regime"]
    if regime == "e2e":
        return encoder  # random init, trained purely by PPO
    encoder = pretrain_encoder(encoder, train_features.to_numpy())
    if regime == "frozen":
        encoder = freeze(encoder)
    elif regime == "finetune":
        pass  # pretrained, but left trainable for PPO to keep adjusting
    else:
        raise ValueError(f"unknown encoder regime: {regime}")
    return encoder


def train(
    config: dict,
    train_features: pd.DataFrame,
    train_forward_returns: pd.DataFrame,
    tickers: list[str],
    panel_for_shapes: pd.DataFrame,
    run_dir: Path,
) -> PPO:
    train_features, train_forward_returns = align_features_and_returns(train_features, train_forward_returns)

    encoder = build_encoder_for_config(config["encoder"], tickers, panel_for_shapes)
    encoder = prepare_encoder(config["encoder"], encoder, train_features)

    env = DummyVecEnv([lambda: make_env(train_features, train_forward_returns, config["env"])])

    n_actions = len(tickers) + 1
    ppo_cfg = config["ppo"]
    pi_arch = ppo_cfg.get("net_arch", [32])
    policy_kwargs = dict(
        features_extractor_class=EncoderFeaturesExtractor,
        features_extractor_kwargs=dict(encoder=encoder, n_weight_dims=n_actions),
        net_arch=dict(pi=pi_arch, vf=pi_arch),
    )

    pretrained_state = {k: v.clone() for k, v in encoder.state_dict().items()}

    ppo_kwargs = dict(
        policy_kwargs=policy_kwargs,
        seed=ppo_cfg["seed"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        verbose=1,
    )
    if "learning_rate" in ppo_cfg:
        ppo_kwargs["learning_rate"] = ppo_cfg["learning_rate"]

    model = PPO("MlpPolicy", env, **ppo_kwargs)
    # ActorCriticPolicy._build() re-applies SB3's default orthogonal init to every
    # nn.Linear/nn.Conv2d in the whole features_extractor (including our encoder),
    # which silently discards prepare_encoder()'s pretrained/frozen weights. Restore
    # them post-construction so frozen/finetune actually start from the pretrained
    # encoder instead of a fresh random one.
    model.policy.features_extractor.encoder.load_state_dict(pretrained_state)
    model.learn(total_timesteps=ppo_cfg["total_timesteps"])

    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "model.zip"))
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return model
