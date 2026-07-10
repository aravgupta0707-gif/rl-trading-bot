"""Portfolio allocation environment.

Observation = the raw (normalized) cross-sectional feature panel row for
day t, concatenated with the agent's current weights. The encoder is NOT
part of the environment -- it lives in the policy/value network as a
feature extractor (see training/train_ppo.py), so the same env works
unchanged whether the encoder is frozen, fine-tuned, or trained end-to-end.

Action = a real-valued logit vector over (assets + cash); softmax'd inside
step() into long-only weights summing to 1. Reward is portfolio return net
of transaction costs, computed from actual forward returns (never used as an
input feature -- only ever revealed after the fact, as a reward, which is
what RL is allowed to do).
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


def compute_forward_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """forward_returns.loc[t, ticker] = close[t+1]/close[t] - 1, i.e. the
    return realized by holding a position set at the close of day t."""
    close = prices.loc[:, (slice(None), "close")]
    close.columns = close.columns.get_level_values(0)
    close = close[tickers]
    return close.pct_change().shift(-1)


def align_features_and_returns(
    features: pd.DataFrame, forward_returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligns to a shared index and drops the final row (no forward return
    exists for the last day in the dataset)."""
    idx = features.index.intersection(forward_returns.index)
    features = features.loc[idx]
    forward_returns = forward_returns.loc[idx]
    valid = forward_returns.notna().all(axis=1)
    return features.loc[valid], forward_returns.loc[valid]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        forward_returns: np.ndarray,
        transaction_cost_bps: float,
        turnover_penalty: float = 0.0,
        reward_type: str = "log_return_net_cost",
    ):
        super().__init__()
        assert len(features) == len(forward_returns)
        self.features = features.astype(np.float32)
        self.forward_returns = forward_returns.astype(np.float32)
        self.n_steps = len(features)
        self.n_assets = forward_returns.shape[1]
        self.n_actions = self.n_assets + 1  # + cash
        self.cost_rate = transaction_cost_bps / 10000.0
        self.turnover_penalty = turnover_penalty
        self.reward_type = reward_type

        obs_dim = self.features.shape[1] + self.n_actions
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)

        self._t = 0
        self._weights = np.zeros(self.n_actions, dtype=np.float32)
        self._portfolio_value = 1.0

    def _obs(self) -> np.ndarray:
        obs = np.concatenate([self.features[self._t], self._weights]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._weights = np.zeros(self.n_actions, dtype=np.float32)
        self._weights[-1] = 1.0  # start fully in cash
        self._portfolio_value = 1.0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        weights = _softmax(np.asarray(action, dtype=np.float32))
        turnover = float(np.abs(weights - self._weights).sum())
        cost = turnover * self.cost_rate

        asset_returns = self.forward_returns[self._t]
        port_return = float((weights[:-1] * asset_returns).sum())  # cash return = 0
        net_return = port_return - cost

        if self.reward_type == "log_return_net_cost":
            reward = float(np.log1p(net_return) - self.turnover_penalty * turnover)
        elif self.reward_type == "excess_return_net_cost":
            # Reward = outperformance vs. an equal-weight hold of the same basket that day,
            # net of cost -- optimizes "beat the passive benchmark" directly, rather than
            # "maximize my own log-utility" (which has no incentive to beat a diversified
            # basket if it can't find a real rotation edge). Portfolio value/backtest metrics
            # below still track the real net_return, not the excess return -- this only
            # reshapes the training signal.
            benchmark_return = float(asset_returns.mean())
            reward = float((net_return - benchmark_return) - self.turnover_penalty * turnover)
        else:
            reward = float(net_return - self.turnover_penalty * turnover)

        self._portfolio_value *= 1.0 + net_return
        self._weights = weights

        terminated = False
        truncated = self._t >= self.n_steps - 1
        self._t = min(self._t + 1, self.n_steps - 1)

        info = {
            "portfolio_value": self._portfolio_value,
            "weights": weights.copy(),
            "turnover": turnover,
            "cost": cost,
            "net_return": net_return,
        }
        return self._obs(), reward, terminated, truncated, info


def make_env(features: pd.DataFrame, forward_returns: pd.DataFrame, env_cfg: dict) -> PortfolioEnv:
    return PortfolioEnv(
        features=features.to_numpy(),
        forward_returns=forward_returns.to_numpy(),
        transaction_cost_bps=env_cfg["transaction_cost_bps"],
        turnover_penalty=env_cfg["turnover_penalty"],
        reward_type=env_cfg["reward_type"],
    )
