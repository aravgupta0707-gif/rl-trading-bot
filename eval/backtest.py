import pandas as pd
from stable_baselines3 import PPO

from envs import align_features_and_returns, make_env

from .metrics import summarize


def run_backtest(model: PPO, features: pd.DataFrame, forward_returns: pd.DataFrame, env_cfg: dict) -> dict:
    features, forward_returns = align_features_and_returns(features, forward_returns)
    env = make_env(features, forward_returns, env_cfg)

    obs, _ = env.reset()
    net_returns, portfolio_values, turnovers = [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        net_returns.append(info["net_return"])
        portfolio_values.append(info["portfolio_value"])
        turnovers.append(info["turnover"])
        done = terminated or truncated

    return summarize(pd.Series(net_returns), pd.Series(portfolio_values), pd.Series(turnovers))
