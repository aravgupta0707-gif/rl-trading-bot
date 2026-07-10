"""Zero-retrain check: does averaging the 3 already-trained seed models'
action logits (before softmax) at each timestep reduce variance / improve
fold-level Sharpe vs. any single seed? Uses saved model.zip files from
runs_rolling/, no training involved.

Usage:
    python scripts/ensemble_backtest.py --cells attention_frozen_excess attention_frozen
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import load_config
from data import fetch_etf_prices, fetch_macro
from envs import align_features_and_returns, compute_forward_returns, make_env
from eval.metrics import summarize
from features import apply_normalizer, build_panel, fit_normalizer, rolling_walk_forward_split

from run_rolling_validation import FOLDS


def ensemble_backtest(models: list[PPO], features: pd.DataFrame, forward_returns: pd.DataFrame, env_cfg: dict) -> dict:
    features, forward_returns = align_features_and_returns(features, forward_returns)
    env = make_env(features, forward_returns, env_cfg)

    obs, _ = env.reset()
    net_returns, portfolio_values, turnovers = [], [], []
    done = False
    while not done:
        actions = [m.predict(obs, deterministic=True)[0] for m in models]
        avg_action = np.mean(actions, axis=0)
        obs, reward, terminated, truncated, info = env.step(avg_action)
        net_returns.append(info["net_return"])
        portfolio_values.append(info["portfolio_value"])
        turnovers.append(info["turnover"])
        done = terminated or truncated

    return summarize(pd.Series(net_returns), pd.Series(portfolio_values), pd.Series(turnovers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--config", default="configs/attention_frozen_excess.yaml")
    parser.add_argument("--runs-dir", default="runs_rolling")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default="runs_rolling/ensemble_summary.json")
    args = parser.parse_args()

    base = load_config(args.config)
    d, f = base["data"], base["features"]

    prices = fetch_etf_prices(d["etfs"], d["start_date"], d["end_date"], d["cache_dir"])
    macro = fetch_macro(d["macro"], d["start_date"], d["end_date"], d["cache_dir"])
    panel = build_panel(prices, macro, d["etfs"], f)
    fwd = compute_forward_returns(prices, d["etfs"])

    runs_dir = Path(args.runs_dir)
    results = {}

    for cell in args.cells:
        cell_config = load_config(f"configs/{cell}.yaml")
        for fold in FOLDS:
            train_df, val_df, test_df = rolling_walk_forward_split(
                panel, fold["train_end"], fold["val_end"], fold["test_end"]
            )
            mean, std = fit_normalizer(train_df)
            test_n = apply_normalizer(test_df, mean, std)

            model_paths = [
                runs_dir / f"{cell}_{fold['name']}_seed{s}" / "model.zip" for s in args.seeds
            ]
            if not all(p.exists() for p in model_paths):
                print(f"SKIP {cell} {fold['name']}: missing model(s)")
                continue
            models = [PPO.load(str(p)) for p in model_paths]

            metrics = ensemble_backtest(models, test_n, fwd, cell_config["env"])
            key = f"{cell}_{fold['name']}"
            results[key] = metrics
            print(f"{key:<45} ensemble_test_sharpe={metrics['sharpe']:>7.3f}  "
                  f"max_dd={metrics['max_drawdown']:>7.3f}  turnover={metrics['avg_daily_turnover']:.4f}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
