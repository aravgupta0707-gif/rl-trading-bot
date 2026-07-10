"""Deeper diagnostics for a trained run: full daily trajectories (equity,
weights, turnover) rather than just summary stats, plus an equal-weight
benchmark for comparison. Writes JSON to runs/<name>/diagnostics.json.

Usage:
    python scripts/diagnostics.py --run-name attention_frozen_seed0
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
from features import apply_normalizer, build_panel, fit_normalizer, walk_forward_split


def trajectory(model: PPO, features: pd.DataFrame, forward_returns: pd.DataFrame, env_cfg: dict, tickers: list[str]) -> dict:
    features, forward_returns = align_features_and_returns(features, forward_returns)
    env = make_env(features, forward_returns, env_cfg)
    dates = features.index

    obs, _ = env.reset()
    rows = []
    done = False
    t = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({
            "date": str(dates[t].date()),
            "portfolio_value": info["portfolio_value"],
            "turnover": info["turnover"],
            "net_return": info["net_return"],
            "weights": info["weights"].tolist(),
        })
        done = terminated or truncated
        t += 1

    equal_weight = 1.0
    ew_curve = []
    n = len(tickers)
    for i in range(len(forward_returns)):
        r = float(forward_returns.iloc[i].mean())
        equal_weight *= (1 + r)
        ew_curve.append(equal_weight)

    return {
        "dates": [r["date"] for r in rows],
        "portfolio_value": [r["portfolio_value"] for r in rows],
        "equal_weight_value": ew_curve,
        "turnover": [r["turnover"] for r in rows],
        "net_return": [r["net_return"] for r in rows],
        "weights": [r["weights"] for r in rows],
        "asset_names": tickers + ["cash"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    run_dir = Path(args.runs_dir) / args.run_name
    config = json.loads((run_dir / "config.json").read_text())
    d, f, s = config["data"], config["features"], config["splits"]

    prices = fetch_etf_prices(d["etfs"], d["start_date"], d["end_date"], d["cache_dir"])
    macro = fetch_macro(d["macro"], d["start_date"], d["end_date"], d["cache_dir"])
    panel = build_panel(prices, macro, d["etfs"], f)

    train_df, val_df, test_df = walk_forward_split(panel, s["train_end"], s["val_end"])
    mean, std = fit_normalizer(train_df)
    val_n = apply_normalizer(val_df, mean, std)
    test_n = apply_normalizer(test_df, mean, std)

    fwd = compute_forward_returns(prices, d["etfs"])

    model = PPO.load(str(run_dir / "model.zip"))

    diag = {
        "val": trajectory(model, val_n, fwd, config["env"], d["etfs"]),
        "test": trajectory(model, test_n, fwd, config["env"], d["etfs"]),
    }

    out_path = run_dir / "diagnostics.json"
    out_path.write_text(json.dumps(diag))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
