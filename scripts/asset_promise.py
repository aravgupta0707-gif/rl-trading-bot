"""Which individual ETFs actually show promise? Two independent views,
neither requiring new training:

1. Revealed preference -- rerun the already-trained attention_frozen_excess
   rolling models (5 folds x 3 seeds, runs_rolling/) deterministically over
   their real test split, and record per-asset average weight. An asset the
   policy consistently allocates near-zero to across folds/seeds isn't one
   it found anything to do with; an asset it consistently overweights is.

2. Model-free cross-check -- plain buy-and-hold Sharpe per individual ETF
   over the same test windows, computed directly from price data, no RL
   involved. Guards against "the policy likes it" meaning "the policy is
   confused" rather than "there's real promise there."

Train-period leakage is not a concern here: this only touches each fold's
already-fixed *test* window, exactly like every other backtest in this
project -- it's reading out results, not selecting features/hyperparameters
on test data.

Usage:
    python scripts/asset_promise.py --config configs/attention_frozen_excess.yaml
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
from eval.metrics import sharpe_ratio
from features import apply_normalizer, build_panel, fit_normalizer, rolling_walk_forward_split

from run_rolling_validation import FOLDS


def weight_trajectory(model: PPO, features: pd.DataFrame, forward_returns: pd.DataFrame, env_cfg: dict) -> np.ndarray:
    features, forward_returns = align_features_and_returns(features, forward_returns)
    env = make_env(features, forward_returns, env_cfg)
    obs, _ = env.reset()
    weights = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        weights.append(info["weights"])
        done = terminated or truncated
    return np.array(weights)  # (n_days, n_assets + 1) -- last col is cash


def buy_and_hold_sharpe(fwd: pd.DataFrame, tickers: list[str]) -> dict:
    return {t: sharpe_ratio(fwd[t].dropna()) for t in tickers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/attention_frozen_excess.yaml")
    parser.add_argument("--runs-dir", default="runs_rolling")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    base = load_config(args.config)
    d, f = base["data"], base["features"]
    cell_name = Path(args.config).stem

    prices = fetch_etf_prices(d["etfs"], d["start_date"], d["end_date"], d["cache_dir"])
    macro = fetch_macro(d["macro"], d["start_date"], d["end_date"], d["cache_dir"])
    panel = build_panel(prices, macro, d["etfs"], f)
    fwd = compute_forward_returns(prices, d["etfs"])

    tickers = d["etfs"]
    weight_rows = []   # one row per (fold, seed), columns = tickers + cash
    bh_rows = []        # one row per fold, columns = tickers

    for fold in FOLDS:
        train_df, val_df, test_df = rolling_walk_forward_split(
            panel, fold["train_end"], fold["val_end"], fold["test_end"]
        )
        mean, std = fit_normalizer(train_df)
        test_n = apply_normalizer(test_df, mean, std)

        _, test_fwd = align_features_and_returns(test_df, fwd)
        bh = buy_and_hold_sharpe(test_fwd, tickers)
        bh_rows.append({"fold": fold["name"], **bh})

        for seed in args.seeds:
            run_dir = Path(args.runs_dir) / f"{cell_name}_{fold['name']}_seed{seed}"
            model_path = run_dir / "model.zip"
            if not model_path.exists():
                print(f"SKIP {run_dir}: no model.zip")
                continue
            model = PPO.load(str(model_path))
            traj = weight_trajectory(model, test_n, fwd, base["env"])
            avg_weights = traj.mean(axis=0)  # (n_assets+1,)
            row = {"fold": fold["name"], "seed": seed}
            row.update({t: avg_weights[i] for i, t in enumerate(tickers)})
            row["cash"] = avg_weights[-1]
            weight_rows.append(row)

    weight_df = pd.DataFrame(weight_rows)
    bh_df = pd.DataFrame(bh_rows).set_index("fold")

    print("\n=== Revealed preference: mean allocation weight (attention_frozen_excess, all folds x seeds) ===")
    avg_weight_per_asset = weight_df[tickers + ["cash"]].mean().sort_values(ascending=False)
    for t, w in avg_weight_per_asset.items():
        print(f"  {t:<8} avg_weight={w:.4f}")

    print("\n=== Model-free cross-check: buy-and-hold Sharpe per asset, per fold ===")
    print(bh_df[tickers].round(2).to_string())
    print("\nmean buy-and-hold Sharpe across folds:")
    print(bh_df[tickers].mean().sort_values(ascending=False).round(3).to_string())

    out = {
        "avg_weight_per_asset": avg_weight_per_asset.to_dict(),
        "buy_and_hold_sharpe_by_fold": bh_df.to_dict(orient="index"),
        "buy_and_hold_sharpe_mean": bh_df[tickers].mean().to_dict(),
    }
    out_path = Path(args.runs_dir) / f"{cell_name}_asset_promise.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
