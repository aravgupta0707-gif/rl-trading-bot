"""Equal-weight benchmark, computed per rolling-validation fold (not one
static split), so each fold's RL result can be compared against its own
regime's benchmark rather than the single historical 1.395 figure.

No training involved -- just fetch (cached) -> panel -> per-fold split ->
daily-rebalanced equal-weight return series -> eval.metrics.summarize().

Usage:
    python scripts/rolling_benchmark.py --config configs/attention_frozen_excess.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import load_config
from data import fetch_etf_prices, fetch_macro
from envs import align_features_and_returns, compute_forward_returns
from eval.metrics import summarize
from features import build_panel, rolling_walk_forward_split

from run_rolling_validation import FOLDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="runs_rolling/equal_weight_benchmark.json")
    args = parser.parse_args()

    base = load_config(args.config)
    d, f = base["data"], base["features"]

    prices = fetch_etf_prices(d["etfs"], d["start_date"], d["end_date"], d["cache_dir"])
    macro = fetch_macro(d["macro"], d["start_date"], d["end_date"], d["cache_dir"])
    panel = build_panel(prices, macro, d["etfs"], f)
    fwd = compute_forward_returns(prices, d["etfs"])

    results = {}
    for fold in FOLDS:
        _, _, test_df = rolling_walk_forward_split(panel, fold["train_end"], fold["val_end"], fold["test_end"])
        _, fwd_aligned = align_features_and_returns(test_df, fwd)

        returns = fwd_aligned.mean(axis=1)  # equal-weight, daily-rebalanced, no cost
        values = (1 + returns).cumprod()
        turnovers = pd.Series(0.0, index=returns.index)  # rebalanced to equal-weight daily, cost not modeled here

        metrics = summarize(returns, values, turnovers)
        results[fold["name"]] = metrics
        print(f"{fold['name']:<16} test_sharpe={metrics['sharpe']:>7.3f}  "
              f"total_return={metrics['total_return']:>7.3f}  max_dd={metrics['max_drawdown']:>7.3f}  "
              f"n_days={metrics['n_days']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
