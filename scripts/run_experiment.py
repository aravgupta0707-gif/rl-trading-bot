"""Entry point: python scripts/run_experiment.py --config configs/linear_frozen.yaml --run-name linear_frozen

Fetches/caches data, builds the feature panel, trains PPO with the encoder
variant specified by the config, and backtests on val + test splits.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import load_config
from data import fetch_etf_prices, fetch_macro
from envs import compute_forward_returns
from eval.backtest import run_backtest
from features import apply_normalizer, build_panel, fit_normalizer, walk_forward_split
from training.train_ppo import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="experiment")
    args = parser.parse_args()

    config = load_config(args.config)
    d, f, s = config["data"], config["features"], config["splits"]

    prices = fetch_etf_prices(d["etfs"], d["start_date"], d["end_date"], d["cache_dir"])
    macro = fetch_macro(d["macro"], d["start_date"], d["end_date"], d["cache_dir"])
    panel = build_panel(prices, macro, d["etfs"], f)

    train_df, val_df, test_df = walk_forward_split(panel, s["train_end"], s["val_end"])
    mean, std = fit_normalizer(train_df)
    train_n = apply_normalizer(train_df, mean, std)
    val_n = apply_normalizer(val_df, mean, std)
    test_n = apply_normalizer(test_df, mean, std)

    fwd = compute_forward_returns(prices, d["etfs"])

    run_dir = Path("runs") / args.run_name
    model = train(config, train_n, fwd, d["etfs"], panel, run_dir)

    print("\n=== Validation backtest ===")
    val_metrics = run_backtest(model, val_n, fwd, config["env"])
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== Test backtest ===")
    test_metrics = run_backtest(model, test_n, fwd, config["env"])
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    with open(run_dir / "metrics.json", "w") as fjson:
        json.dump({"val": val_metrics, "test": test_metrics}, fjson, indent=2)


if __name__ == "__main__":
    main()
