"""Test whether block-bootstrap-augmented training data closes the gap to
the benchmark, on the same fold2 (2022 bear)/fold5 (2025+) x 3-seed
comparison used for the PPO-optimization ablation. Doubles the effective
training history by prepending a same-length synthetic block (see
features/bootstrap.py) to each fold's real train split; val/test stay 100%
real, same protocol as every other rolling-validation run in this project.

Usage:
    python scripts/run_bootstrap_ablation.py --config configs/attention_frozen_excess_lowlr.yaml --seeds 0 1 2
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import load_config
from data import fetch_etf_prices, fetch_macro
from envs import compute_forward_returns
from eval.backtest import run_backtest
from features import apply_normalizer, build_panel, fit_normalizer, rolling_walk_forward_split
from features.bootstrap import generate_synthetic_panel
from training.train_ppo import train

from run_rolling_validation import FOLDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--folds", nargs="+", default=["fold2_test2021", "fold5_test2024"])
    parser.add_argument("--runs-dir", default="runs_bootstrap")
    parser.add_argument("--augment-multiplier", type=float, default=1.0,
                         help="synthetic length as a multiple of the real train panel length")
    parser.add_argument("--block-len", type=int, default=21)
    args = parser.parse_args()

    base = load_config(args.config)
    d, f = base["data"], base["features"]

    prices = fetch_etf_prices(d["etfs"], d["start_date"], d["end_date"], d["cache_dir"])
    macro = fetch_macro(d["macro"], d["start_date"], d["end_date"], d["cache_dir"])
    panel = build_panel(prices, macro, d["etfs"], f)
    fwd = compute_forward_returns(prices, d["etfs"])

    cell_name = Path(args.config).stem
    folds = [fo for fo in FOLDS if fo["name"] in args.folds]
    results = {}

    for fold in folds:
        train_df, val_df, test_df = rolling_walk_forward_split(
            panel, fold["train_end"], fold["val_end"], fold["test_end"]
        )
        target_length = int(len(train_df) * args.augment_multiplier)
        synth_panel, synth_fwd = generate_synthetic_panel(
            prices, macro, d["etfs"], f, fold["train_end"], target_length, args.block_len,
            seed=hash(fold["name"]) % (2**31),
        )
        aug_train_df = pd.concat([synth_panel, train_df])
        aug_fwd = pd.concat([synth_fwd, fwd])
        print(f"\n[{fold['name']}] real train rows={len(train_df)}  synthetic rows={len(synth_panel)}  "
              f"augmented total={len(aug_train_df)}")

        mean, std = fit_normalizer(aug_train_df)
        aug_train_n = apply_normalizer(aug_train_df, mean, std)
        val_n = apply_normalizer(val_df, mean, std)
        test_n = apply_normalizer(test_df, mean, std)

        for seed in args.seeds:
            config = load_config(args.config)
            config["ppo"]["seed"] = seed
            run_name = f"{cell_name}_bootstrap_{fold['name']}_seed{seed}"
            run_dir = Path(args.runs_dir) / run_name
            metrics_path = run_dir / "metrics.json"

            if metrics_path.exists():
                results[run_name] = json.loads(metrics_path.read_text())
                print(f"\n{run_name}: skipped (already exists)")
                continue

            print(f"\n{'=' * 60}\n{run_name}\n{'=' * 60}")
            model = train(config, aug_train_n, aug_fwd, d["etfs"], panel, run_dir)

            val_metrics = run_backtest(model, val_n, fwd, config["env"])
            test_metrics = run_backtest(model, test_n, fwd, config["env"])

            metrics = {"val": val_metrics, "test": test_metrics, "fold": fold["name"]}
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2))
            results[run_name] = metrics
            print(f"val sharpe={val_metrics['sharpe']:.3f}  test sharpe={test_metrics['sharpe']:.3f}")

    summary_path = Path(args.runs_dir) / f"{cell_name}_bootstrap_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nBootstrap ablation complete. Summary: {summary_path}\n")


if __name__ == "__main__":
    main()
