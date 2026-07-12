"""Rolling walk-forward validation: retrain and backtest one cell across
several sequential (train, val, test) folds spanning different market
regimes, instead of a single static split. Answers "is this cell's edge
stable across regimes" rather than "did it win on one particular window."

Usage:
    python scripts/run_rolling_validation.py --config configs/attention_frozen.yaml --seeds 0 1 2
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
from features import apply_normalizer, build_panel, fit_normalizer, rolling_walk_forward_split
from training.train_ppo import train

# Expanding train window, 1-year val, 1-year test per fold (last fold's test
# runs to the end of available data). Chosen to span distinct regimes: 2020
# COVID crash/recovery, 2021 bull, 2022 rate-hike bear, 2023 recovery, 2024+.
FOLDS = [
    {"name": "fold1_test2020", "train_end": "2019-12-31", "val_end": "2020-12-31", "test_end": "2021-12-31"},
    {"name": "fold2_test2021", "train_end": "2020-12-31", "val_end": "2021-12-31", "test_end": "2022-12-31"},
    {"name": "fold3_test2022", "train_end": "2021-12-31", "val_end": "2022-12-31", "test_end": "2023-12-31"},
    {"name": "fold4_test2023", "train_end": "2022-12-31", "val_end": "2023-12-31", "test_end": "2024-12-31"},
    {"name": "fold5_test2024", "train_end": "2023-12-31", "val_end": "2024-12-31", "test_end": "2099-12-31"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--runs-dir", default="runs_rolling")
    parser.add_argument("--folds", nargs="+", default=None,
                         help="subset of fold names to run (default: all 5), e.g. --folds fold2_test2021 fold5_test2024")
    args = parser.parse_args()

    base = load_config(args.config)
    d, f = base["data"], base["features"]
    context_etfs = d.get("context_etfs", [])

    prices = fetch_etf_prices(d["etfs"] + context_etfs, d["start_date"], d["end_date"], d["cache_dir"])
    macro = fetch_macro(d["macro"], d["start_date"], d["end_date"], d["cache_dir"])
    panel = build_panel(prices, macro, d["etfs"], f, context_tickers=context_etfs)
    fwd = compute_forward_returns(prices, d["etfs"])

    cell_name = Path(args.config).stem
    results = {}

    folds_to_run = FOLDS if args.folds is None else [fo for fo in FOLDS if fo["name"] in args.folds]

    for fold in folds_to_run:
        train_df, val_df, test_df = rolling_walk_forward_split(
            panel, fold["train_end"], fold["val_end"], fold["test_end"]
        )
        mean, std = fit_normalizer(train_df)
        train_n = apply_normalizer(train_df, mean, std)
        val_n = apply_normalizer(val_df, mean, std)
        test_n = apply_normalizer(test_df, mean, std)

        for seed in args.seeds:
            config = load_config(args.config)
            config["ppo"]["seed"] = seed
            run_name = f"{cell_name}_{fold['name']}_seed{seed}"
            run_dir = Path(args.runs_dir) / run_name
            metrics_path = run_dir / "metrics.json"

            if metrics_path.exists():
                results[run_name] = json.loads(metrics_path.read_text())
                print(f"\n{run_name}: skipped (already exists)")
                continue

            print(f"\n{'=' * 60}\n{run_name}  (train<={fold['train_end']}  test<={fold['test_end']})\n{'=' * 60}")

            model = train(config, train_n, fwd, d["etfs"], panel, run_dir)

            val_metrics = run_backtest(model, val_n, fwd, config["env"])
            test_metrics = run_backtest(model, test_n, fwd, config["env"])

            metrics = {"val": val_metrics, "test": test_metrics, "fold": fold["name"]}
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, indent=2))
            results[run_name] = metrics
            print(f"val sharpe={val_metrics['sharpe']:.3f}  test sharpe={test_metrics['sharpe']:.3f}")

    summary_path = Path(args.runs_dir) / f"{cell_name}_rolling_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))

    print(f"\nRolling validation complete. Summary: {summary_path}\n")
    header = f"{'run':<38}{'test_sharpe':>12}{'test_maxdd':>12}{'test_turnover':>14}"
    print(header)
    for name, m in results.items():
        print(f"{name:<38}{m['test']['sharpe']:>12.3f}{m['test']['max_drawdown']:>12.3f}{m['test']['avg_daily_turnover']:>14.4f}")


if __name__ == "__main__":
    main()
