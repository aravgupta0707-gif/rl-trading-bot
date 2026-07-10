"""Run the full encoder x regime grid (optionally x multiple seeds).

Builds the feature panel once and reuses it across all cells (fetch is
cached anyway, but this also skips rebuilding the panel 9x).

Usage:
    python scripts/run_grid.py
    python scripts/run_grid.py --seeds 0 1 2
    python scripts/run_grid.py --configs configs/linear_frozen.yaml configs/mlp_e2e.yaml --seeds 0
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

GRID_CONFIGS = [
    "configs/linear_frozen.yaml", "configs/linear_finetune.yaml", "configs/linear_e2e.yaml",
    "configs/mlp_frozen.yaml", "configs/mlp_finetune.yaml", "configs/mlp_e2e.yaml",
    "configs/attention_frozen.yaml", "configs/attention_finetune.yaml", "configs/attention_e2e.yaml",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=GRID_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--skip-existing", action="store_true", help="skip cells with an existing metrics.json")
    args = parser.parse_args()

    base = load_config(args.configs[0])
    d, f, s = base["data"], base["features"], base["splits"]

    prices = fetch_etf_prices(d["etfs"], d["start_date"], d["end_date"], d["cache_dir"])
    macro = fetch_macro(d["macro"], d["start_date"], d["end_date"], d["cache_dir"])
    panel = build_panel(prices, macro, d["etfs"], f)

    train_df, val_df, test_df = walk_forward_split(panel, s["train_end"], s["val_end"])
    mean, std = fit_normalizer(train_df)
    train_n = apply_normalizer(train_df, mean, std)
    val_n = apply_normalizer(val_df, mean, std)
    test_n = apply_normalizer(test_df, mean, std)

    fwd = compute_forward_returns(prices, d["etfs"])

    results = {}
    for config_path in args.configs:
        for seed in args.seeds:
            config = load_config(config_path)
            config["ppo"]["seed"] = seed
            run_name = f"{Path(config_path).stem}_seed{seed}"
            run_dir = Path(args.runs_dir) / run_name
            metrics_path = run_dir / "metrics.json"

            if args.skip_existing and metrics_path.exists():
                with open(metrics_path) as fjson:
                    results[run_name] = json.load(fjson)
                print(f"\n{run_name}: skipped (metrics.json already exists)")
                continue

            print(f"\n{'=' * 60}\n{run_name}\n{'=' * 60}")

            model = train(config, train_n, fwd, d["etfs"], panel, run_dir)

            val_metrics = run_backtest(model, val_n, fwd, config["env"])
            test_metrics = run_backtest(model, test_n, fwd, config["env"])

            with open(run_dir / "metrics.json", "w") as fjson:
                json.dump({"val": val_metrics, "test": test_metrics}, fjson, indent=2)

            results[run_name] = {"val": val_metrics, "test": test_metrics}
            print(f"val sharpe={val_metrics['sharpe']:.3f}  test sharpe={test_metrics['sharpe']:.3f}")

    summary_path = Path(args.runs_dir) / "grid_summary.json"
    with open(summary_path, "w") as fjson:
        json.dump(results, fjson, indent=2)

    header = f"{'run':<30}{'val_sharpe':>12}{'test_sharpe':>12}{'test_maxdd':>12}{'test_turnover':>14}"
    print(f"\nGrid complete. Summary written to {summary_path}\n")
    print(header)
    for run_name, m in results.items():
        print(
            f"{run_name:<30}{m['val']['sharpe']:>12.3f}{m['test']['sharpe']:>12.3f}"
            f"{m['test']['max_drawdown']:>12.3f}{m['test']['avg_daily_turnover']:>14.4f}"
        )


if __name__ == "__main__":
    main()
