"""Combine rolling-validation summaries for two cells plus the per-fold
equal-weight benchmark into one comparison table.

Usage:
    python scripts/rolling_summary_compare.py \
        --cells attention_frozen_excess attention_frozen \
        --benchmark runs_rolling/equal_weight_benchmark.json
"""
import argparse
import json
from pathlib import Path

FOLD_TEST_YEAR = {
    "fold1_test2020": "2021",
    "fold2_test2021": "2022",
    "fold3_test2022": "2023",
    "fold4_test2023": "2024",
    "fold5_test2024": "2025+",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--runs-dir", default="runs_rolling")
    parser.add_argument("--benchmark", default="runs_rolling/equal_weight_benchmark.json")
    parser.add_argument("--out", default="runs_rolling/comparison_summary.txt")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    benchmark = json.loads(Path(args.benchmark).read_text())

    cell_summaries = {}
    for cell in args.cells:
        summary_path = runs_dir / f"{cell}_rolling_summary.json"
        if not summary_path.exists():
            print(f"WARNING: {summary_path} not found, skipping {cell}")
            continue
        cell_summaries[cell] = json.loads(summary_path.read_text())

    lines = []
    header = f"{'fold':<14}{'test_yr':<9}{'benchmark_sharpe':>18}"
    for cell in cell_summaries:
        header += f"{cell + '_mean':>28}{cell + '_std':>12}"
    lines.append(header)

    for fold_name, test_yr in FOLD_TEST_YEAR.items():
        bench_sharpe = benchmark.get(fold_name, {}).get("sharpe", float("nan"))
        row = f"{fold_name:<14}{test_yr:<9}{bench_sharpe:>18.3f}"
        for cell, summary in cell_summaries.items():
            seed_sharpes = [
                m["test"]["sharpe"] for name, m in summary.items() if m["fold"] == fold_name
            ]
            if seed_sharpes:
                mean = sum(seed_sharpes) / len(seed_sharpes)
                std = (sum((s - mean) ** 2 for s in seed_sharpes) / len(seed_sharpes)) ** 0.5
                beats = "BEATS" if mean > bench_sharpe else "loses"
                row += f"{mean:>22.3f} {beats:<5}{std:>12.3f}"
            else:
                row += f"{'--':>28}{'--':>12}"
        lines.append(row)

    out_text = "\n".join(lines)
    print(out_text)
    Path(args.out).write_text(out_text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
