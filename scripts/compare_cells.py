"""Compare two cells fold by fold with an explicit noise test.

This project has spent eight rounds asking "is this difference real?" and
answering it by eyeballing means against seed spreads. Round 8 showed how badly
that fails: block-bootstrap augmentation looked like it crossed the benchmark on
the 2022 bear fold at 3 seeds (-0.423 vs -0.436) and clearly did not at 10
(-0.510). Any claim about a lever needs a standard error, not a gap.

Reports, per fold: each cell's mean and sd, the difference, Welch's standard
error and t statistic, and the same for each cell against its benchmark (where
the benchmark is a fixed series, so its uncertainty is zero and the standard
error is the cell's alone).

Welch rather than paired: seed i of a bootstrap-augmented cell trains on
different synthetic data than seed i of the baseline, so the seeds are not
matched pairs.

Usage:
    python scripts/compare_cells.py attention_frozen_excess_lowlr \
        attention_frozen_excess_lowlr_bootstrap \
        --benchmark runs_rolling/equal_weight_benchmark_round8.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_results import collect, load_bench, RUNS_DIRS, FOLD_TEST_YEAR


def stats(vals):
    n = len(vals)
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    return mean, sd, n, (sd / (n ** 0.5) if n else float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cell_a")
    p.add_argument("cell_b")
    p.add_argument("--benchmark", default="runs_rolling/equal_weight_benchmark.json")
    p.add_argument("--runs-dirs", nargs="+", default=RUNS_DIRS)
    p.add_argument("--t-threshold", type=float, default=2.0,
                   help="|t| above which a difference is called distinguishable (default 2.0, ~95%%)")
    args = p.parse_args()

    data = collect(args.runs_dirs)
    for c in (args.cell_a, args.cell_b):
        if c not in data:
            raise SystemExit(f"no runs found for cell {c}")
    bench = load_bench(Path(args.benchmark)) or {}

    print(f"A = {args.cell_a}")
    print(f"B = {args.cell_b}")
    print(f"benchmark = {args.benchmark}\n")
    hdr = (f"{'fold':<18}{'A mean(sd,n)':>18}{'B mean(sd,n)':>18}"
           f"{'B-A':>8}{'SE':>7}{'t':>7}{'verdict':>16}")
    print(hdr)
    print("-" * len(hdr))

    for fold in sorted(set(data[args.cell_a]) & set(data[args.cell_b])):
        a = [m["test"]["sharpe"] for m in data[args.cell_a][fold].values()]
        b = [m["test"]["sharpe"] for m in data[args.cell_b][fold].values()]
        am, asd, an, ase = stats(a)
        bm, bsd, bn, bse = stats(b)
        diff = bm - am
        se = (ase ** 2 + bse ** 2) ** 0.5
        t = diff / se if se else float("nan")
        verdict = "B better" if t > args.t_threshold else ("A better" if t < -args.t_threshold else "noise")
        label = f"{fold.split('_')[0]} ({FOLD_TEST_YEAR.get(fold, '?')})"
        print(f"{label:<18}{f'{am:+.3f}({asd:.3f},{an})':>18}{f'{bm:+.3f}({bsd:.3f},{bn})':>18}"
              f"{diff:>+8.3f}{se:>7.3f}{t:>7.2f}{verdict:>16}")

    if not bench:
        return
    print(f"\nvs benchmark (t = (cell mean - benchmark) / cell SE):")
    hdr2 = f"{'fold':<18}{'benchmark':>11}{'A gap':>9}{'A t':>7}{'B gap':>9}{'B t':>7}"
    print(hdr2)
    print("-" * len(hdr2))
    for fold in sorted(set(data[args.cell_a]) & set(data[args.cell_b])):
        bmk = bench.get(fold)
        if bmk is None:
            continue
        am, asd, an, ase = stats([m["test"]["sharpe"] for m in data[args.cell_a][fold].values()])
        bm, bsd, bn, bse = stats([m["test"]["sharpe"] for m in data[args.cell_b][fold].values()])
        label = f"{fold.split('_')[0]} ({FOLD_TEST_YEAR.get(fold, '?')})"
        print(f"{label:<18}{bmk:>+11.3f}{am - bmk:>+9.3f}{(am - bmk) / ase:>7.2f}"
              f"{bm - bmk:>+9.3f}{(bm - bmk) / bse:>7.2f}")


if __name__ == "__main__":
    main()
