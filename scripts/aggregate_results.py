"""Rebuild per-cell summaries and a comparison table from per-run metrics.json.

Why this exists: each runner writes <cell>_rolling_summary.json covering only
the folds/seeds that invocation was asked for, so a --folds-filtered run
overwrites a fuller summary, and parallel runs (e.g. a Slurm job array, where
every task runs one fold/seed) race on the same file. The per-run
<run_dir>/metrics.json files are the single source of truth -- each is written
once, on completion -- so summaries are better derived than trusted.

Usage:
    python scripts/aggregate_results.py                      # table for every cell found
    python scripts/aggregate_results.py --cell attention_frozen_excess_lowlr
    python scripts/aggregate_results.py --write-summaries    # regenerate the *_summary.json files
    python scripts/aggregate_results.py --benchmark runs_rolling/equal_weight_benchmark_round8.json
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

RUNS_DIRS = ["runs_rolling", "runs_bootstrap"]
FOLD_TEST_YEAR = {  # a fold's name carries its VALIDATION year; this is the real test year
    "fold1_test2020": "2021",
    "fold2_test2021": "2022 bear",
    "fold3_test2022": "2023",
    "fold4_test2023": "2024",
    "fold5_test2024": "2025+",
}


def collect(runs_dirs):
    """-> {cell: {fold: {seed: metrics}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for d in runs_dirs:
        for metrics_path in sorted(Path(d).glob("*/metrics.json")):
            run = metrics_path.parent.name
            try:
                m = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                print(f"  ! skipping unreadable {metrics_path}")
                continue
            if "_seed" not in run:
                continue
            stem, seed = run.rsplit("_seed", 1)
            fold = m.get("fold")
            if fold is None:
                continue
            cell = stem.replace(f"_{fold}", "")
            out[cell][fold][seed] = m
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dirs", nargs="+", default=RUNS_DIRS)
    parser.add_argument("--cell", default=None, help="only this cell")
    parser.add_argument("--benchmark", default="runs_rolling/equal_weight_benchmark.json")
    parser.add_argument("--write-summaries", action="store_true",
                        help="also (re)write <cell>_summary.json next to the runs")
    args = parser.parse_args()

    data = collect(args.runs_dirs)
    if not data:
        raise SystemExit("no metrics.json found under " + ", ".join(args.runs_dirs))

    bench = {}
    bench_path = Path(args.benchmark)
    if bench_path.exists():
        bench = {k: v["sharpe"] for k, v in json.loads(bench_path.read_text()).items()}
        print(f"benchmark: {bench_path}")
    else:
        print(f"benchmark: {bench_path} not found -- gaps omitted")

    for cell in sorted(data):
        if args.cell and cell != args.cell:
            continue
        print(f"\n{cell}")
        print(f"  {'fold':<18}{'test':>18}{'mean':>9}{'sd':>8}{'n':>4}{'bench':>9}{'gap':>9}")
        for fold in sorted(data[cell]):
            seeds = data[cell][fold]
            vals = [m["test"]["sharpe"] for m in seeds.values()]
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            label = f"{fold.split('_')[0]} ({FOLD_TEST_YEAR.get(fold, '?')})"
            per_seed = " ".join(f"{v:+.2f}" for v in (vals[i] for i in range(len(vals))))
            b = bench.get(fold)
            bcol = f"{b:+.3f}" if b is not None else "     -"
            gcol = f"{mean - b:+.3f}" if b is not None else "     -"
            flag = ""
            if b is not None:
                flag = "  <= BEATS BENCHMARK" if mean > b else ""
            print(f"  {label:<18}{per_seed:>18}{mean:>9.3f}{sd:>8.3f}{len(vals):>4}"
                  f"{bcol:>9}{gcol:>9}{flag}")

        if args.write_summaries:
            flat = {}
            for fold, seeds in data[cell].items():
                for seed, m in seeds.items():
                    flat[f"{cell}_{fold}_seed{seed}"] = m
            for d in args.runs_dirs:
                if any(Path(d).glob(f"{cell}_*seed*/metrics.json")):
                    out = Path(d) / f"{cell}_summary.json"
                    out.write_text(json.dumps(flat, indent=2, sort_keys=True))
                    print(f"  wrote {out} ({len(flat)} runs)")


if __name__ == "__main__":
    main()
