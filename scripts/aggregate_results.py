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
import sys
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


def load_bench(path: Path):
    if not path.exists():
        return None
    return {k: v["sharpe"] for k, v in json.loads(path.read_text()).items()}


def default_etfs():
    """The default 13-asset universe, via the project's own config loader."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from configs import load_config
        return tuple(load_config("configs/default.yaml")["data"]["etfs"])
    except Exception:
        return None


def cell_universe(cell, cell_folds, runs_dirs):
    """The tradeable universe a cell was trained on, read from any of its runs'
    config.json. Returns None if no config is readable.

    The glob is anchored to the cell name: an unanchored '*<fold>_seed<n>'
    matches every cell sharing that fold and seed, so it would happily report
    some other basket's universe as this cell's.
    """
    for fold, seeds in cell_folds.items():
        for seed in seeds:
            for d in runs_dirs:
                for cfg in Path(d).glob(f"{cell}_{fold}_seed{seed}/config.json"):
                    try:
                        return tuple(json.loads(cfg.read_text())["data"]["etfs"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return None


def resolve_bench(cell, cell_folds, runs_dirs, default_bench, default_path):
    """Pick the benchmark whose universe matches this cell's.

    The default benchmark is the 13-asset one. For a cell trained on a narrowed
    basket, look for runs_rolling/equal_weight_benchmark_<suffix>.json matching
    the cell name; if there isn't one, return no benchmark rather than a wrong
    one -- a 6-asset model scored against the 13-asset bar is meaningless.
    """
    default_universe = default_etfs()
    universe = cell_universe(cell, cell_folds, runs_dirs)
    if universe is None or default_universe is None or universe == default_universe:
        return default_bench, (default_path if default_bench else "none")

    # Narrowed basket: find its own benchmark by the cell's distinguishing suffix.
    for marker in ("basket_universe", "basket_div", "basket_sharpe"):
        if marker in cell:
            path = Path(f"runs_rolling/equal_weight_benchmark_{marker}.json")
            b = load_bench(path)
            if b is not None:
                return b, str(path)
            break
    return None, f"none (universe of {len(universe)} assets has no matching benchmark)"


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

    default_bench = load_bench(Path(args.benchmark))
    if default_bench is None:
        print(f"benchmark: {args.benchmark} not found -- gaps omitted")
    else:
        print(f"default benchmark: {args.benchmark}")

    for cell in sorted(data):
        if args.cell and cell != args.cell:
            continue
        # A narrowed basket must be judged against ITS OWN equal-weight
        # benchmark -- "equal weight" means something different at 6 assets than
        # at 13, and using the wrong one manufactures crossings out of nothing
        # (see NOTES.md). Resolve per cell from the run's own config.json, and
        # omit the gap entirely rather than print a misleading one.
        bench, bench_src = resolve_bench(cell, data[cell], args.runs_dirs, default_bench, args.benchmark)
        print(f"\n{cell}    [benchmark: {bench_src}]")
        print(f"  {'fold':<18}{'test':>18}{'mean':>9}{'sd':>8}{'n':>4}{'bench':>9}{'gap':>9}")
        for fold in sorted(data[cell]):
            seeds = data[cell][fold]
            vals = [m["test"]["sharpe"] for m in seeds.values()]
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            label = f"{fold.split('_')[0]} ({FOLD_TEST_YEAR.get(fold, '?')})"
            per_seed = " ".join(f"{v:+.2f}" for v in (vals[i] for i in range(len(vals))))
            b = bench.get(fold) if bench else None
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
            # Anchor the fold name so a cell's glob cannot match a longer cell
            # name that merely starts with it -- "..._lowlr" would otherwise
            # match "..._lowlr_bootstrap" runs and drop its summary in the
            # wrong directory.
            for d in args.runs_dirs:
                if any(Path(d).glob(f"{cell}_fold*_seed*/metrics.json")):
                    out = Path(d) / f"{cell}_summary.json"
                    out.write_text(json.dumps(flat, indent=2, sort_keys=True))
                    print(f"  wrote {out} ({len(flat)} runs)")


if __name__ == "__main__":
    main()
