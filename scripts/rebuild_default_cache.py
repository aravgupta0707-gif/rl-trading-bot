"""Rebuild the default 13-asset cache/ from the committed cache_widectx/ parquet.

cache/ is gitignored, so a fresh clone has no data and the first run tries to
download from Yahoo Finance -- which fails on any machine without outbound
access to it (the cloud sandbox this was developed in, and typically HPC
compute nodes).

cache_widectx/ IS committed and is a strict superset: all 13 default tickers
plus 6 observed-only context ETFs, and 11 macro series including the default 5.
Subsetting it reproduces the default panel exactly -- folds 1-4 come back with
the same row counts as the original runs (252/251/249/251). Fold 5 runs to the
end of available data, so its length tracks the cache vintage rather than the
fold definition (364 days here vs 358 in the earliest runs); recompute fold 5's
benchmark on the same vintage before comparing fold 5 numbers across caches.

Usage:
    python scripts/rebuild_default_cache.py            # writes cache/
    python scripts/rebuild_default_cache.py --force    # overwrite existing
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import load_config

SOURCE = Path("cache_widectx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(SOURCE), help="committed cache dir to subset")
    parser.add_argument("--out", default="cache", help="cache dir to write")
    parser.add_argument("--force", action="store_true", help="overwrite an existing cache")
    args = parser.parse_args()

    out = Path(args.out)
    if (out / "etf_prices.parquet").exists() and not args.force:
        print(f"{out}/ already populated; pass --force to overwrite")
        return

    cfg = load_config("configs/default.yaml")["data"]
    etfs, macro_names = cfg["etfs"], list(cfg["macro"].values())

    src = Path(args.source)
    prices = pd.read_parquet(src / "etf_prices.parquet")
    macro = pd.read_parquet(src / "macro.parquet")

    missing_t = [t for t in etfs if t not in set(prices.columns.get_level_values(0))]
    missing_m = [m for m in macro_names if m not in macro.columns]
    if missing_t or missing_m:
        raise SystemExit(f"{src} is not a superset of configs/default.yaml: "
                         f"missing tickers {missing_t}, missing macro {missing_m}")

    sub_prices = prices.reindex(
        columns=pd.MultiIndex.from_product([etfs, ["close", "volume"]], names=["ticker", "field"])
    )
    sub_macro = macro[macro_names]

    out.mkdir(parents=True, exist_ok=True)
    sub_prices.to_parquet(out / "etf_prices.parquet")
    sub_macro.to_parquet(out / "macro.parquet")

    print(f"wrote {out}/etf_prices.parquet  {sub_prices.shape}  "
          f"{sub_prices.index.min().date()} -> {sub_prices.index.max().date()}")
    print(f"wrote {out}/macro.parquet       {sub_macro.shape}")


if __name__ == "__main__":
    main()
