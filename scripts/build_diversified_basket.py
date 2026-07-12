"""Select a basket that's actually diversified, not just multiple tickers
from different-sounding categories. "XLK, XLI, XLF, SMH" all sound distinct
but could be 90%+ correlated with each other as US equity beta -- Sharpe
alone (scripts/screen_etf_universe.py) can't tell you that, only the
correlation matrix can.

Greedy max-diversification selection: sort candidates by pre-2020
(leakage-free) Sharpe descending, then add each one only if its pairwise
correlation with every already-selected asset is below a threshold --
among assets that pass, still prefer higher Sharpe. Falls back to picking
the least-correlated remaining candidate once no more pass the strict
threshold, so the target basket size is always reached.

Usage:
    python scripts/build_diversified_basket.py --size 9 --corr-threshold 0.6
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("cache_universe_screen")
PRE2020_END = "2019-12-31"


def load_returns() -> pd.DataFrame:
    series = {}
    for p in CACHE_DIR.glob("*.parquet"):
        ticker = p.stem
        close = pd.read_parquet(p)["close"]
        series[ticker] = close.loc[:PRE2020_END].pct_change()
    return pd.DataFrame(series).dropna(how="all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--corr-threshold", type=float, default=0.6,
                         help="max allowed pairwise correlation with any already-selected asset")
    args = parser.parse_args()

    returns = load_returns()
    corr = returns.corr()

    screen = json.loads(Path("runs_rolling/etf_universe_screen.json").read_text())
    sharpe = {r["ticker"]: r["pre2020_sharpe"] for r in screen}
    candidates = sorted([t for t in sharpe if t in corr.columns], key=lambda t: -sharpe[t])

    selected = []
    skipped_for_correlation = []
    for t in candidates:
        if not selected:
            selected.append(t)
            continue
        max_corr = max(abs(corr.loc[t, s]) for s in selected)
        if max_corr < args.corr_threshold:
            selected.append(t)
        else:
            skipped_for_correlation.append((t, max_corr, corr.loc[t, selected].abs().idxmax()))
        if len(selected) >= args.size:
            break

    # if threshold was too strict to reach target size, backfill with the
    # least-correlated remaining candidates regardless of threshold
    if len(selected) < args.size:
        remaining = [t for t in candidates if t not in selected]
        remaining.sort(key=lambda t: max(abs(corr.loc[t, s]) for s in selected))
        for t in remaining:
            selected.append(t)
            if len(selected) >= args.size:
                break

    print(f"=== Selected basket (target size {args.size}, corr threshold {args.corr_threshold}) ===")
    for t in selected:
        print(f"  {t:<8} pre2020_sharpe={sharpe[t]:+.3f}")

    print(f"\n=== Rejected for correlation (too similar to an already-picked asset) ===")
    for t, c, closest in skipped_for_correlation[:20]:
        print(f"  {t:<8} corr={c:.3f} with {closest} (sharpe was {sharpe[t]:+.3f})")

    print(f"\n=== Pairwise correlation matrix of the selected basket ===")
    sub = corr.loc[selected, selected]
    pd.set_option("display.width", 120)
    print(sub.round(2).to_string())

    print(f"\n=== Max off-diagonal correlation per asset (lower = more genuinely diversifying) ===")
    for t in selected:
        others = [s for s in selected if s != t]
        max_c = sub.loc[t, others].abs().max()
        print(f"  {t:<8} max|corr|={max_c:.3f}")

    out = {"selected": selected, "correlation_matrix": sub.round(3).to_dict(), "sharpe": {t: sharpe[t] for t in selected}}
    out_path = Path("runs_rolling/diversified_basket_selection.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
