"""Leakage-free promise screen across a broad, curated ETF universe -- not a
literal "top 100 by volume" scrape (that's dominated by leveraged/inverse/
single-stock-daily-target products with decay mechanics unsuitable for a
buy-and-hold portfolio universe, confirmed by checking Yahoo Finance's most-
active list directly). Instead: every major liquid, non-leveraged ETF
category a real allocator would consider -- broad market, style factors,
single countries, credit/duration tiers, commodities, currencies, sub-
sectors -- restricted to funds with pre-2016 inception so they have full
history for this project's start_date.

Computes buy-and-hold Sharpe using ONLY pre-2020 data (fold 1's train
period, the one cutoff that's legitimately out-of-sample for every rolling
fold used elsewhere in this project) -- same corrected methodology as
scripts/asset_promise.py, to avoid the leakage trap of screening on data
that overlaps the folds this would later be evaluated on.

Fetches each ticker independently (not the shared fetch_etf_prices cache)
so one bad/delisted ticker doesn't kill the whole batch.

Usage:
    python scripts/screen_etf_universe.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import sharpe_ratio

CURRENT_BASKET = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "TLT", "GLD", "EFA"]

CANDIDATES = {
    "broad_market_style": ["SPY", "VOO", "IVV", "VTI", "QQQ", "DIA", "IWM", "IJH", "VTV", "VUG", "IWD", "IWF", "MTUM", "USMV"],
    "international_country": ["VEA", "VWO", "IEFA", "IEMG", "EWJ", "FXI", "EWZ", "EWG", "EWU", "INDA", "EWC", "EWA", "EWH", "EWY", "EWT"],
    "fixed_income": ["SHY", "IEF", "AGG", "BND", "LQD", "HYG", "MUB", "TIP", "BIL"],
    "commodities": ["SLV", "USO", "UNG", "DBC", "DBA"],
    "currency": ["UUP", "FXE", "FXY", "FXB"],
    "real_estate": ["VNQ"],
    "sub_sector": ["SMH", "XBI", "KRE", "XOP", "IYT"],
    "thematic": ["ARKK"],
}

START = "2016-01-01"
PRE2020_END = "2019-12-31"
CACHE_DIR = Path("cache_universe_screen")


def fetch_one(ticker: str) -> pd.Series | None:
    cache_path = CACHE_DIR / f"{ticker}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)["close"]
    try:
        df = yf.download(ticker, start=START, end=None, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  SKIP {ticker}: no data")
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if close.index.min() > pd.Timestamp("2016-06-01"):
            print(f"  SKIP {ticker}: inception too recent ({close.index.min().date()}) for pre-2020 screen")
            return None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        close.to_frame("close").to_parquet(cache_path)
        time.sleep(0.3)
        return close
    except Exception as e:
        print(f"  SKIP {ticker}: {e}")
        return None


def main():
    all_candidates = [t for cat in CANDIDATES.values() for t in cat] + CURRENT_BASKET
    print(f"Fetching {len(all_candidates)} tickers ({len(all_candidates) - len(CURRENT_BASKET)} new candidates + {len(CURRENT_BASKET)} current basket)...\n")

    results = []
    ticker_to_cat = {t: cat for cat, tickers in CANDIDATES.items() for t in tickers}
    for ticker in CURRENT_BASKET:
        ticker_to_cat[ticker] = "current_basket"

    for ticker in all_candidates:
        close = fetch_one(ticker)
        if close is None:
            continue
        pre2020 = close.loc[:PRE2020_END]
        if len(pre2020) < 252:
            print(f"  SKIP {ticker}: only {len(pre2020)} pre-2020 rows")
            continue
        ret = pre2020.pct_change().dropna()
        s = sharpe_ratio(ret)
        results.append({"ticker": ticker, "category": ticker_to_cat[ticker], "pre2020_sharpe": s, "n_days": len(ret)})

    df = pd.DataFrame(results).sort_values("pre2020_sharpe", ascending=False)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 120)
    print("\n=== Pre-2020 (leakage-free) buy-and-hold Sharpe, ranked ===\n")
    print(df.to_string(index=False))

    out_path = Path("runs_rolling") / "etf_universe_screen.json"
    out_path.parent.mkdir(exist_ok=True)
    df.to_json(out_path, orient="records", indent=2)
    print(f"\nwrote {out_path}  ({len(df)}/{len(all_candidates)} tickers screened)")


if __name__ == "__main__":
    main()
