"""Download and cache raw ETF price data and macro series.

Both fetch functions cache to parquet under cache_dir and, by default, load
from cache instead of hitting the network on repeat calls. Pass refresh=True
to force a re-download.
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr


def fetch_etf_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str = "cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """Returns a DataFrame indexed by date with MultiIndex columns (ticker, field),
    field in {"close", "volume"}. "close" is already split/dividend adjusted.
    """
    cache_path = Path(cache_dir) / "etf_prices.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    frames = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No price data returned for {ticker}")
        close = df["Close"]
        volume = df["Volume"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        frames[(ticker, "close")] = close
        frames[(ticker, "volume")] = volume
        time.sleep(0.3)  # avoid hammering the endpoint across 10+ tickers

    prices = pd.concat(frames, axis=1)
    prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["ticker", "field"])
    prices = prices.sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path)
    return prices


def fetch_macro(
    series_map: dict[str, str],
    start: str,
    end: str | None = None,
    cache_dir: str = "cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """series_map maps FRED series code -> readable column name.
    Returns a daily-frequency DataFrame (lower-frequency series forward-filled).
    """
    cache_path = Path(cache_dir) / "macro.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()

    frames = {}
    for code, name in series_map.items():
        series = pdr.DataReader(code, "fred", start, end_ts)[code]
        frames[name] = series

    macro = pd.concat(frames, axis=1).sort_index()
    macro = macro.asfreq("D").ffill()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_parquet(cache_path)
    return macro
