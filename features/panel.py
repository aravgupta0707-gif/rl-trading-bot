"""Cross-sectional feature panel construction.

Every row (date) carries every ETF's own features AND every other ETF's
features, plus a shared macro block. This is what lets an encoder learn
cross-asset spillover effects (e.g. "TLT's move informs how to trade XLE")
instead of each ETF being modeled in isolation.

No column here is a forward-looking label -- every feature is computable
from information available as of the close of the row's date. (Caveat: FRED
macro series are used at their as-published value with no publication-lag
adjustment, which is a simplification -- a live system would need
real-time-vintage macro data to avoid a subtle lookahead via data revisions.)
"""
import numpy as np
import pandas as pd


def _etf_features(
    close: pd.Series, volume: pd.Series, return_windows: list[int], vol_window: int, rsi_window: int
) -> pd.DataFrame:
    feats = {}
    for w in return_windows:
        feats[f"ret_{w}d"] = close.pct_change(w)

    log_ret = np.log(close / close.shift(1))
    feats[f"vol_{vol_window}d"] = log_ret.rolling(vol_window).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # standard RSI convention: zero average loss in the window -> RSI=100
    # (nothing but gains), rather than the NaN that gain/0 produces. Without
    # this, any near-zero-volatility instrument (e.g. T-bill ETFs, which can
    # go many consecutive days with no down-days at all) poisons this single
    # column with NaN often enough to wipe out most of the panel once
    # build_panel()'s global dropna() runs -- silently, since every other
    # ticker's data is untouched.
    feats[f"rsi_{rsi_window}"] = rsi.where(loss != 0, 100.0)

    vol_mean = volume.rolling(vol_window).mean()
    vol_std = volume.rolling(vol_window).std()
    feats[f"volume_z_{vol_window}d"] = (volume - vol_mean) / vol_std

    return pd.DataFrame(feats)


def _macro_features(macro: pd.DataFrame, change_window: int) -> pd.DataFrame:
    feats = {}
    for col in macro.columns:
        roll_mean = macro[col].rolling(252).mean()
        roll_std = macro[col].rolling(252).std()
        feats[f"{col}_z"] = (macro[col] - roll_mean) / roll_std
        feats[f"{col}_chg_{change_window}d"] = macro[col].diff(change_window)
    return pd.DataFrame(feats)


def build_panel(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    tickers: list[str],
    feature_cfg: dict,
    context_tickers: list[str] | None = None,
) -> pd.DataFrame:
    """context_tickers (optional): assets observed for their technical
    features but NOT part of the tradeable/action universe -- e.g. credit
    ETFs, curve ETFs, broad-market ETFs, used purely to give the encoder
    more regime context. Their columns are folded into the macro__ block
    (same prefix macro_feature_columns() filters on), so every encoder
    picks them up automatically -- for the attention encoder specifically,
    that means they join the existing macro context token rather than
    getting their own per-asset token, so the action space (envs/
    portfolio_env.py, sized off `tickers` only) is unaffected."""
    return_windows = feature_cfg["return_windows"]
    vol_window = feature_cfg["vol_window"]
    rsi_window = feature_cfg["rsi_window"]
    change_window = feature_cfg["macro_change_window"]

    per_etf = {}
    for ticker in tickers:
        close = prices[(ticker, "close")]
        volume = prices[(ticker, "volume")]
        per_etf[ticker] = _etf_features(close, volume, return_windows, vol_window, rsi_window)

    panel = pd.concat(per_etf, axis=1)
    panel.columns = [f"{ticker}__{col}" for ticker, col in panel.columns]

    macro_feats = _macro_features(macro, change_window)
    macro_feats.columns = [f"macro__{c}" for c in macro_feats.columns]

    frames = [panel, macro_feats]
    if context_tickers:
        per_ctx = {}
        for ticker in context_tickers:
            close = prices[(ticker, "close")]
            volume = prices[(ticker, "volume")]
            per_ctx[ticker] = _etf_features(close, volume, return_windows, vol_window, rsi_window)
        ctx_feats = pd.concat(per_ctx, axis=1)
        ctx_feats.columns = [f"macro__ctx_{ticker}__{col}" for ticker, col in ctx_feats.columns]
        frames.append(ctx_feats)

    full = frames[0]
    for f in frames[1:]:
        full = full.join(f, how="left")
    full = full.dropna()
    return full


def etf_feature_columns(panel: pd.DataFrame, ticker: str) -> list[str]:
    prefix = f"{ticker}__"
    return [c for c in panel.columns if c.startswith(prefix)]


def macro_feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("macro__")]
