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
    feats[f"rsi_{rsi_window}"] = 100 - (100 / (1 + rs))

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
) -> pd.DataFrame:
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

    full = panel.join(macro_feats, how="left")
    full = full.dropna()
    return full


def etf_feature_columns(panel: pd.DataFrame, ticker: str) -> list[str]:
    prefix = f"{ticker}__"
    return [c for c in panel.columns if c.startswith(prefix)]


def macro_feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.startswith("macro__")]
