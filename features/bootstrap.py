"""Block-bootstrap synthetic price/macro data, for augmenting PPO's training
history beyond what's really in the train split.

Not parametric GBM: GBM assumes Gaussian returns and constant drift/vol,
which would smooth over exactly the fat tails and regime shifts (2020 crash,
2022 bear) this project's rolling validation is testing against. Instead,
this resamples contiguous historical blocks (with replacement) across
*all* tickers, volume, and macro series using the same block indices --
preserving real cross-asset co-movement (e.g. VIX spiking with equity
drawdowns) and short-horizon autocorrelation/vol-clustering within a block.
Trades that for artificial "seams" at block boundaries -- an accepted
tradeoff of block bootstrap, not eliminated here.

Train-only by construction (blocks are drawn from train_end and earlier),
same no-leakage discipline as fit_normalizer(). Reuses build_panel() and
compute_forward_returns() unchanged, so the synthetic panel is
schema-identical to the real one -- no separate feature logic to maintain.
"""
import numpy as np
import pandas as pd

from envs import compute_forward_returns
from features.panel import build_panel


def _log_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    close = prices.loc[:, (slice(None), "close")]
    close.columns = close.columns.get_level_values(0)
    return np.log(close[tickers] / close[tickers].shift(1))


def generate_synthetic_prices_macro(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    tickers: list[str],
    train_end: str,
    target_length: int,
    block_len: int = 21,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (synthetic_prices, synthetic_macro) with a fabricated business-day
    index, built by concatenating random contiguous blocks drawn from
    prices/macro up to train_end. Same column schema as the real inputs."""
    train_prices = prices.loc[:train_end]
    train_macro = macro.loc[:train_end]
    n = len(train_prices)
    if n <= block_len:
        raise ValueError(f"train period ({n} rows) shorter than block_len ({block_len})")

    rng = np.random.default_rng(seed)
    log_ret = _log_returns(train_prices, tickers)
    volume = train_prices.loc[:, (slice(None), "volume")]
    volume.columns = volume.columns.get_level_values(0)
    volume = volume[tickers]

    n_blocks = int(np.ceil(target_length / block_len))
    starts = rng.integers(0, n - block_len, size=n_blocks)

    ret_blocks, vol_blocks, macro_blocks = [], [], []
    for s in starts:
        ret_blocks.append(log_ret.iloc[s : s + block_len].reset_index(drop=True))
        vol_blocks.append(volume.iloc[s : s + block_len].reset_index(drop=True))
        macro_blocks.append(train_macro.iloc[s : s + block_len].reset_index(drop=True))

    synth_ret = pd.concat(ret_blocks, ignore_index=True).iloc[:target_length].fillna(0.0)
    synth_vol = pd.concat(vol_blocks, ignore_index=True).iloc[:target_length]
    synth_macro = pd.concat(macro_blocks, ignore_index=True).iloc[:target_length]

    synth_close = 100.0 * (1.0 + synth_ret).cumprod()
    synth_dates = pd.bdate_range(start="1990-01-01", periods=target_length)
    synth_close.index = synth_dates
    synth_vol.index = synth_dates
    synth_macro.index = synth_dates

    synth_prices = pd.concat(
        {t: pd.DataFrame({"close": synth_close[t], "volume": synth_vol[t]}) for t in tickers},
        axis=1,
    )
    synth_prices.columns = pd.MultiIndex.from_tuples(
        [(t, f) for t, f in synth_prices.columns], names=["ticker", "field"]
    )
    return synth_prices, synth_macro


def generate_synthetic_panel(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    tickers: list[str],
    feature_cfg: dict,
    train_end: str,
    target_length: int,
    block_len: int = 21,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (synthetic_panel, synthetic_forward_returns), built with the
    exact same feature/forward-return code as the real pipeline."""
    synth_prices, synth_macro = generate_synthetic_prices_macro(
        prices, macro, tickers, train_end, target_length, block_len, seed
    )
    synth_panel = build_panel(synth_prices, synth_macro, tickers, feature_cfg)
    synth_fwd = compute_forward_returns(synth_prices, tickers)
    return synth_panel, synth_fwd
