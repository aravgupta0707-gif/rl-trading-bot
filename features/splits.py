"""Time-based walk-forward splitting and train-only normalization.

Normalizer stats are fit on the train split only and applied to val/test --
fitting on the full panel (including future data) would leak test-period
distribution information into the features the agent trains on.
"""
import pandas as pd


def walk_forward_split(
    panel: pd.DataFrame, train_end: str, val_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = panel.loc[:train_end]
    val = panel.loc[train_end:val_end].iloc[1:]
    test = panel.loc[val_end:].iloc[1:]
    return train, val, test


def rolling_walk_forward_split(
    panel: pd.DataFrame, train_end: str, val_end: str, test_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Like walk_forward_split, but with an explicit test_end so multiple
    folds can each get a bounded (not open-ended) test window -- needed to
    compare Sharpe/drawdown across folds of comparable length."""
    train = panel.loc[:train_end]
    val = panel.loc[train_end:val_end].iloc[1:]
    test = panel.loc[val_end:test_end].iloc[1:]
    return train, val, test


def fit_normalizer(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = train.mean()
    std = train.std().replace(0, 1.0)
    return mean, std


def apply_normalizer(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    return (df - mean) / std
