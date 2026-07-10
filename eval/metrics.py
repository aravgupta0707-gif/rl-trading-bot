import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def max_drawdown(portfolio_values: pd.Series) -> float:
    running_max = portfolio_values.cummax()
    drawdown = portfolio_values / running_max - 1.0
    return float(drawdown.min())


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    cumulative = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float(cumulative ** (1 / n_years) - 1)


def summarize(returns: pd.Series, portfolio_values: pd.Series, turnovers: pd.Series) -> dict:
    return {
        "total_return": float(portfolio_values.iloc[-1] - 1.0),
        "annualized_return": annualized_return(returns),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(portfolio_values),
        "avg_daily_turnover": float(turnovers.mean()),
        "n_days": int(len(returns)),
    }
