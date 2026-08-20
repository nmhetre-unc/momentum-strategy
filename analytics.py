"""
Turns a backtest result into risk-adjusted performance metrics -- the
layer that separates "it made money" from "here's a rigorous evaluation."
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate, based on however many days are in equity_curve."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] == 0:
        return 0.0
    years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Return per unit of total volatility (upside and downside both count)."""
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Like Sharpe, but only penalizes downside volatility, not upside swings."""
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std):
        return 0.0
    return (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    The full peak-to-trough decline at every point in time, as a negative
    fraction (e.g. -0.23 = -23% below the running peak so far). This is
    the single source of truth for the drawdown formula -- max_drawdown()
    below and plot_drawdown() in visualize.py both call this rather than
    each recomputing the same formula independently.
    """
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.23 = -23%)."""
    return drawdown_series(equity_curve).min()


def full_report(result: pd.DataFrame) -> dict:
    """
    Recomputes a normalized equity curve (always starting at 1.0) from
    `strategy_return`, rather than trusting result['equity_curve'] directly.
    This makes full_report() safe to call on an arbitrary date-sliced
    subset of a backtest result -- e.g. an out-of-sample period that
    doesn't start at the beginning of the original backtest -- which is
    exactly how walk_forward.py uses it.
    """
    strategy_return = result["strategy_return"].fillna(0)
    equity = (1 + strategy_return).cumprod()

    trades = (result["position"].diff().abs() > 0).sum()
    nonzero_returns = strategy_return[strategy_return != 0]
    win_rate = (nonzero_returns > 0).sum() / len(nonzero_returns) if len(nonzero_returns) > 0 else 0.0

    return {
        "total_return": equity.iloc[-1] - 1,
        "num_trades": int(trades),
        "win_rate": win_rate,
        "cagr": cagr(equity),
        "annualized_volatility": annualized_volatility(strategy_return),
        "sharpe_ratio": sharpe_ratio(strategy_return),
        "sortino_ratio": sortino_ratio(strategy_return),
        "max_drawdown": max_drawdown(equity),
    }
