"""
Computes risk-adjusted performance metrics from a backtest result.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _annualized_return_from_log(log_growth: float, periods_per_year: float) -> float:
    """
    exp(log_growth * periods_per_year) - 1: what a per-period log growth
    rate compounds to over a year. Computed in log space, with the
    exponent clipped to stay inside float64's exp() range, rather than
    raising a ratio directly to a power -- a short window or a single
    volatile period can make the "true" annualized figure astronomical
    (e.g. a 300x move over two days), so this saturates at a very large
    but finite number instead of overflowing to inf.
    """
    exponent = np.clip(log_growth * periods_per_year, -700.0, 700.0)
    return float(np.exp(exponent) - 1)


def cagr(equity_curve: pd.Series) -> float:
    """
    Compound Annual Growth Rate, based on however many days are in
    equity_curve. A non-positive or non-finite start/end leaves the
    growth ratio undefined (returns 0.0); a total loss (end == 0) is a
    real -100% annualized return, reported as exactly -1.0.
    """
    if len(equity_curve) < 2:
        return 0.0
    start, end = float(equity_curve.iloc[0]), float(equity_curve.iloc[-1])
    if not (np.isfinite(start) and np.isfinite(end)):
        return 0.0
    if end == 0.0 and start > 0:
        return -1.0
    if start <= 0 or end <= 0:
        return 0.0
    periods_per_year = TRADING_DAYS_PER_YEAR / len(equity_curve)
    return _annualized_return_from_log(np.log(end) - np.log(start), periods_per_year)


def annualized_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series | np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Return per unit of total volatility (upside and downside both count).
    Accepts a plain ndarray too -- stats.py's bootstrap resamplers call
    this on raw numpy arrays rather than reconstructing a Series per draw.
    """
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std()
    # A repeated constant should have zero std, but mean-then-subtract
    # floating-point arithmetic leaves ~1e-17 rather than exactly 0 --
    # `== 0` misses that and lets the ratio blow up. Daily return values
    # live at the 1e-3 to 1e-1 scale, so 1e-12 is generous headroom above
    # that noise floor without masking any real (if tiny) volatility.
    if pd.isna(std) or np.isclose(std, 0.0, atol=1e-12):
        return 0.0
    return (excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Like Sharpe, but only penalizes downside volatility, not upside swings."""
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    downside_std = downside.std()
    if pd.isna(downside_std) or np.isclose(downside_std, 0.0, atol=1e-12):
        return 0.0
    return (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Peak-to-trough decline at every point in time, as a negative fraction
    (e.g. -0.23 = -23% below the running peak). Canonical source for the
    drawdown formula; max_drawdown() and visualize.plot_drawdown() both
    call this rather than recomputing it.
    """
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction (e.g. -0.23 = -23%)."""
    return drawdown_series(equity_curve).min()


def full_report(result: pd.DataFrame, n_boot: int = 0) -> dict:
    """
    Recomputes a normalized equity curve (starting at 1.0) from
    `strategy_return` rather than trusting result['equity_curve']
    directly, so this is safe to call on an arbitrary date-sliced subset
    of a backtest result -- e.g. an out-of-sample period -- which is how
    walk_forward.py uses it. `n_boot=0` (default) skips the bootstrap
    entirely and reports `sharpe_ci_low`/`sharpe_ci_high` as None, since
    a stationary bootstrap over thousands of replicates dominates this
    function's cost; pass `n_boot > 0` to compute them.
    """
    strategy_return = result["strategy_return"].fillna(0)
    equity = (1 + strategy_return).cumprod()

    trades = (result["position"].diff().abs() > 0).sum()
    nonzero_returns = strategy_return[strategy_return != 0]
    win_rate = (
        (nonzero_returns > 0).sum() / len(nonzero_returns) if len(nonzero_returns) > 0 else 0.0
    )

    if n_boot > 0:
        # Lazy import: qbt.stats imports sharpe_ratio from this module,
        # so a top-level import here would be circular.
        from qbt.stats import stationary_bootstrap_sharpe
        _, sharpe_ci_low, sharpe_ci_high = stationary_bootstrap_sharpe(
            strategy_return, n_boot=n_boot
        )
    else:
        sharpe_ci_low = sharpe_ci_high = None

    return {
        "total_return": equity.iloc[-1] - 1,
        "num_trades": int(trades),
        "win_rate": win_rate,
        # exposure/turnover matter for adaptive strategies with fractional
        # positions, where num_trades alone can't distinguish 100% invested
        # from 10% invested.
        "exposure": exposure(result["position"]),
        "turnover": turnover(result["position"]),
        "cagr": cagr(equity),
        "annualized_volatility": annualized_volatility(strategy_return),
        "sharpe_ratio": sharpe_ratio(strategy_return),
        "sharpe_ci_low": sharpe_ci_low,
        "sharpe_ci_high": sharpe_ci_high,
        "sortino_ratio": sortino_ratio(strategy_return),
        "max_drawdown": max_drawdown(equity),
    }


def exposure(position: pd.Series) -> float:
    """Fraction of the period spent holding a nonzero position. Distinguishes
    a Sharpe earned at 8% exposure (mostly cash, few observations) from the
    same Sharpe at 95% exposure."""
    if len(position) == 0:
        return 0.0
    return float(position.abs().mean())


def turnover(position: pd.Series) -> float:
    """
    Total position change per year, in units of a full position turned
    over. Captures trading cost for continuously-resizing adaptive
    strategies, where num_trades (which counts discrete flips) understates
    activity.
    """
    if len(position) < 2:
        return 0.0
    total = position.diff().abs().sum()
    return float(total / (len(position) / TRADING_DAYS_PER_YEAR))


def _daily_mean_to_ann_return(mean_daily_return: float) -> float:
    """
    (1 + mean_daily_return) ** TRADING_DAYS_PER_YEAR - 1, computed the
    same safe way as cagr(): a short or volatile regime segment can leave
    a single extreme day undiluted in the mean, and TRADING_DAYS_PER_YEAR
    is already a large fixed exponent, so this is vulnerable to the same
    float64 overflow cagr() had.
    """
    base = 1 + mean_daily_return
    if not np.isfinite(base):
        return 0.0
    if base == 0.0:
        return -1.0
    if base < 0:
        return 0.0
    return _annualized_return_from_log(np.log(base), TRADING_DAYS_PER_YEAR)


def performance_by_regime(
    result: pd.DataFrame, labels: pd.Series, names: dict | None = None
) -> pd.DataFrame:
    """
    Splits a backtest result by market regime and reports the full metric
    set inside each one. A day's return is attributed to the regime in
    force on that day; the position itself was decided the day before
    (backtest.py shifts signals forward), so no future information enters
    the attribution. Regimes with few days produce Sharpe ratios with wide
    error bars -- check the `days` column before trusting any row.
    """
    labels = labels.reindex(result.index).fillna(-1).astype(int)
    strategy_return = result["strategy_return"].fillna(0)
    rows = []

    for regime_id in sorted(labels.unique()):
        if regime_id == -1:
            continue
        mask = labels == regime_id
        segment_returns = strategy_return[mask]
        if segment_returns.empty:
            continue

        # Chains in-regime days into one equity curve; gaps (other regimes)
        # are skipped, not filled, since this measures performance while
        # in the regime, not a standalone tradeable strategy.
        equity = (1 + segment_returns).cumprod()
        nonzero = segment_returns[segment_returns != 0]

        rows.append({
            "regime": regime_id,
            "name": (names or {}).get(regime_id, str(regime_id)),
            "days": int(mask.sum()),
            "total_return": equity.iloc[-1] - 1,
            "ann_return": _daily_mean_to_ann_return(segment_returns.mean()),
            "annualized_volatility": annualized_volatility(segment_returns),
            "sharpe_ratio": sharpe_ratio(segment_returns),
            "sortino_ratio": sortino_ratio(segment_returns),
            "max_drawdown": max_drawdown(equity),
            "win_rate": (nonzero > 0).sum() / len(nonzero) if len(nonzero) else 0.0,
            "exposure": exposure(result.loc[mask, "position"]),
        })

    return pd.DataFrame(rows)


def benchmark_by_regime(
    result: pd.DataFrame, labels: pd.Series, names: dict | None = None
) -> pd.DataFrame:
    """
    Applies the same regime split to buy-and-hold, so a strategy's
    per-regime numbers can be read against simply holding the asset in
    that regime.
    """
    benchmark = result.copy()
    benchmark["strategy_return"] = result["daily_return"]
    benchmark["position"] = 1.0
    return performance_by_regime(benchmark, labels, names)
