"""
Turns a price series + a position signal into an equity curve and basic
stats. See analytics.py for the richer risk-adjusted metrics layer.
"""

import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 5.0,
    regimes: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Shifts `signal` forward by one day before applying it, so a position
    can only act on the close after it was generated; skipping this
    shift is the most common way to introduce lookahead bias. `signal`
    may be binary (1/0) or fractional in [0, 1] -- the arithmetic is the
    same either way. `cost_bps` charges a one-way cost per unit of
    position change (default 5.0; pass 0.0 for gross returns), and
    `regimes`, if supplied, is carried into the result for later
    per-regime attribution.
    """
    daily_return = df["Close"].pct_change()
    position = signal.reindex(df.index).shift(1).fillna(0)

    # Cost is charged on the day the position actually changes, which is
    # the day the trade would have been executed.
    cost = position.diff().abs().fillna(position.abs()) * (cost_bps / 10_000.0)

    gross_return = position * daily_return
    strategy_return = gross_return - cost
    equity_curve = (1 + strategy_return).cumprod()
    benchmark_curve = (1 + daily_return).cumprod()

    result = pd.DataFrame({
        "close": df["Close"],
        "signal": signal.reindex(df.index),
        "position": position,
        "daily_return": daily_return,
        "gross_return": gross_return,
        "cost": cost,
        "strategy_return": strategy_return,
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
    })

    if regimes is not None:
        result["regime"] = regimes.reindex(df.index).fillna(-1).astype(int)

    return result


def run_portfolio_backtest(
    weights: pd.DataFrame, prices: pd.DataFrame, cost_bps: float = 5.0
) -> pd.DataFrame:
    """
    Portfolio-level analog of run_backtest(): takes a target-weight matrix
    (dates x tickers, e.g. from cross_sectional_momentum()) instead of a
    single-asset signal, and a matching price matrix instead of a single
    `Close` column. Applies the exact same shift(1)-then-apply convention
    -- a weight decided on day t can only act on day t+1's return -- and
    the exact same day-0 entry-cost convention (a NaN diff on the first
    row falls back to that row's own position size, as if entering fresh
    from flat), just summed across tickers: `cost_bps` is charged on the
    sum of absolute weight changes across every ticker at each rebalance,
    not per ticker independently.

    Returns the same output columns as run_backtest() wherever a column
    reduces to one scalar per date at the portfolio level (gross_return,
    cost, strategy_return, equity_curve); `close`, `signal`, `position`
    and `daily_return` don't -- they're inherently per-ticker matrices
    here, not single columns, so they're left out rather than forced into
    a shape they don't fit. `benchmark_curve` is the equal-weight
    buy-and-hold of every ticker in `prices`, the portfolio analog of
    run_backtest()'s single-asset buy-and-hold.
    """
    position = weights.reindex(index=prices.index, columns=prices.columns).shift(1).fillna(0.0)
    daily_return = prices.pct_change()

    cost = (
        position.diff().abs().fillna(position.abs()).sum(axis=1) * (cost_bps / 10_000.0)
    )

    # A ticker's own data gap (held, but its price is momentarily NaN)
    # contributes 0 that day rather than NaN-ing the whole portfolio's
    # return -- a data gap in one name shouldn't erase every other
    # holding's return on the same day.
    gross_return = (position * daily_return.fillna(0.0)).sum(axis=1)
    strategy_return = gross_return - cost
    equity_curve = (1 + strategy_return).cumprod()
    benchmark_curve = (1 + daily_return.mean(axis=1)).cumprod()

    return pd.DataFrame({
        "gross_return": gross_return,
        "cost": cost,
        "strategy_return": strategy_return,
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
    })


def summary_stats(result: pd.DataFrame) -> dict:
    """Basic stats from a full-period backtest result. See analytics.full_report
    for the richer version that's also safe to use on sliced sub-periods."""
    total_return = result["equity_curve"].iloc[-1] - 1
    benchmark_return = result["benchmark_curve"].iloc[-1] - 1

    trades = (result["position"].diff().abs() > 0).sum()

    nonzero_returns = result["strategy_return"][result["strategy_return"] != 0]
    win_rate = (
        (nonzero_returns > 0).sum() / len(nonzero_returns) if len(nonzero_returns) > 0 else 0.0
    )

    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "num_trades": int(trades),
        "win_rate": win_rate,
        "total_cost": float(result["cost"].sum()) if "cost" in result else 0.0,
    }
