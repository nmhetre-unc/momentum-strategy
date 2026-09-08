"""
Turns a price series + a position signal into an equity curve and basic
stats. See analytics.py for the richer risk-adjusted metrics layer.
"""

import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 5.0,
    regimes: pd.Series = None,
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


def summary_stats(result: pd.DataFrame) -> dict:
    """Basic stats from a full-period backtest result. See analytics.full_report
    for the richer version that's also safe to use on sliced sub-periods."""
    total_return = result["equity_curve"].iloc[-1] - 1
    benchmark_return = result["benchmark_curve"].iloc[-1] - 1

    trades = (result["position"].diff().abs() > 0).sum()

    nonzero_returns = result["strategy_return"][result["strategy_return"] != 0]
    win_rate = (nonzero_returns > 0).sum() / len(nonzero_returns) if len(nonzero_returns) > 0 else 0.0

    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "num_trades": int(trades),
        "win_rate": win_rate,
        "total_cost": float(result["cost"].sum()) if "cost" in result else 0.0,
    }
