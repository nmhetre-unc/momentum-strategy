"""
Turns a price series + a position signal into an equity curve and basic
stats. See analytics.py for the richer risk-adjusted metrics layer.
"""

import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 0.0,
    regimes: pd.Series = None,
) -> pd.DataFrame:
    """
    Signal is shifted forward by 1 day before being applied: you can only
    act on a signal the day AFTER it's generated, using that day's close.
    Skipping this shift is the single most common way to accidentally
    build a backtest with lookahead bias.

    `signal` may be binary (1 = long, 0 = flat), which is what every
    function in strategies.py returns, or fractional in [0, 1], which is
    what the volatility-targeting and regime-sizing wrappers in
    adaptive.py return. The arithmetic is identical either way.

    `cost_bps` charges a one-way transaction cost, in basis points, on
    every unit of position change -- so flipping 0 -> 1 costs cost_bps
    and resizing 0.4 -> 0.5 costs a tenth of it. It defaults to 0.0 so
    existing results are unchanged, but it is worth turning on before
    believing any adaptive strategy: adaptive logic buys its improved
    risk profile with extra trading, and at 5-10bps round-trip a lot of
    apparent edge quietly disappears. That disappearance is a finding,
    not a nuisance.

    `regimes` is optional; when supplied, its labels are carried along in
    the result so analytics.performance_by_regime() can split the P&L up
    afterwards without needing to re-align anything.
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
