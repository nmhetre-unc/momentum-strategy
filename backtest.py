"""
Turns a price series + a position signal into an equity curve and
summary stats. This is the piece every strategy shares, regardless of
how its signal was generated.
"""

import pandas as pd


def run_backtest(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    """
    Signal is shifted forward by 1 day before being applied: you can only
    act on a signal the day AFTER it's generated, using that day's close.
    Skipping this shift is the single most common way to accidentally
    build a backtest with lookahead bias.
    """
    daily_return = df["Close"].pct_change()
    position = signal.shift(1).fillna(0)

    strategy_return = position * daily_return
    equity_curve = (1 + strategy_return).cumprod()
    benchmark_curve = (1 + daily_return).cumprod()

    return pd.DataFrame({
        "close": df["Close"],
        "signal": signal,
        "position": position,
        "daily_return": daily_return,
        "strategy_return": strategy_return,
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
    })


def summary_stats(result: pd.DataFrame) -> dict:
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
    }
