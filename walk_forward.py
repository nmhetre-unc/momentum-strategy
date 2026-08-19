"""
Checks whether a strategy's performance holds up on data it wasn't
"tuned" on -- the single biggest thing separating a rigorous backtest
from one that's secretly overfit to one lucky period.
"""

import pandas as pd

from backtest import run_backtest
from analytics import full_report


def evaluate_out_of_sample(df: pd.DataFrame, strategy_fn, split_frac: float = 0.7, **strategy_params) -> dict:
    """
    Computes the signal and backtest ONCE on the full dataset (so rolling
    windows have full history available from the start), then splits the
    result chronologically: the first `split_frac` of days is "in-sample,"
    the rest is "out-of-sample." Reports full metrics for each separately.

    A strategy whose Sharpe ratio collapses (or flips negative) out-of-sample
    is a strategy that was fit to noise in the in-sample period, not one
    that captures something real.
    """
    signal = strategy_fn(df, **strategy_params)
    result = run_backtest(df, signal)

    split_idx = int(len(result) * split_frac)
    split_date = result.index[split_idx]

    in_sample = result.loc[:split_date]
    out_sample = result.loc[split_date:]

    return {
        "split_date": str(split_date.date()),
        "in_sample": full_report(in_sample),
        "out_sample": full_report(out_sample),
    }
