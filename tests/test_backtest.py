"""
Unit tests for run_backtest()'s core arithmetic: the buy-and-hold
identity, a true no-op signal, the cost formula's proportionality and
entry-day convention, and full_report()'s safety on a date-sliced result.
"""

import numpy as np
import pandas as pd
import pytest

from qbt.analytics import full_report
from qbt.backtest import run_backtest, summary_stats


def test_constant_signal_at_zero_cost_reproduces_buy_and_hold():
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    returns = rng.normal(0.0004, 0.01, n)
    close = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({"Close": close}, index=dates)

    signal = pd.Series(1, index=df.index)
    result = run_backtest(df, signal, cost_bps=0)

    # Both curves are NaN on day 0 (pct_change has nothing to compare
    # against yet), so drop that before comparing.
    assert np.allclose(result["equity_curve"].dropna(), result["benchmark_curve"].dropna())


def test_all_zero_signal_is_a_true_no_op():
    rng = np.random.default_rng(1)
    n = 100
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    returns = rng.normal(0.0004, 0.01, n)
    close = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({"Close": close}, index=dates)

    signal = pd.Series(0, index=df.index)
    result = run_backtest(df, signal, cost_bps=5)
    stats = summary_stats(result)

    assert stats["total_return"] == 0.0
    assert stats["total_cost"] == 0.0
    assert stats["num_trades"] == 0


def test_cost_proportional_to_position_change():
    # signal -> position is shifted by one day, so this signal produces a
    # clean 0 -> 1 flip (magnitude 1.0) followed later by an isolated
    # 0.4 -> 0.5 resize (magnitude 0.1), with an unrelated 1 -> 0.4 step
    # in between that we don't use. Comparing the TOTAL cost of two
    # separate backtests would not give a clean 10x ratio here -- the
    # smaller position's entry-day charge (see below) is a bigger share of
    # its own total, pulling the ratio to 0.5 instead of 0.1 -- so this
    # compares the single transition's cost directly instead.
    dates = pd.date_range("2021-01-01", periods=7, freq="B")
    close = pd.Series(100.0, index=dates)
    df = pd.DataFrame({"Close": close})
    signal = pd.Series([0, 0, 1, 1, 0.4, 0.5, 0.5], index=dates, dtype=float)

    cost_bps = 7.0
    result = run_backtest(df, signal, cost_bps=cost_bps)
    position = result["position"]

    flip_idx, resize_idx = 3, 6
    assert position.iloc[flip_idx] - position.iloc[flip_idx - 1] == pytest.approx(1.0)
    assert position.iloc[resize_idx] - position.iloc[resize_idx - 1] == pytest.approx(0.1)

    flip_cost = result["cost"].iloc[flip_idx]
    resize_cost = result["cost"].iloc[resize_idx]
    assert resize_cost == pytest.approx(flip_cost * 0.1)

    # Separately: cost on day 0 of any backtest is charged on that day's
    # own position magnitude, not a diff against an assumed-flat prior day
    # (there is no prior day) -- pin that convention on its own terms.
    assert result["cost"].iloc[0] == abs(position.iloc[0]) * cost_bps / 1e4


def test_full_report_on_slice_matches_independent_rerun():
    rng = np.random.default_rng(2)
    n = 300
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    returns = rng.normal(0.0003, 0.011, n)
    close = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({"Close": close}, index=dates)

    signal = pd.Series(rng.integers(0, 2, n), index=dates, dtype=float)
    boundary = 150
    # Force the two days before the slice flat, so both the position AND
    # the cost of entering it at the slice boundary come out the same
    # whether that day is read out of a continuous backtest or is day 0 of
    # a fresh one -- full_report() only promises a correctly renormalized
    # equity curve on an arbitrary slice, not boundary-day cost parity,
    # unless the position going into the slice was already flat.
    signal.iloc[boundary - 2] = 0
    signal.iloc[boundary - 1] = 0

    full_result = run_backtest(df, signal, cost_bps=5)
    sliced_stats = full_report(full_result.loc[dates[boundary]:])

    slice_df = df.loc[dates[boundary]:]
    slice_signal = signal.loc[dates[boundary]:]
    independent_stats = full_report(run_backtest(slice_df, slice_signal, cost_bps=5))

    for key in sliced_stats:
        if key in ("sharpe_ci_low", "sharpe_ci_high"):
            assert sliced_stats[key] is None and independent_stats[key] is None
            continue
        assert sliced_stats[key] == pytest.approx(independent_stats[key]), key
