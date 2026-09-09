"""
Sanity-checks strategies.py, backtest.py, analytics.py, and walk_forward.py
against synthetic price data, since this sandbox can't reach the Yahoo
Finance API. Run main.py directly on your own machine for real data.
"""
import time

import numpy as np
import pandas as pd
import pytest

from qbt.analytics import full_report
from qbt.backtest import run_backtest
from qbt.strategies import STRATEGIES
from qbt.walk_forward import evaluate_out_of_sample


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(42)
    n = 600
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    returns = rng.normal(0.0004, 0.012, n)
    prices = 100 * (1 + pd.Series(returns)).cumprod()
    return pd.DataFrame({"Close": prices.values}, index=dates)


@pytest.fixture(scope="module")
def big_df():
    rng = np.random.default_rng(43)
    n = 4000
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    returns = rng.normal(0.0004, 0.012, n)
    prices = 100 * (1 + pd.Series(returns)).cumprod()
    return pd.DataFrame({"Close": prices.values}, index=dates)


@pytest.mark.parametrize("name", STRATEGIES)
def test_strategy_backtest_stats_in_range(df, name):
    fn = STRATEGIES[name]
    signal = fn(df)
    assert signal.isin([0, 1]).all(), f"{name} produced a non-binary signal"
    assert not signal.isna().any(), f"{name} produced a NaN in the signal"

    result = run_backtest(df, signal)
    stats = full_report(result)

    assert -1.0 <= stats["max_drawdown"] <= 0.0, (
        f"{name} max_drawdown out of range: {stats['max_drawdown']}"
    )
    assert 0.0 <= stats["win_rate"] <= 1.0, f"{name} win_rate out of range: {stats['win_rate']}"


@pytest.mark.parametrize("name", STRATEGIES)
def test_walk_forward_runs_for_every_strategy(df, name):
    fn = STRATEGIES[name]
    wf = evaluate_out_of_sample(df, fn, split_frac=0.7)
    in_s, out_s = wf["in_sample"], wf["out_sample"]
    assert np.isfinite(in_s["sharpe_ratio"]), f"{name}: non-finite in-sample sharpe"
    assert np.isfinite(out_s["sharpe_ratio"]), f"{name}: non-finite out-of-sample sharpe"


def test_rsi_handles_zero_average_loss_edge_case():
    # force a monotonically increasing price series so avg_loss == 0 for a
    # whole stretch, and confirm the strategy doesn't crash or produce NaN.
    straight_up = pd.DataFrame(
        {"Close": np.linspace(100, 200, 60)},
        index=pd.date_range("2023-01-01", periods=60, freq="B"),
    )
    rsi_signal = STRATEGIES["mean_reversion"](straight_up)
    assert not rsi_signal.isna().any(), "RSI signal produced NaN on the zero-avg-loss edge case"


def test_full_report_bootstrap_stays_opt_in(big_df):
    big_result = run_backtest(big_df, STRATEGIES["sma_crossover"](big_df))

    t0 = time.perf_counter()
    big_stats = full_report(big_result)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert big_stats["sharpe_ci_low"] is None, (
        "sharpe_ci_low should be None when n_boot=0 (the default)"
    )
    assert big_stats["sharpe_ci_high"] is None, (
        "sharpe_ci_high should be None when n_boot=0 (the default)"
    )
    assert elapsed_ms < 50, (
        f"full_report() with default args took {elapsed_ms:.1f}ms on 4000 rows, expected <50ms"
    )
