"""
qbt.runner: the pure function the service's workers call. Checks that it
gives exactly what the CLI path gives, that it is deterministic (the
property that makes re-running a job after a worker crash safe), that
the bootstrap implementation never changes the numbers, and that bad
input is a ValueError rather than some other crash.
"""

import os

import numpy as np
import pandas as pd
import pytest

import qbt.bootstrap as qb
from qbt.analytics import full_report
from qbt.backtest import run_backtest
from qbt.runner import metrics_equal, run, validate_prices
from qbt.strategies import STRATEGIES, default_params

# Same rule as test_bootstrap.py: CI sets QBT_REQUIRE_CPP=1 so a missing
# extension fails the "cpp" case there instead of skipping it.
_skip_cpp = pytest.mark.skipif(
    not qb.cpp_available() and os.environ.get("QBT_REQUIRE_CPP") != "1",
    reason=f"C++ extension unavailable: {qb._EXT_UNAVAILABLE}",
)
IMPLS = [pytest.param(i, marks=_skip_cpp) if i == "cpp" else i for i in qb.IMPLEMENTATIONS]


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 600
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    return pd.DataFrame({"Close": close}, index=dates)


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_matches_the_cli_path(prices, strategy):
    params = default_params(strategy)
    out = run(prices, strategy, params, cost_bps=5.0, n_boot=50, seed=3)

    signal = STRATEGIES[strategy](prices, **params)
    expected = full_report(run_backtest(prices, signal, cost_bps=5.0), n_boot=50, seed=3)
    assert metrics_equal(out.metrics, {k: float(v) for k, v in expected.items()})
    assert out.result["equity"].iloc[-1] - 1 == pytest.approx(out.metrics["total_return"])


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_two_runs_are_bit_identical(prices, strategy):
    a = run(prices, strategy, default_params(strategy), n_boot=100, seed=11)
    b = run(prices, strategy, default_params(strategy), n_boot=100, seed=11)
    assert metrics_equal(a.metrics, b.metrics)
    pd.testing.assert_frame_equal(a.result, b.result, check_exact=True)


@pytest.mark.parametrize("impl", IMPLS)
def test_bootstrap_impl_changes_nothing_but_the_label(prices, impl):
    reference = run(prices, "momentum", {}, n_boot=200, seed=5, impl="python")
    out = run(prices, "momentum", {}, n_boot=200, seed=5, impl=impl)
    assert out.bootstrap_impl == impl
    assert metrics_equal(out.metrics, reference.metrics)


def test_auto_records_the_impl_that_ran(prices):
    out = run(prices, "momentum", {}, n_boot=10)
    assert out.bootstrap_impl == ("cpp" if qb.cpp_available() else "numpy")


def test_no_bootstrap_means_no_interval_and_no_impl(prices):
    out = run(prices, "sma_crossover", {}, n_boot=0)
    assert out.bootstrap_impl is None
    assert "sharpe_ci_low" not in out.metrics and "sharpe_ci_high" not in out.metrics
    assert "sharpe_ratio" in out.metrics


def test_seed_changes_the_interval(prices):
    a = run(prices, "momentum", {}, n_boot=200, seed=1)
    b = run(prices, "momentum", {}, n_boot=200, seed=2)
    assert a.metrics["sharpe_ratio"] == b.metrics["sharpe_ratio"]
    assert a.metrics["sharpe_ci_low"] != b.metrics["sharpe_ci_low"]


def test_metrics_are_plain_floats(prices):
    out = run(prices, "mean_reversion", {}, n_boot=20)
    assert all(type(v) is float for v in out.metrics.values())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"strategy": "nope", "params": {}}, "unknown strategy"),
        ({"strategy": "regime_switch", "params": {}}, "unknown strategy"),
        ({"strategy": "momentum", "params": {"window": 5}}, "unknown parameter"),
        ({"strategy": "momentum", "params": {}, "cost_bps": -1}, "cost_bps"),
        ({"strategy": "momentum", "params": {}, "n_boot": -1}, "n_boot"),
    ],
)
def test_bad_requests_raise_value_error(prices, kwargs, message):
    with pytest.raises(ValueError, match=message):
        run(prices, **kwargs)


def _frame(close, index=None):
    index = pd.date_range("2020-01-01", periods=len(close), freq="B") if index is None else index
    return pd.DataFrame({"Close": close}, index=index)


@pytest.mark.parametrize(
    ("df", "message"),
    [
        (pd.DataFrame({"Open": [1.0, 2.0]}), "Close"),
        (_frame([100.0]), "at least 2"),
        (_frame([100.0, 101.0], index=pd.Index([0, 1])), "DatetimeIndex"),
        (_frame([100.0, 101.0], index=pd.to_datetime(["2020-01-02", "2020-01-01"])), "increasing"),
        (_frame([100.0, 101.0], index=pd.to_datetime(["2020-01-01", "2020-01-01"])), "unique"),
        (_frame([100.0, np.nan]), "missing"),
        (_frame([100.0, 0.0]), "positive"),
    ],
)
def test_bad_price_frames_are_rejected(df, message):
    with pytest.raises(ValueError, match=message):
        validate_prices(df)
