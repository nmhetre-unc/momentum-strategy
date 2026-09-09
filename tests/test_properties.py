"""
Property-based tests for run_backtest() / full_report(): invariants that
should hold for ANY signal and price path, not just the specific fixtures
the other test files hand-pick.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from qbt.analytics import full_report
from qbt.backtest import run_backtest

# Price and position bounds are chosen so that, even in Hypothesis's most
# adversarial case -- a position pinned at 1.0 riding an oscillating
# max/min price every single day -- the compounded equity curve
# ((max/min)**(n-1), worst case ~1e156 here) stays inside float64 range
# instead of overflowing to inf. That is a property of the test's inputs,
# not of run_backtest(), so it doesn't weaken what's being checked.
_PRICES = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
_POSITIONS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def signal_and_price_path(draw):
    n = draw(st.integers(min_value=2, max_value=40))
    prices = draw(st.lists(_PRICES, min_size=n, max_size=n))
    positions = draw(st.lists(_POSITIONS, min_size=n, max_size=n))

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Close": prices}, index=dates)
    signal = pd.Series(positions, index=dates)
    return df, signal


@settings(max_examples=200, deadline=None)
@given(signal_and_price_path())
def test_backtest_invariants_hold_for_any_signal_and_price_path(data):
    df, signal = data
    # cost_bps=0: a position can never be more than fully long (signal is
    # in [0, 1]) and a day's return can never be below -100% (price stays
    # positive), so gross equity is provably bounded away from zero here.
    # A nonzero cost charged on top of an already near-total daily loss
    # could tip 1 + strategy_return negative -- a real property of the
    # cost formula, but not the one this test is checking.
    result = run_backtest(df, signal, cost_bps=0)
    stats = full_report(result)

    equity = result["equity_curve"].dropna().to_numpy()
    assert np.isfinite(equity).all(), "equity curve produced a NaN or inf"
    assert (equity > 0).all(), "equity curve touched zero or went negative"

    assert -1.0 <= stats["max_drawdown"] <= 0.0, f"max_drawdown out of range: {stats['max_drawdown']}"
    assert 0.0 <= stats["win_rate"] <= 1.0, f"win_rate out of range: {stats['win_rate']}"
