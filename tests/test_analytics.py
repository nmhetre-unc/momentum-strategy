"""
Unit tests for individual metric functions in analytics.py, ml.py, and
features.py -- edge cases (zero variance, an unknown regime, a maximally
overbought RSI window) that a full backtest wouldn't reliably exercise.
"""

import numpy as np
import pandas as pd
import pytest

from qbt.analytics import max_drawdown, sharpe_ratio, sortino_ratio, turnover
from qbt.features import build_features
from qbt.ml import _regime_dummies


def test_max_drawdown_on_monotonic_increase_is_zero():
    equity = pd.Series(np.linspace(1.0, 2.0, 50))
    assert max_drawdown(equity) == 0.0


def test_sharpe_and_sortino_on_constant_returns_are_zero():
    # A repeated NONZERO constant: excess.std() computes to ~1e-17 rather
    # than exactly 0.0 (mean-then-subtract floating-point noise), so this
    # exercises the guard's tolerance, not just its exact-zero case -- a
    # `std() == 0` check passes on zeros but is skipped here, letting the
    # ratio blow up to the order of 1e16.
    returns = pd.Series(np.full(100, 0.0007))
    assert sharpe_ratio(returns) == 0.0
    assert sortino_ratio(returns) == 0.0


def test_rsi_at_avg_loss_zero_is_100_not_nan():
    # A strictly increasing price series has avg_loss == 0 for the whole
    # trailing window once warm-up (14 rows) completes -- maximally
    # overbought, which the RSI formula's 100 - 100/(1+gain/loss) would
    # divide by zero on unless avg_loss == 0 is special-cased to 100.
    straight_up = pd.DataFrame(
        {"Close": np.linspace(100, 200, 60)},
        index=pd.date_range("2023-01-01", periods=60, freq="B"),
    )
    rsi = build_features(straight_up)["rsi_14"]
    assert not rsi.iloc[14:].isna().any()
    assert (rsi.iloc[14:] == 100).all()


def test_regime_dummies_maps_unknown_to_all_zero_row():
    regimes = pd.Series([-1, 0, 1, -1, 2])
    dummies = _regime_dummies(regimes, regimes.index)
    unknown_rows = dummies.loc[regimes == -1]
    assert (unknown_rows == 0).all().all()


def test_turnover_on_constant_position_is_zero():
    position = pd.Series(np.full(300, 0.5))
    assert turnover(position) == 0.0


def test_turnover_on_one_round_trip_per_year_is_about_two():
    # One buy and one sell within a single trading year sums to 2 units of
    # position change (up by 1, back down by 1); turnover is that total
    # normalized to a year, so it lands at ~2.0 regardless of how long the
    # position was actually held within the year.
    n = 252
    position = pd.Series(0.0, index=range(n))
    position.iloc[50:150] = 1.0
    assert turnover(position) == pytest.approx(2.0)
