"""
qbt.strategies.out_of_sample_from: a learning strategy's first out-of-sample
day, computed without running it, must be the day its own signal records
(qbt.timesplit.OUT_OF_SAMPLE_ATTR) -- for every strategy, so a new learning
strategy without an entry fails here instead of being reported as having
no training period.
"""

import numpy as np
import pandas as pd
import pytest

from qbt.strategies import STRATEGIES, learns_from_history, out_of_sample_from
from qbt.timesplit import out_of_sample_start, split_date


def frame(periods: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", periods=periods)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, periods))
    return pd.DataFrame({
        "Close": close,
        "High": close * (1 + rng.uniform(0, 0.01, periods)),
        "Low": close * (1 - rng.uniform(0, 0.01, periods)),
        "Open": close * (1 + rng.normal(0, 0.004, periods)),
        "Volume": rng.integers(10**6, 10**8, periods),
    }, index=pd.DatetimeIndex(days.to_numpy(), name="Date").as_unit("us"))


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_matches_what_the_signal_records(strategy):
    df = frame()
    recorded = out_of_sample_start(STRATEGIES[strategy](df))
    assert out_of_sample_from(strategy, df, {}) == recorded
    assert learns_from_history(strategy) == (recorded is not None)


@pytest.mark.parametrize("train_frac", [0.5, 0.7, 0.85])
def test_ml_direction_uses_the_requested_train_frac(train_frac):
    df = frame()
    signal = STRATEGIES["ml_direction"](df, train_frac=train_frac)
    start = out_of_sample_from("ml_direction", df, {"train_frac": train_frac})
    assert start == out_of_sample_start(signal) == split_date(df.index, train_frac)
    # Flat on every day before it: the training period holds no position.
    assert (signal[signal.index < start] == 0).all()


def test_ml_direction_default_is_the_signal_default():
    df = frame()
    assert out_of_sample_from("ml_direction", df, {}) == split_date(df.index, 0.7)


def test_unknown_strategy():
    with pytest.raises(ValueError, match="unknown strategy"):
        learns_from_history("nope")
