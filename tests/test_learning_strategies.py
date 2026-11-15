"""
No strategy may be scored on data it learned from
(scripts/audit_learning_strategies.py,
results/ml/learning_strategies_audit.json).

Every regime wrapper fits a regime model on a leading slice of history
and the `auto` ones also learn a choice from the first `learn_frac`.
Before version 2 they traded, and full-period metrics scored them, on
those same days; regime_sized also sized positions from every labelled
day's volatility, future days included.

Tests that must fail on the pre-fix code (commit 7b5bb84) import only
names that existed there, and compute each wrapper's first
out-of-sample day independently of the code under test, so they fail
on an assertion, not on an import.
"""

import numpy as np
import pandas as pd
import pytest

from qbt.adaptive import (
    ADAPTIVE_STRATEGIES,
    _learn_cutoff,
    regime_filtered,
    regime_sized,
    regime_switch,
)
from qbt.regime import UNKNOWN, detect_regimes
from qbt.strategies import STRATEGIES
from qbt.walk_forward import evaluate_out_of_sample, rolling_walk_forward

N_DAYS = 1500
LEARN_FRAC = 0.6
# The wrappers that learn a choice from the first LEARN_FRAC of history,
# on top of the regime model every wrapper here fits.
LEARNS_A_CHOICE = ("regime_filtered", "regime_switch", "regime_sized", "adaptive_ensemble")
REGIME_WRAPPERS = (*LEARNS_A_CHOICE, "regime_parameters")


def _prices(n: int = N_DAYS, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2012-01-02", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n))
    volume = rng.integers(50_000_000, 150_000_000, n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=index)


def _rewrite_from(df: pd.DataFrame, position: int) -> pd.DataFrame:
    """Same history before row `position`; from it on, every daily return
    is negated and tripled, so any statistic of that tail changes."""
    out = df.copy()
    returns = out["Close"].pct_change().to_numpy()
    start = out["Close"].iloc[position - 1]
    out.iloc[position:, out.columns.get_loc("Close")] = (
        start * np.cumprod(1 - 3 * returns[position:]))
    return out


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return _prices()


@pytest.fixture(scope="module")
def first_out_of_sample_day(prices) -> dict:
    """Each wrapper's first out-of-sample day, from first principles: the
    first labelled day after the regime model's fit window, and for a
    wrapper that learns a choice, also after its learning window."""
    regime_oos = pd.Timestamp(detect_regimes(prices, fit_frac=0.7).causal_from)
    learn_end = _learn_cutoff(prices, LEARN_FRAC)
    after_learning = prices.index[prices.index > learn_end][0]
    return {name: max(regime_oos, after_learning) if name in LEARNS_A_CHOICE else regime_oos
            for name in REGIME_WRAPPERS}


# 1. The wrappers traded (and were scored) on the days they learned from.
@pytest.mark.parametrize("name", REGIME_WRAPPERS)
def test_no_position_before_the_wrapper_is_out_of_sample(prices, first_out_of_sample_day,
                                                         name):
    signal = ADAPTIVE_STRATEGIES[name](prices)
    start = first_out_of_sample_day[name]
    assert (signal[signal.index < start] == 0).all()
    assert (signal[signal.index >= start] != 0).any()


# 2. regime_sized sized every position from the volatility of all labelled
#    days, out-of-sample ones included: rewriting the future changed the past.
@pytest.mark.parametrize("name", REGIME_WRAPPERS)
def test_positions_never_depend_on_later_prices(prices, name):
    n = len(prices)
    # Fixed, price-independent regime labels, so the only thing that can
    # carry the rewritten tail back into earlier positions is the wrapper.
    labels = pd.Series((np.arange(n) // 90) % 3, index=prices.index)
    labels.iloc[:60] = UNKNOWN
    cut = int(n * 0.85)
    original = ADAPTIVE_STRATEGIES[name](prices, regimes=labels)
    rewritten = ADAPTIVE_STRATEGIES[name](_rewrite_from(prices, cut), regimes=labels)
    before = prices.index[:cut]
    pd.testing.assert_series_equal(original[before], rewritten[before], check_names=False)
    # Not vacuous: the wrapper trades between its learning window and the cut.
    assert (original.iloc[int(n * LEARN_FRAC) + 1:cut] != 0).any()


def test_regime_sized_learns_sizes_from_the_learning_window_only(prices):
    n = len(prices)
    labels = pd.Series((np.arange(n) // 90) % 3, index=prices.index)
    labels.iloc[:60] = UNKNOWN
    signal = regime_sized(prices, regimes=labels)
    base = STRATEGIES["sma_crossover"](prices).astype(float)
    learn_end = _learn_cutoff(prices, LEARN_FRAC)
    returns = prices["Close"].pct_change()
    window = prices.index <= learn_end
    vol = {r: returns[(labels == r).to_numpy() & window].std() for r in (0, 1, 2)}
    expected_size = labels.map({r: min(vol.values()) / v for r, v in vol.items()})
    later = (prices.index > learn_end) & (base == 1).to_numpy()
    np.testing.assert_allclose(signal[later], expected_size[later])


# 5. regime_filtered(base="ml_direction") silently returned flat everywhere.
def test_a_base_with_no_position_in_the_learning_window_raises(prices):
    with pytest.raises(ValueError, match="holds no position on or before learn_end"):
        regime_filtered(prices, base="ml_direction")
    with pytest.raises(ValueError, match="holds no position on or before learn_end"):
        regime_switch(prices, candidates=("sma_crossover", "ml_direction"))


def test_regime_labels_that_are_never_out_of_sample_raise(prices):
    # A Viterbi-decoded HMM labels every day with knowledge of the whole
    # series, so no day of it is out of sample.
    regimes = detect_regimes(prices, decode="viterbi")
    with pytest.raises(ValueError, match="never out of sample"):
        regime_filtered(prices, regimes=regimes)


# 3. evaluate_out_of_sample scored a wrapper from the shared 70% split,
#    inside its regime model's fit window.
def test_evaluate_out_of_sample_starts_where_the_wrapper_may_trade(prices,
                                                                  first_out_of_sample_day):
    evaluated = evaluate_out_of_sample(prices, ADAPTIVE_STRATEGIES["regime_switch"])
    assert pd.Timestamp(evaluated["split_date"]) == first_out_of_sample_day["regime_switch"]


def test_evaluate_out_of_sample_keeps_the_shared_split_for_a_rule(prices):
    evaluated = evaluate_out_of_sample(prices, STRATEGIES["sma_crossover"])
    assert pd.Timestamp(evaluated["split_date"]) == prices.index[int(N_DAYS * 0.7)]


# ---- new API: the signal's own record, and rolling fold eligibility ----

@pytest.mark.parametrize("name", REGIME_WRAPPERS)
def test_signal_records_its_first_out_of_sample_day(prices, first_out_of_sample_day, name):
    from qbt.timesplit import out_of_sample_start

    signal = ADAPTIVE_STRATEGIES[name](prices)
    assert out_of_sample_start(signal) == first_out_of_sample_day[name]


def test_a_rule_records_no_out_of_sample_day(prices):
    from qbt.timesplit import out_of_sample_start

    assert out_of_sample_start(STRATEGIES["sma_crossover"](prices)) is None
    assert out_of_sample_start(ADAPTIVE_STRATEGIES["volatility_targeted"](prices)) is None


def test_rolling_scores_only_folds_where_the_strategy_can_hold_a_position(prices):
    from qbt.timesplit import split_date

    rolled = rolling_walk_forward(prices, STRATEGIES["ml_direction"], train_days=504,
                                  test_days=126, refit_per_fold=False)
    folds = rolled["folds"]
    first_tradable = split_date(prices.index, 0.7)
    eligible = folds["test_start"] >= first_tradable
    assert folds["can_hold_position"].tolist() == eligible.tolist()
    assert 0 < rolled["n_folds_scored"] == eligible.sum() < rolled["n_folds"]
    assert rolled["mean_sharpe"] == pytest.approx(folds.loc[eligible, "sharpe_ratio"].mean())


def test_refit_per_fold_can_hold_a_position_in_every_fold(prices):
    rolled = rolling_walk_forward(prices, STRATEGIES["ml_direction"], train_days=504,
                                  test_days=126)
    assert rolled["n_folds_scored"] == rolled["n_folds"]
    assert rolled["folds"]["can_hold_position"].all()
    # The fixed-model comparison pass scores only its own eligible folds.
    fixed = rolled["fixed_model_folds"]
    assert rolled["mean_sharpe_fixed_model"] == pytest.approx(
        fixed.loc[fixed["can_hold_position"], "sharpe_ratio"].mean())
