"""
Sanity-checks regime.py, regime_features.py, adaptive.py and the
regime-aware paths through backtest.py / ml_strategy.py / walk_forward.py,
against synthetic data with KNOWN regime structure.

Same approach as test_logic.py -- assertions against synthetic data, no
network required. The synthetic series here cycles through three
deliberately different states (calm drift up, flat and choppy, violent
selloff), so a working detector should recover something close to them.
That's the point: on real data you can never check a regime label against
ground truth, so the only place you can verify the machinery is where you
built the truth yourself.
"""

import numpy as np
import pandas as pd
import pytest

from qbt.adaptive import ADAPTIVE_STRATEGIES, ALL_STRATEGIES, describe_choices, describe_filter
from qbt.analytics import full_report, performance_by_regime
from qbt.backtest import run_backtest
from qbt.ml import model_report
from qbt.regime import (
    REGIME_METHODS, UNKNOWN, detect_regimes, detect_regimes_walk_forward,
    regime_episodes, regime_stability, regime_summary, smooth_labels,
)
from qbt.regime_features import build_regime_features, standardize_features
from qbt.strategies import STRATEGIES
from qbt.walk_forward import (
    compare_strategies, evaluate_out_of_sample, evaluate_with_regimes, rolling_walk_forward,
)

# --------------------------------------------------------------------------
# Synthetic data with known regimes
# --------------------------------------------------------------------------
N = 2600
SPECS = [(0.0009, 0.0065), (0.0000, 0.0110), (-0.0016, 0.0290)]


def _make_series(rng: np.random.Generator, with_ohlcv: bool = True) -> pd.DataFrame:
    segments, total, i = [], 0, 0
    while total < N:
        drift, vol = SPECS[i % 3]
        length = int(rng.integers(150, 320))
        segments.append(rng.normal(drift, vol, length))
        total += length
        i += 1

    returns = np.concatenate(segments)[:N]
    close = 100 * np.cumprod(1 + returns)
    index = pd.date_range("2008-01-01", periods=N, freq="B")

    if not with_ohlcv:
        return pd.DataFrame({"Close": close}, index=index)
    return pd.DataFrame({
        "Close": close,
        "High": close * (1 + np.abs(rng.normal(0, 0.004, N))),
        "Low": close * (1 - np.abs(rng.normal(0, 0.004, N))),
        "Volume": rng.lognormal(15, 0.4, N),
    }, index=index)


@pytest.fixture(scope="module")
def df():
    return _make_series(np.random.default_rng(20240826), with_ohlcv=True)


@pytest.fixture(scope="module")
def df_close_only():
    return _make_series(np.random.default_rng(20240826), with_ohlcv=False)


@pytest.fixture(scope="module")
def features(df):
    return build_regime_features(df)


@pytest.fixture(scope="module")
def features_close_only(df_close_only):
    return build_regime_features(df_close_only)


@pytest.fixture(scope="module")
def results(df):
    return {
        method: detect_regimes(df, method=method, n_regimes=3, fit_frac=0.7)
        for method in REGIME_METHODS
    }


@pytest.fixture(scope="module")
def hmm_result(results):
    return results["hmm"]


@pytest.fixture(scope="module")
def backtest_costs(df, hmm_result):
    signal = STRATEGIES["sma_crossover"](df)
    free = run_backtest(df, signal, cost_bps=0)
    costed = run_backtest(df, signal, cost_bps=10, regimes=hmm_result.labels)
    return free, costed


@pytest.fixture(scope="module")
def described_choices(df, hmm_result):
    return describe_choices(df, regimes=hmm_result)


@pytest.fixture(scope="module")
def strategy_comparison(df, hmm_result):
    return compare_strategies(
        df,
        {name: (fn, {"regimes": hmm_result} if name in ADAPTIVE_STRATEGIES else {})
         for name, fn in ALL_STRATEGIES.items()},
        cost_bps=5,
    )


@pytest.fixture(scope="module")
def raw_kmeans_labels(df):
    return detect_regimes(df, method="kmeans", n_regimes=3, fit_frac=0.7, smooth="none").labels


@pytest.fixture(scope="module")
def wf_regimes(df):
    # refit_every=252 (vs. the 63-day default) keeps this fixture's ~30
    # EM refits down to ~8 without changing what's being asserted: the
    # causal flag and the absence of labels before the initial window
    # don't depend on refit cadence.
    return detect_regimes_walk_forward(df, method="hmm", n_regimes=3, refit_every=252)


# --------------------------------------------------------------------------
# 1. Features must be point-in-time and survive a Close-only frame
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table_name", ["OHLCV", "Close-only"])
def test_regime_features_usable(features, features_close_only, table_name):
    table = features if table_name == "OHLCV" else features_close_only
    assert not table.empty, f"{table_name}: no features produced"
    all_nan = [c for c in table.columns if table[c].notna().sum() == 0]
    assert not all_nan, f"{table_name}: columns are entirely NaN: {all_nan}"
    clean = standardize_features(table).dropna()
    assert len(clean) > 500, f"{table_name}: only {len(clean)} usable rows after warm-up"


def test_regime_features_point_in_time(df, features):
    # The expanding percentile must never see the future: recomputing it on a
    # truncated series has to give the same answer for the rows that survive.
    truncated = build_regime_features(df.iloc[:1500])
    overlap = truncated.index.intersection(features.index)
    for column in ("vol_percentile", "vol_20d", "trend_60d", "efficiency_ratio"):
        left = features.loc[overlap, column].dropna()
        right = truncated.loc[overlap, column].dropna()
        shared = left.index.intersection(right.index)
        assert np.allclose(left[shared], right[shared], equal_nan=True), (
            f"{column} changed when later data was removed -- it is looking ahead"
        )


# --------------------------------------------------------------------------
# 2. Every detection method produces usable, persistent labels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", REGIME_METHODS)
def test_detect_regimes_produces_usable_labels(df, results, method):
    result = results[method]
    labels = result.labels

    assert labels.index.equals(df.index), f"{method}: labels not aligned to the price index"
    assert not labels.isna().any(), f"{method}: NaN in labels"
    assert labels.isin(list(result.names) + [UNKNOWN]).all(), f"{method}: label outside the named set"
    assert result.valid().sum() > 400, f"{method}: only {result.valid().sum()} labelled days"
    assert set(result.names) == set(range(len(result.names))), f"{method}: regime IDs are not 0..k-1"

    stability = regime_stability(labels)
    assert stability["labelled_days"] > 0, f"{method}: regime_stability found no labelled days"


@pytest.mark.parametrize("method", REGIME_METHODS)
def test_regime_ids_ordered_by_volatility(df, results, method):
    # Regime IDs are ordered by volatility -- regime 0 must be the calmest.
    # Everything downstream (color ramps, default parameter ladders, size
    # maps) depends on this holding.
    result = results[method]
    returns = df["Close"].pct_change()
    vols = [returns[result.labels == r].std() for r in sorted(result.names)]
    vols = [v for v in vols if not pd.isna(v)]
    assert vols == sorted(vols), f"{method}: regime IDs not ordered by volatility: {vols}"


# --------------------------------------------------------------------------
# 3. The HMM should find persistent states, not noise
# --------------------------------------------------------------------------

def test_hmm_states_are_persistent(hmm_result):
    matrix = hmm_result.transition_matrix()
    diagonal = np.diag(matrix.to_numpy())
    assert (diagonal > 0.85).all(), f"HMM states are not persistent enough: diagonal={diagonal}"


def test_hmm_converged(hmm_result):
    assert hmm_result.model.converged_, "HMM did not converge"


def test_hmm_regime_summary_has_meaningful_spread(df, hmm_result):
    summary = regime_summary(hmm_result, df)
    assert len(summary) >= 2, "HMM produced fewer than two populated regimes"
    vol_spread = summary["ann_volatility"].max() - summary["ann_volatility"].min()
    assert vol_spread > 0.05, (
        f"Detected regimes differ by only {vol_spread:.1%} in volatility -- on data built with "
        "three very different states, that means detection failed"
    )


# --------------------------------------------------------------------------
# 4. Smoothing is causal and actually reduces flicker
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method, kwargs", [
    ("min_duration", {"min_duration": 5}),
    ("median", {"window": 5}),
])
def test_smoothing_does_not_increase_episode_count(raw_kmeans_labels, method, kwargs):
    raw_episodes = len(regime_episodes(raw_kmeans_labels))
    smoothed = smooth_labels(raw_kmeans_labels, method=method, **kwargs)
    smoothed_episodes = len(regime_episodes(smoothed))
    assert smoothed_episodes <= raw_episodes, f"{method} increased the episode count"


def test_min_duration_smoothing_is_causal(raw_kmeans_labels):
    # Smoothing row t must not depend on anything after t.
    prefix = smooth_labels(raw_kmeans_labels.iloc[:1200], method="min_duration", min_duration=5)
    full = smooth_labels(raw_kmeans_labels, method="min_duration", min_duration=5)
    assert (prefix == full.iloc[:1200]).all(), "min_duration smoothing is using future labels"


# --------------------------------------------------------------------------
# 5. Walk-forward detection labels nothing it hasn't earned
# --------------------------------------------------------------------------

def test_walk_forward_regimes_marked_causal(wf_regimes):
    assert wf_regimes.causal, "walk-forward result is not marked causal"


def test_walk_forward_regimes_no_labels_before_training_window(wf_regimes):
    first_labelled = wf_regimes.labels[wf_regimes.labels != UNKNOWN].index[0]
    assert wf_regimes.labels.loc[:first_labelled].iloc[:-1].eq(UNKNOWN).all(), (
        "walk-forward produced labels before its initial training window"
    )


# --------------------------------------------------------------------------
# 6. backtest.py stays backward compatible and costs behave
# --------------------------------------------------------------------------

def test_zero_cost_bps_charges_nothing(backtest_costs):
    free, _ = backtest_costs
    assert np.allclose(free["cost"], 0), "cost_bps=0 charged a cost"


def test_nonzero_cost_bps_charges_something(backtest_costs):
    _, costed = backtest_costs
    assert costed["cost"].sum() > 0, "cost_bps=10 charged nothing"


def test_costs_reduce_strategy_returns(backtest_costs):
    free, costed = backtest_costs
    assert costed["strategy_return"].sum() < free["strategy_return"].sum(), "costs did not reduce returns"


def test_regime_column_passthrough(backtest_costs, hmm_result):
    _, costed = backtest_costs
    assert "regime" in costed.columns
    assert costed["regime"].isin(list(hmm_result.names) + [UNKNOWN]).all()


def test_performance_by_regime_table(backtest_costs, hmm_result):
    _, costed = backtest_costs
    by_regime = performance_by_regime(costed, hmm_result.labels, hmm_result.names)
    assert len(by_regime) >= 2, "per-regime performance table is degenerate"
    assert by_regime["days"].sum() == int(hmm_result.valid().sum()), (
        "per-regime day counts don't add up to the labelled days"
    )


# --------------------------------------------------------------------------
# 7. Adaptive strategies keep the strategy interface
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ADAPTIVE_STRATEGIES)
def test_adaptive_strategy_position_and_stats(df, hmm_result, name):
    fn = ADAPTIVE_STRATEGIES[name]
    position = fn(df, regimes=hmm_result)

    assert position.index.equals(df.index), f"{name}: position not aligned to the price index"
    assert not position.isna().any(), f"{name}: NaN in position"
    assert ((position >= 0) & (position <= 1)).all(), f"{name}: position outside [0, 1]"

    stats = full_report(run_backtest(df, position, cost_bps=5))
    assert -1.0 <= stats["max_drawdown"] <= 0.0, f"{name}: max_drawdown out of range"
    assert 0.0 <= stats["exposure"] <= 1.0, f"{name}: exposure out of range"


@pytest.mark.parametrize("name", STRATEGIES)
def test_base_strategies_remain_binary(df, name):
    # adaptive.py exists precisely so that the fractional-position logic
    # never leaked into the base strategies.
    assert STRATEGIES[name](df).isin([0, 1]).all(), f"{name} is no longer binary"


# --------------------------------------------------------------------------
# 8. Automatic choices are learned from the past only
# --------------------------------------------------------------------------

def test_learning_window_ends_before_full_sample(df, described_choices):
    assert described_choices["learn_end"] < df.index[-1], "the learning window covers the whole sample"


def test_learning_window_has_days(described_choices):
    assert (described_choices["table"]["days"] > 0).any(), "no learning-window days found"


def test_every_regime_gets_a_decision(hmm_result, described_choices):
    assert set(described_choices["choices"]) == set(hmm_result.names), "a regime got no decision"


def test_describe_filter_produces_allow_list(df, hmm_result):
    filtered = describe_filter(df, base="sma_crossover", regimes=hmm_result)
    assert "allowed" in filtered
    assert set(filtered["allowed"]) <= set(hmm_result.names)


def test_learned_choice_ignores_post_learning_data(df, hmm_result, described_choices):
    # Changing data AFTER the learning window must not change what was learned.
    learn_end = described_choices["learn_end"]
    mutated = df.copy()
    tail = mutated.index > learn_end
    mutated.loc[tail, "Close"] = mutated.loc[tail, "Close"] * 1.5
    mutated_choices = describe_choices(mutated, regimes=hmm_result)["choices"]
    assert mutated_choices == described_choices["choices"], (
        "the learned per-regime choice changed when only post-learning data changed -- it is looking ahead"
    )


# --------------------------------------------------------------------------
# 9. Regime-aware ML paths
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["feature", "conditional"])
def test_regime_aware_ml_report(df, hmm_result, mode):
    report = model_report(df, model_type="logistic", regimes=hmm_result.labels, regime_mode=mode)
    assert 0 <= report["test_accuracy"] <= 1, f"{mode}: implausible test accuracy"
    assert "by_regime" in report, f"{mode}: missing per-regime breakdown"


def test_ml_report_without_regimes(df):
    baseline = model_report(df, model_type="logistic")
    assert 0 <= baseline["test_accuracy"] <= 1, "implausible baseline test accuracy"


# --------------------------------------------------------------------------
# 10. Validation layer
# --------------------------------------------------------------------------

def test_rolling_walk_forward_produces_enough_folds(df):
    rolling = rolling_walk_forward(df, STRATEGIES["sma_crossover"], train_days=756, test_days=126)
    assert rolling["n_folds"] >= 5, f"only {rolling['n_folds']} folds"
    assert 0 <= rolling["pct_folds_positive"] <= 1
    assert len(rolling["oos_equity"]) == rolling["n_folds"] * 126


def test_compare_strategies_no_errors(strategy_comparison):
    assert strategy_comparison["error"].isna().all(), (
        f"strategies errored: {strategy_comparison[strategy_comparison['error'].notna()]}"
    )


def test_compare_strategies_covers_every_strategy(strategy_comparison):
    assert len(strategy_comparison) == len(ALL_STRATEGIES)


@pytest.mark.parametrize("name", ALL_STRATEGIES)
def test_validation_entry_points_for_every_strategy(df, hmm_result, name):
    # Every strategy must survive every validation entry point. This test
    # exists because it didn't: evaluate_with_regimes() originally took
    # **strategy_params, which collided with its own `regimes` argument the
    # moment an adaptive wrapper (which takes `regimes` too) was passed
    # through it. Nothing caught that until this ran.
    fn = ALL_STRATEGIES[name]
    strategy_params = {"regimes": hmm_result} if name in ADAPTIVE_STRATEGIES else {}

    evaluation = evaluate_with_regimes(
        df, fn, hmm_result, cost_bps=5, strategy_params=strategy_params
    )
    assert set(evaluation) >= {"in_sample", "out_sample", "in_sample_by_regime",
                               "out_sample_by_regime", "regime_mix"}
    assert not evaluation["regime_mix"].empty, f"{name}: empty regime mix"

    fold_result = rolling_walk_forward(df, fn, cost_bps=5, **strategy_params)
    assert fold_result["n_folds"] >= 5, f"{name}: too few folds"


def test_evaluate_out_of_sample_shape_unchanged(df):
    # The original single-split entry point must be untouched.
    legacy = evaluate_out_of_sample(df, STRATEGIES["sma_crossover"])
    assert set(legacy) == {"split_date", "in_sample", "out_sample"}, "evaluate_out_of_sample changed shape"
