"""
Tests for C2 (qbt.ml.models, qbt.ml.metrics, the new baselines and the
HMM feature cache) on synthetic data; no network.

The leakage tests follow the C1 canaries: change everything a fold is
not allowed to see (its test labels and every later row) and require
that fold's out-of-sample probabilities to be bit-identical.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.metrics import roc_auc_score

from qbt.ml import features as features_module
from qbt.ml.baselines import conditional_prior_proba, predict_rules_regime, prior_proba
from qbt.ml.evaluate import build_dataset
from qbt.ml.features import clear_hmm_cache, daily_features, hmm_cache_info
from qbt.ml.metrics import (
    auc,
    brier,
    brier_binary,
    calibration_bins,
    log_loss,
    monthly_sharpe,
    paired_mean_bootstrap,
    paired_sharpe_bootstrap,
    predicted_class,
)
from qbt.ml.models import (
    LIGHTGBM_GRID,
    LOGISTIC_GRID,
    PROB_FLOOR,
    classes_for,
    config_id,
    fit_predict_proba,
    grid_for,
    tune,
    walk_forward_predict,
)
from qbt.ml.splits import Fold, walk_forward_splits

HMM_KWARGS = {"initial_train": 252, "refit_every": 42}


def _prices(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 200) % 2 == 0, 0.007, 0.02)
    index = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0.0004, vol))}, index=index)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return _prices()


def _toy(n: int = 90, n_classes: int = 2, seed: int = 0):
    """A monthly-shaped dataset whose label depends on the first feature."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2000-01-31", periods=n, freq="ME")
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "regime_rules": rng.integers(0, 4, n).astype(float)}, index=index)
    if n_classes == 2:
        y = (X["a"] + 0.5 * rng.normal(size=n) > 0).astype(float)
    else:
        y = pd.Series(np.digitize(X["a"], [-0.7, 0.0, 0.7]).astype(float), index=index)
    t1 = pd.Series(index, index=index).shift(-1).fillna(index[-1] + pd.offsets.MonthEnd(1))
    return X, y, t1


def _lightgbm():
    pytest.importorskip("lightgbm")


# ---- HMM cache ----

def test_hmm_features_are_computed_once_per_dataset(prices, monkeypatch):
    clear_hmm_cache()
    calls = []
    original = features_module._compute_regime_hmm

    def counting(df, kwargs):
        calls.append(len(df))
        return original(df, kwargs)

    monkeypatch.setattr(features_module, "_compute_regime_hmm", counting)
    first = build_dataset(prices, "direction", include_hmm=True, hmm_kwargs=HMM_KWARGS)
    second = build_dataset(prices, "regime", include_hmm=True, hmm_kwargs=HMM_KWARGS)
    assert calls == [len(prices)]
    assert hmm_cache_info()["computed"] == 1 and hmm_cache_info()["hits"] == 1
    hmm_cols = [c for c in first.X.columns if c.startswith(("hmm_", "regime_hmm"))]
    common = first.X.index.intersection(second.X.index)
    pd.testing.assert_frame_equal(first.X.loc[common, hmm_cols], second.X.loc[common, hmm_cols])

    # Different prices or different HMM arguments are a different dataset.
    build_dataset(prices.iloc[:-30], "direction", include_hmm=True, hmm_kwargs=HMM_KWARGS)
    build_dataset(prices, "direction", include_hmm=True,
                  hmm_kwargs={**HMM_KWARGS, "refit_every": 63})
    assert len(calls) == 3
    clear_hmm_cache()


def test_cached_hmm_features_equal_uncached_and_are_not_aliased(prices):
    clear_hmm_cache()
    cold = daily_features(prices, hmm_kwargs=HMM_KWARGS)
    cold["regime_hmm"] = -99.0          # mutating a result must not reach the cache
    warm = daily_features(prices, hmm_kwargs=HMM_KWARGS)
    clear_hmm_cache()
    fresh = daily_features(prices, hmm_kwargs=HMM_KWARGS)
    pd.testing.assert_frame_equal(warm, fresh)
    clear_hmm_cache()


# ---- models ----

def test_grids_and_config_ids():
    assert grid_for("logistic") is LOGISTIC_GRID and grid_for("lightgbm") is LIGHTGBM_GRID
    assert len({config_id("lightgbm", p) for p in LIGHTGBM_GRID}) == len(LIGHTGBM_GRID)
    with pytest.raises(ValueError):
        grid_for("mlp")
    with pytest.raises(ValueError):
        classes_for("volatility")


@pytest.mark.parametrize("model", ["logistic", "lightgbm"])
def test_probabilities_cover_every_class_and_sum_to_one(model):
    if model == "lightgbm":
        _lightgbm()
    X, y, _ = _toy(n_classes=4)
    classes = classes_for("regime")
    train = y != 3.0                     # a class the training set never shows
    proba = fit_predict_proba(model, grid_for(model)[0], X[train], y[train], X, classes)
    assert proba.shape == (len(X), 4)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= PROB_FLOOR / (1 + 4 * PROB_FLOOR)).all()
    assert np.allclose(proba[:, 3], proba[:, 3].min())   # floored, nothing learned


def test_single_class_training_set_predicts_that_class():
    X, y, _ = _toy()
    ones = pd.Series(1.0, index=y.index)
    proba = fit_predict_proba("logistic", {"C": 1.0}, X, ones, X.iloc[:5], (0.0, 1.0))
    assert (proba.argmax(axis=1) == 1).all()
    with pytest.raises(ValueError):
        fit_predict_proba("logistic", {"C": 1.0}, X, ones + 5, X, (0.0, 1.0))


@pytest.mark.parametrize("model", ["logistic", "lightgbm"])
def test_tuning_selects_lowest_inner_log_loss(model):
    if model == "lightgbm":
        _lightgbm()
    X, y, t1 = _toy()
    t0 = pd.Series(X.index, index=X.index)
    result = tune(model, X, y, t0, t1, (0.0, 1.0), n_splits=4, embargo=2)
    assert len(result.trials) == len(grid_for(model))
    assert sum(t["selected"] for t in result.trials) == 1
    best = min(result.trials, key=lambda t: t["inner_log_loss"])
    assert best["selected"] and best["params"] == result.params


@pytest.mark.parametrize("model", ["logistic", "lightgbm"])
def test_outer_predictions_never_see_test_or_later_rows(model):
    """
    Scrambling a fold's test labels and every row after its test block
    (features and labels) must leave that fold's probabilities unchanged:
    neither the inner tuning nor the refit may read them.
    """
    if model == "lightgbm":
        _lightgbm()
    X, y, t1 = _toy(n=84)
    folds = walk_forward_splits(pd.Series(X.index, index=X.index), t1,
                                min_train=36, test_size=12, embargo=2)
    clean = walk_forward_predict(model, X, y, t1, folds, "direction")
    rng = np.random.default_rng(1)
    for fold in folds:
        X2, y2 = X.copy(), y.copy()
        y2.iloc[fold.test[0]:] = rng.integers(0, 2, len(y) - fold.test[0]).astype(float)
        after = slice(fold.test[-1] + 1, None)
        X2.iloc[after] = rng.normal(size=X2.iloc[after].shape)
        scrambled = walk_forward_predict(model, X2, y2, t1, folds, "direction")
        rows = X.index[fold.test]
        np.testing.assert_array_equal(clean.proba.loc[rows].to_numpy(),
                                      scrambled.proba.loc[rows].to_numpy())


def test_tuning_uses_the_outer_training_rows_only():
    X, y, t1 = _toy(n=84)
    folds = walk_forward_splits(pd.Series(X.index, index=X.index), t1,
                                min_train=36, test_size=12, embargo=2)
    wf = walk_forward_predict("logistic", X, y, t1, folds, "direction")
    for fold in folds:
        trials = [t for t in wf.trials if t["outer_fold"] == fold.number]
        assert len(trials) == len(LOGISTIC_GRID)
        assert {t["n_train"] for t in trials} == {len(fold.train)}
        assert all(t["inner_n"] <= len(fold.train) for t in trials)
    assert wf.proba.index.is_unique
    assert list(wf.proba.columns) == [0.0, 1.0]


def test_lightgbm_is_deterministic():
    _lightgbm()
    X, y, _ = _toy(n_classes=4)
    a = fit_predict_proba("lightgbm", LIGHTGBM_GRID[-1], X, y, X, classes_for("regime"))
    b = fit_predict_proba("lightgbm", LIGHTGBM_GRID[-1], X, y, X, classes_for("regime"))
    np.testing.assert_array_equal(a, b)


# ---- baselines ----

def test_prior_and_conditional_prior_use_only_training_rows():
    y = pd.Series([0.0, 0.0, 1.0, 3.0, 3.0, 3.0])
    given = pd.Series([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    classes = classes_for("regime")
    prior = prior_proba(y, 2, classes, floor=0.0)
    np.testing.assert_allclose(prior, [[2 / 6, 1 / 6, 0, 3 / 6]] * 2)
    cond = conditional_prior_proba(y, given, pd.Series([0.0, 2.0, 1.0]), classes, floor=0.0)
    np.testing.assert_allclose(cond[0], [2 / 3, 1 / 3, 0, 0])
    np.testing.assert_allclose(cond[1], [0, 0, 0, 1])
    np.testing.assert_allclose(cond[2], prior[0])            # unseen regime -> prior
    with pytest.raises(ValueError):
        prior_proba(y.iloc[:0], 1, classes)


def test_rules_regime_baseline_reads_the_decision_date_feature():
    X, _, _ = _toy()
    fold = Fold(0, np.arange(10), np.arange(10, 15), np.array([], int), np.array([], int))
    np.testing.assert_array_equal(predict_rules_regime(X, fold),
                                  X["regime_rules"].iloc[10:15].to_numpy())


# ---- metrics ----

def test_scores_match_scikit_learn():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200).astype(float)
    p = np.clip(0.3 * y + rng.uniform(0, 0.7, 200), 0.01, 0.99)
    proba = np.column_stack([1 - p, p])
    classes = (0.0, 1.0)
    assert auc(y, proba, classes)[0] == pytest.approx(roc_auc_score(y, p))
    assert log_loss(y, proba, classes) == pytest.approx(sk_log_loss(y, proba))
    assert brier(y, proba, classes) == pytest.approx(2 * brier_binary(y, p))

    y4 = rng.integers(0, 4, 300).astype(float)
    raw = rng.uniform(size=(300, 4)) + np.eye(4)[y4.astype(int)]
    proba4 = raw / raw.sum(axis=1, keepdims=True)
    value, n_classes = auc(y4, proba4, classes_for("regime"))
    assert n_classes == 4
    assert value == pytest.approx(roc_auc_score(y4, proba4, multi_class="ovr",
                                                average="macro"))
    # A class absent from y has no AUC and is skipped, not counted as 0.5.
    present = y4 != 2.0
    assert auc(y4[present], proba4[present], classes_for("regime"))[1] == 3
    assert np.isnan(auc(np.ones(5), np.full((5, 2), 0.5), classes)[0])
    np.testing.assert_array_equal(predicted_class(proba4, classes_for("regime")),
                                  proba4.argmax(axis=1).astype(float))


def test_calibration_bins_partition_the_forecasts():
    p = np.linspace(0, 1, 23)
    bins = calibration_bins(p, (p > 0.5).astype(float), n_bins=5)
    assert sum(b["n"] for b in bins) == 23 and len(bins) == 5
    assert [b["mean_forecast"] for b in bins] == sorted(b["mean_forecast"] for b in bins)


def test_paired_bootstrap_is_paired_and_seeded():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 120).astype(bool)
    same = paired_mean_bootstrap({"m": a, "copy": a.copy()}, "m")
    assert same["vs"]["copy"]["difference"] == 0
    assert same["vs"]["copy"]["difference_ci"] == [0.0, 0.0]
    better = paired_mean_bootstrap({"m": np.ones(120), "b": a}, "m")
    assert better["vs"]["b"]["difference_ci"][0] > 0
    assert better["vs"]["b"]["share_of_replicates_not_better"] == 0
    loss = paired_mean_bootstrap({"m": np.zeros(120), "b": np.ones(120)}, "m",
                                 higher_is_better=False)
    assert loss["vs"]["b"]["share_of_replicates_not_better"] == 0
    assert paired_mean_bootstrap({"m": a, "b": ~a}, "m") == \
        paired_mean_bootstrap({"m": a, "b": ~a}, "m")


def test_paired_sharpe_bootstrap():
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.04, 100)
    same = paired_sharpe_bootstrap(r, r.copy())
    assert same["difference"] == 0 and same["difference_ci"] == [0.0, 0.0]
    assert same["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(12))
    assert monthly_sharpe(np.zeros(5)) != monthly_sharpe(np.zeros(5))   # NaN, not 0
    with pytest.raises(ValueError):
        paired_sharpe_bootstrap(r, r[:-1])


def test_tuning_accepts_an_explicit_grid():
    X, y, t1 = _toy()
    t0 = pd.Series(X.index, index=X.index)
    grid = [{"C": 0.5}, {"C": 5.0}]
    result = tune("logistic", X, y, t0, t1, (0.0, 1.0), n_splits=4, embargo=2, grid=grid)
    assert [t["params"] for t in result.trials] == grid
    assert result.params in grid
