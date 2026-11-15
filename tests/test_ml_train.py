"""
Tests for C3's library side: model artifacts (qbt.ml.models.to_artifact /
from_artifact) and training for serving (qbt.ml.train), on synthetic
prices; no network, no database.

Reproducibility is asserted at the strongest level that holds: the same
frame and configuration give a byte-identical artifact (LightGBM writes
its text format deterministically, with no timestamps), and the loaded
artifact predicts bit-identically to the fitted model.
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from qbt.ml.evaluate import BAR, baseline_predictions, build_dataset  # noqa: E402
from qbt.ml.features import month_end_dates  # noqa: E402
from qbt.ml.models import (  # noqa: E402
    ARTIFACT_FORMAT,
    LIGHTGBM_GRID,
    classes_for,
    fit,
    from_artifact,
    to_artifact,
)
from qbt.ml.splits import walk_forward_splits  # noqa: E402
from qbt.ml.train import (  # noqa: E402
    CLASS_NAMES,
    TrainingConfig,
    features_hash,
    serving_features,
    train_for_serving,
)

SMALL = {"min_train": 24, "test_size": 12, "n_boot": 200}
HMM_KWARGS = {"initial_train": 252, "refit_every": 42}


def _prices(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 200) % 2 == 0, 0.007, 0.02)
    index = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0.0004, vol))}, index=index)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return _prices()


@pytest.fixture(scope="module")
def trained(prices):
    return train_for_serving(prices, TrainingConfig(feature_set="base", **SMALL))


def _toy(n_classes: int = 4, n: int = 120):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series(np.digitize(X["a"], [-0.7, 0.0, 0.7]).astype(float)
                  % n_classes)
    return X, y


# ---- artifacts ----

@pytest.mark.parametrize("n_classes", [2, 4])
def test_artifact_round_trip_predicts_bit_identically(n_classes):
    X, y = _toy(n_classes)
    classes = classes_for("direction" if n_classes == 2 else "regime")
    fitted = fit("lightgbm", LIGHTGBM_GRID[-1], X, y, classes)
    loaded = from_artifact(to_artifact(fitted))
    np.testing.assert_array_equal(loaded.predict_proba(X), fitted.predict_proba(X))
    assert loaded.columns == fitted.columns and loaded.classes == fitted.classes


def test_same_fit_gives_byte_identical_artifact():
    X, y = _toy()
    a = to_artifact(fit("lightgbm", LIGHTGBM_GRID[0], X, y, classes_for("regime")))
    b = to_artifact(fit("lightgbm", LIGHTGBM_GRID[0], X, y, classes_for("regime")))
    assert a == b
    document = json.loads(a)
    assert document["format"] == ARTIFACT_FORMAT and "lightgbm_model" in document


def test_missing_class_and_single_class_models_round_trip():
    X, y = _toy()
    classes = classes_for("regime")
    without_3 = y != 3.0
    fitted = fit("lightgbm", LIGHTGBM_GRID[0], X[without_3], y[without_3], classes)
    loaded = from_artifact(to_artifact(fitted))
    np.testing.assert_array_equal(loaded.predict_proba(X), fitted.predict_proba(X))
    constant = fit("lightgbm", LIGHTGBM_GRID[0], X, pd.Series(2.0, index=X.index), classes)
    loaded = from_artifact(to_artifact(constant))
    assert (loaded.predict_proba(X).argmax(axis=1) == 2).all()


def test_artifact_checks():
    X, y = _toy(2)
    with pytest.raises(NotImplementedError):
        to_artifact(fit("logistic", {"C": 1.0}, X, y, (0.0, 1.0)))
    artifact = to_artifact(fit("lightgbm", LIGHTGBM_GRID[0], X, y, (0.0, 1.0)))
    document = json.loads(artifact)
    with pytest.raises(ValueError, match="format"):
        from_artifact(json.dumps(document | {"format": "pickle"}).encode())
    with pytest.raises(ValueError, match="floors"):
        from_artifact(json.dumps(document | {"prob_floor": 0.01}).encode())
    with pytest.raises(ValueError, match="columns"):
        from_artifact(json.dumps(document | {"columns": ["a"]}).encode())
    fitted = from_artifact(artifact)
    with pytest.raises(ValueError, match="expected columns"):
        fitted.predict_proba(X[["b", "a", "c"]])


# ---- training for serving ----

def test_training_is_reproducible_byte_for_byte(prices, trained):
    again = train_for_serving(prices, TrainingConfig(feature_set="base", **SMALL))
    assert again.artifact == trained.artifact
    assert again.artifact_sha256 == trained.artifact_sha256
    assert again.metrics == trained.metrics
    assert again.serving_features_sha256 == trained.serving_features_sha256


def test_a_different_seed_changes_only_the_recorded_seed(prices, trained):
    """
    Bagging and feature subsampling are off, so LightGBM's seed changes no
    tree: the artifact differs only in the parameters it records, and the
    predictions are identical.
    """
    other = train_for_serving(prices, TrainingConfig(feature_set="base", seed=5, **SMALL))
    assert other.artifact != trained.artifact
    a, b = json.loads(trained.artifact), json.loads(other.artifact)
    diff = [(x, y) for x, y in zip(a["lightgbm_model"].splitlines(),
                                   b["lightgbm_model"].splitlines(), strict=True) if x != y]
    assert diff and all("seed" in x for x, _ in diff)
    row = pd.DataFrame([trained.serving_features])
    np.testing.assert_array_equal(from_artifact(trained.artifact).predict_proba(row),
                                  from_artifact(other.artifact).predict_proba(row))


def test_metrics_report_out_of_sample_accuracy_against_the_bar(prices, trained):
    ds = build_dataset(prices, "regime", include_hmm=False)
    folds = walk_forward_splits(ds.t0, ds.t1, min_train=24, test_size=12)
    base = baseline_predictions(ds, folds)
    oos, bar = trained.metrics["out_of_sample"], trained.metrics["bar"]
    assert oos["n_months"] == len(base)
    assert bar["name"] == BAR["regime"] == "rules_regime_lookup"
    assert bar["accuracy"] == pytest.approx(
        float((base["rules_regime_lookup"] == base["y"]).mean()))
    assert bar["difference"] == pytest.approx(oos["accuracy"] - bar["accuracy"])
    assert bar["beats"] == (bar["difference_ci"][0] > 0 and bar["mcnemar_p"] < 0.05)
    assert set(trained.metrics["baselines"]) == {
        "majority_class", "previous_label", "rules_regime_at_decision", "rules_regime_lookup"}


def test_trained_on_every_labelled_month_and_served_on_the_next(prices, trained):
    ds = build_dataset(prices, "regime", include_hmm=False)
    assert trained.n_train == len(ds.y)
    assert trained.train_last_decision == ds.t0.iloc[-1].date()
    last_month_end = month_end_dates(pd.DatetimeIndex(prices.index))[-1].date()
    # The served month is the last decision date: its label isn't known yet,
    # and the last training label ends exactly there.
    assert trained.serving_as_of == trained.train_last_label_end == last_month_end
    assert list(trained.serving_features) == trained.features == list(ds.X.columns)
    assert set(CLASS_NAMES["regime"]) == set(trained.classes)


def test_serving_features_are_point_in_time(prices, trained):
    """Prices after as_of (an incomplete month) don't change the served input."""
    extended = _prices(1510)
    assert (extended.loc[prices.index] == prices).all().all()
    as_of, row = serving_features(extended, trained.features, False, None)
    assert as_of == trained.serving_as_of
    assert features_hash(as_of, row) == trained.serving_features_sha256


def test_features_hash_sees_one_ulp():
    row = pd.Series([0.1, 0.2], index=["a", "b"])
    nudged = row.copy()
    nudged["b"] = np.nextafter(0.2, 1.0)
    as_of = pd.Timestamp("2024-12-31").date()
    assert features_hash(as_of, row) != features_hash(as_of, nudged)
    assert features_hash(as_of, row) != features_hash(as_of, row.rename({"a": "c"}))
    assert features_hash(as_of, row) == features_hash(as_of, row.copy())


def test_hmm_feature_set_trains_and_serves(prices):
    trained = train_for_serving(prices, TrainingConfig(hmm_kwargs=HMM_KWARGS, **SMALL))
    assert {"regime_hmm", "hmm_p0"} <= set(trained.features)
    assert trained.metrics["out_of_sample"]["n_months"] > 0


def test_direction_target_uses_the_majority_bar(prices):
    trained = train_for_serving(prices, TrainingConfig(target="direction", feature_set="base",
                                                       **SMALL))
    assert trained.metrics["bar"]["name"] == "majority_class"
    assert trained.classes == [0.0, 1.0]


def test_config_validation():
    for bad in ({"target": "volatility"}, {"feature_set": "all"}, {"model": "logistic"},
                {"seed": -1}):
        with pytest.raises(ValueError):
            TrainingConfig(**bad)
    assert dataclasses.asdict(TrainingConfig())["model"] == "lightgbm"
