"""
Regression tests for the daily ML strategy's leakage audit
(scripts/audit_daily_ml.py, results/ml/daily_ml_audit.json), one per
finding. Each goes through functions that existed before the fix, so
against the pre-fix code (commit 1a51dc6) each fails on an assertion
rather than an import.

The central check is a canary in the style of tests/test_causality.py:
rewrite every price from the first out-of-sample day onward and require
the fitted model to come out bit-identical. A model that trained on any
label or feature from that day on would change.
"""

import numpy as np
import pandas as pd
import pytest

import qbt.adaptive as adaptive
import qbt.ml as ml
from qbt.features import build_labels
from qbt.ml import _prepare_data, ml_direction_signal, model_report, train_ml_model
from qbt.regime import detect_regimes
from qbt.walk_forward import evaluate_out_of_sample

N_DAYS = 1500


def _prices(n: int = N_DAYS, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2012-01-02", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n))
    volume = rng.integers(50_000_000, 150_000_000, n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=index)


def _rewrite_from(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """
    Same history before `date`; from `date` on, every daily return is
    negated and volume halved. Negating flips the direction of every
    label that touches `date` or later, including the one ending on it,
    so a model that used any of them must change.
    """
    out = df.copy()
    later = out.index >= date
    returns = out["Close"].pct_change().to_numpy()
    start = out.loc[~later, "Close"].iloc[-1]
    out.loc[later, "Close"] = start * np.cumprod(1 - returns[later])
    out.loc[later, "Volume"] = out.loc[later, "Volume"] / 2
    return out


def _coefficients(model) -> np.ndarray:
    clf = model.named_steps["logisticregression"]
    return np.concatenate([clf.coef_.ravel(), clf.intercept_])


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return _prices()


@pytest.fixture(scope="module")
def split(prices) -> pd.Timestamp:
    return pd.Timestamp(train_ml_model(prices)[2])


# 1. The full-period signal predicted its own training period.
def test_signal_is_flat_before_the_first_out_of_sample_day(prices, split):
    signal = ml_direction_signal(prices)
    assert (signal[signal.index < split] == 0).all()
    assert (signal[signal.index >= split] != 0).any()


# 1-3. Nothing the model trains on may come from the first out-of-sample day on.
def test_model_is_unchanged_when_everything_from_the_split_on_is_rewritten(prices, split):
    model, _cols, _split, _data = train_ml_model(prices)
    rewritten, _cols, split_again, _data = train_ml_model(_rewrite_from(prices, split))
    assert split_again == split
    np.testing.assert_array_equal(_coefficients(rewritten), _coefficients(model))


# 2. evaluate_out_of_sample split on all rows, the model on post-warm-up rows.
def test_evaluate_out_of_sample_splits_where_the_model_does(prices, split):
    evaluated = evaluate_out_of_sample(prices, ml_direction_signal)
    assert pd.Timestamp(evaluated["split_date"]) == split


# 3. rolling_walk_forward's per-fold train_frac must put the model's split on
#    the fold's first test day.
@pytest.mark.parametrize(("train_days", "test_days"), [(756, 126), (504, 63), (300, 50)])
def test_fold_override_splits_on_the_first_test_day(prices, train_days, test_days):
    fold = prices.iloc[:train_days + test_days]
    _model, _cols, fold_split, _data = train_ml_model(
        fold, train_frac=train_days / (train_days + test_days))
    assert fold_split == fold.index[train_days]


# 4. ml_regime_conditional fitted its regime model past the ML split.
def test_regime_model_is_fitted_only_before_the_ml_split(prices, split, monkeypatch):
    fitted = []
    original = adaptive.detect_regimes

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        fitted.append(result)
        return result

    monkeypatch.setattr(adaptive, "detect_regimes", recording)
    adaptive.ml_regime_conditional(prices)
    (result,) = fitted
    assert pd.Timestamp(result.fit_end) < split


# 4. ...and called a single fit "causal" although it labels its own fit window.
def test_single_fit_regimes_are_not_called_causal(prices):
    result = detect_regimes(prices, method="hmm", fit_frac=0.7)
    labelled_in_fit = (result.labels != -1) & (result.labels.index <= result.fit_end)
    assert labelled_in_fit.any()
    assert result.causal is False
    assert pd.Timestamp(result.causal_from) > pd.Timestamp(result.fit_end)


# 5. The last row's label (no next close yet) was 0 and was kept.
def test_last_label_is_unknown_and_never_used(prices):
    assert np.isnan(build_labels(prices).iloc[-1])
    data, _cols = _prepare_data(prices)
    assert prices.index[-1] not in data.index


# 6. model_report scored the split day in both train and test accuracy.
def test_model_report_train_and_test_rows_do_not_overlap(prices, monkeypatch):
    scored = []
    original = ml.accuracy_score

    def recording(y_true, y_pred, *args, **kwargs):
        if hasattr(y_true, "index"):
            scored.append(pd.Index(y_true.index))
        return original(y_true, y_pred, *args, **kwargs)

    monkeypatch.setattr(ml, "accuracy_score", recording)
    model_report(prices)
    train_rows, test_rows = scored[0], scored[1]
    assert len(train_rows.intersection(test_rows)) == 0
    assert train_rows.max() < test_rows.min()


# 7. Regime dummy columns were chosen from all rows, test rows included.
def test_regime_dummy_columns_come_from_training_rows_only(prices, split):
    probe = pd.Series(np.where(prices.index < split, 0, 1), index=prices.index)
    _model, cols, _split, data = train_ml_model(prices, regimes=probe)
    assert "regime_0" in cols and "regime_1" not in cols
    # A regime first seen out of sample encodes as all zeros, like an unknown day.
    assert (data.loc[data.index >= split, "regime_0"] == 0).all()


# 4. Part of a truthful causal flag: which cluster is "regime 0" must not
#    depend on days after the fit window, or a post-fit label would.
def test_regime_numbering_uses_the_fit_window_only():
    from qbt.regime import _order_and_name

    index = pd.bdate_range("2020-01-01", periods=6)
    labels = pd.Series([0, 0, 1, 1, 1, 1], index=index)
    features = pd.DataFrame({"vol_20d": [0.3, 0.3, 0.1, 0.1, 0.9, 0.9],
                             "trend_60d": 0.0, "vol_percentile": 0.5}, index=index)
    # In the fit window (first four days) cluster 1 is the calmer one. Over
    # all six days its mean volatility (0.5) is above cluster 0's (0.3).
    renumbered, _names, _remap = _order_and_name(labels, features, fit_index=index[:4])
    assert renumbered.tolist() == [1, 1, 0, 0, 0, 0]
