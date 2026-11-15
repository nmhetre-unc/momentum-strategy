"""
Baselines any monthly model has to beat, each predicting only from what
its fold is allowed to see.

- majority_class: the most common label in the fold's (purged) training
  set, predicted for every test sample. Ties go to the smaller label.
- previous_label: last month's label, i.e. "last month's sign" for
  direction and "the current regime persists" for regime. The label of
  the month ending at t is known at t's close, so it needs no training.
- buy_and_hold: always long; the trading comparison for direction.
- rules_regime_at_decision (regime only): the daily `rules` regime on
  the decision date, i.e. the regime_rules feature. Much of the next
  month's label is already fixed at t (see qbt.ml.labels), so this is
  the bar a regime model has to clear.
- prior_proba: the probabilistic counterpart of majority_class, the
  training set's class frequencies predicted for every test month; the
  reference for log loss and Brier score.
- conditional_prior_proba (regime): class frequencies among training
  months that had the same rules regime on their decision date, i.e.
  the probabilistic counterpart of rules_regime_at_decision; it falls
  back to prior_proba for a regime the training set never saw.
"""

import numpy as np
import pandas as pd

from qbt.ml.splits import Fold


def majority_class(y_train: pd.Series) -> float:
    if len(y_train) == 0:
        raise ValueError("majority_class needs at least one training label")
    values, counts = np.unique(y_train.to_numpy(dtype=float), return_counts=True)
    return float(values[np.argmax(counts)])


def predict_majority(y: pd.Series, fold: Fold) -> np.ndarray:
    return np.full(len(fold.test), majority_class(y.iloc[fold.train]))


def predict_previous(previous: pd.Series, fold: Fold) -> np.ndarray:
    return previous.iloc[fold.test].to_numpy(dtype=float)


def predict_buy_and_hold(fold: Fold) -> np.ndarray:
    return np.ones(len(fold.test))


def predict_rules_regime(X: pd.DataFrame, fold: Fold) -> np.ndarray:
    return X["regime_rules"].iloc[fold.test].to_numpy(dtype=float)


def prior_proba(y_train: pd.Series, n_test: int, classes: tuple[float, ...],
                floor: float = 1e-4) -> np.ndarray:
    """Training class frequencies over `classes`, floored and renormalized, per test row."""
    if len(y_train) == 0:
        raise ValueError("prior_proba needs at least one training label")
    values = y_train.to_numpy(dtype=float)
    freq = np.array([np.mean(values == c) for c in classes])
    freq = np.maximum(freq, floor)
    return np.tile(freq / freq.sum(), (n_test, 1))


def conditional_prior_proba(y_train: pd.Series, given_train: pd.Series,
                            given_test: pd.Series, classes: tuple[float, ...],
                            floor: float = 1e-4) -> np.ndarray:
    """
    For each test row, the training class frequencies among rows whose
    `given` value matches it; prior_proba() for an unseen value.
    """
    y = y_train.to_numpy(dtype=float)
    given = given_train.to_numpy(dtype=float)
    fallback = prior_proba(y_train, 1, classes, floor)[0]
    out = np.empty((len(given_test), len(classes)))
    for i, value in enumerate(given_test.to_numpy(dtype=float)):
        match = y[given == value]
        if len(match) == 0:
            out[i] = fallback
            continue
        freq = np.maximum(np.array([np.mean(match == c) for c in classes]), floor)
        out[i] = freq / freq.sum()
    return out
