"""
Baselines any monthly model has to beat, each predicting only from what
its fold is allowed to see.

- majority_class: the most common label in the fold's (purged) training
  set, predicted for every test sample. Ties go to the smaller label.
- previous_label: last month's label, i.e. "last month's sign" for
  direction and "the current regime persists" for regime. The label of
  the month ending at t is known at t's close, so it needs no training.
- buy_and_hold: always long; the trading comparison for direction.
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
