"""
C2 models for monthly prediction: logistic regression (scikit-learn) and
LightGBM, trained inside the C1 purged walk-forward folds with their
hyperparameters chosen only on inner folds.

Nesting, per outer fold (qbt.ml.splits.walk_forward_splits):
- the outer fold's training set is the only data the fold may use;
- every configuration in the model's grid is scored on purged k-fold
  splits of that training set (qbt.ml.splits.purged_kfold_splits, which
  is meant for exactly this: model selection inside a training window),
  by pooled log loss over the inner test blocks;
- the configuration with the lowest inner log loss (ties to the earlier
  grid entry) is refit on the whole outer training set and predicts the
  outer test block, which no step above has seen.

Probabilities always come back over the target's full class list
(classes_for()), so a class missing from a training set gets a column
too: it gets probability PROB_FLOOR, as does any class a model rates
below that, and rows are renormalized. Without the floor a single month
of a class the training data never showed would make log loss infinite.

Determinism: LightGBM runs with deterministic=True, force_row_wise=True,
n_jobs=1 and a fixed seed; logistic regression with lbfgs is
deterministic. LightGBM is imported lazily, so the rest of qbt doesn't
need it (it is the `ml` extra).
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from qbt.ml.metrics import log_loss
from qbt.ml.splits import EMBARGO_MONTHS, Fold, purged_kfold_splits

SEED = 0
PROB_FLOOR = 1e-4
INNER_SPLITS = 5

DIRECTION_CLASSES = (0.0, 1.0)
REGIME_CLASSES = (0.0, 1.0, 2.0, 3.0)
# Integer-coded regime features: one-hot for the linear model (an order
# 0 < 1 < 2 < 3 means nothing), left as codes for the trees.
CATEGORICAL_FEATURES = {"regime_rules": [0.0, 1.0, 2.0, 3.0], "regime_hmm": [0.0, 1.0, 2.0]}

# Both grids reach well into heavy regularization: with 60-180 training
# months, a first pass whose inner tuning kept choosing the grid's most
# regularized corner (C=0.01; 50 trees, min_child_samples=15) was extended
# one step further in that direction, using inner-fold selections only.
LOGISTIC_GRID: list[dict[str, Any]] = [{"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0)]
LIGHTGBM_GRID: list[dict[str, Any]] = [
    {"num_leaves": leaves, "min_child_samples": child, "n_estimators": trees}
    for leaves, child, trees in product((3, 7), (5, 15, 30), (25, 50, 200))
]
LIGHTGBM_FIXED: dict[str, Any] = {
    "learning_rate": 0.05, "reg_lambda": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
    "random_state": SEED, "deterministic": True, "force_row_wise": True, "n_jobs": 1,
    "verbose": -1,
}
MODELS = ("logistic", "lightgbm")


def classes_for(target: str) -> tuple[float, ...]:
    if target == "direction":
        return DIRECTION_CLASSES
    if target == "regime":
        return REGIME_CLASSES
    raise ValueError(f"unknown target {target!r}")


def grid_for(model: str) -> list[dict[str, Any]]:
    if model == "logistic":
        return LOGISTIC_GRID
    if model == "lightgbm":
        return LIGHTGBM_GRID
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


def config_id(model: str, params: dict[str, Any]) -> str:
    return model + "(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")"


def _estimator(model: str, params: dict[str, Any], columns: list[str]) -> Any:
    if model == "logistic":
        categorical = [c for c in columns if c in CATEGORICAL_FEATURES]
        numeric = [c for c in columns if c not in CATEGORICAL_FEATURES]
        encode = ColumnTransformer([
            ("num", StandardScaler(), [columns.index(c) for c in numeric]),
            ("cat", OneHotEncoder(categories=[CATEGORICAL_FEATURES[c] for c in categorical],
                                  handle_unknown="ignore", sparse_output=False),
             [columns.index(c) for c in categorical]),
        ])
        return Pipeline([("encode", encode),
                         ("model", LogisticRegression(C=params["C"], max_iter=10_000))])
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**LIGHTGBM_FIXED, **params)


def fit_predict_proba(model: str, params: dict[str, Any], X_train: pd.DataFrame,
                      y_train: pd.Series, X_test: pd.DataFrame,
                      classes: tuple[float, ...]) -> np.ndarray:
    """
    Fits on (X_train, y_train) and returns P(class) for every row of
    X_test over `classes`, floored at PROB_FLOOR and renormalized. A
    training set with a single class predicts that class (no model can
    be fitted).
    """
    present = np.unique(y_train.to_numpy(dtype=float))
    if not set(present) <= set(classes):
        raise ValueError(f"labels {present} outside classes {classes}")
    out = np.zeros((len(X_test), len(classes)))
    if len(present) == 1:
        out[:, classes.index(float(present[0]))] = 1.0
    else:
        estimator = _estimator(model, params, list(X_train.columns))
        # Plain arrays in and out: column names never reach either library,
        # so there is no feature-name check to drift between fit and predict.
        estimator.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))
        proba = estimator.predict_proba(X_test.to_numpy(dtype=float))
        fitted = [float(c) for c in estimator.classes_]
        for j, c in enumerate(fitted):
            out[:, classes.index(c)] = proba[:, j]
    out = np.maximum(out, PROB_FLOOR)
    return out / out.sum(axis=1, keepdims=True)


@dataclass
class TuningResult:
    params: dict[str, Any]
    trials: list[dict[str, Any]] = field(default_factory=list)  # one per configuration


def tune(model: str, X: pd.DataFrame, y: pd.Series, t0: pd.Series, t1: pd.Series,
         classes: tuple[float, ...], n_splits: int = INNER_SPLITS,
         embargo: int = EMBARGO_MONTHS,
         grid: list[dict[str, Any]] | None = None) -> TuningResult:
    """
    Scores every configuration of `grid` (default: `model`'s grid) on
    purged k-fold splits of (X, y) -- which must be one outer fold's
    training set only -- and returns the one with the lowest pooled
    inner log loss.
    """
    grid = grid_for(model) if grid is None else grid
    inner = purged_kfold_splits(t0, t1, n_splits=n_splits, embargo=embargo)
    inner = [f for f in inner if len(f.train) > 0]
    if not inner:
        raise ValueError("no inner fold has training data")
    trials = []
    for params in grid:
        y_true, probas = [], []
        for fold in inner:
            probas.append(fit_predict_proba(model, params, X.iloc[fold.train],
                                            y.iloc[fold.train], X.iloc[fold.test], classes))
            y_true.append(y.iloc[fold.test].to_numpy(dtype=float))
        yt, pr = np.concatenate(y_true), np.vstack(probas)
        trials.append({
            "config": config_id(model, params), "params": dict(params),
            "inner_folds": len(inner), "inner_n": len(yt),
            "inner_log_loss": log_loss(yt, pr, classes),
            "inner_accuracy": float(np.mean(np.asarray(classes)[pr.argmax(axis=1)] == yt)),
        })
    best = min(range(len(trials)), key=lambda i: (trials[i]["inner_log_loss"], i))
    for i, trial in enumerate(trials):
        trial["selected"] = i == best
    return TuningResult(params=dict(grid[best]), trials=trials)


@dataclass
class WalkForwardPredictions:
    model: str
    proba: pd.DataFrame          # per test month: P(class) over classes_for(target)
    fold: pd.Series              # outer fold number per test month
    selected: list[dict[str, Any]]   # per outer fold: chosen params
    trials: list[dict[str, Any]]     # per outer fold x configuration: inner scores


def walk_forward_predict(model: str, X: pd.DataFrame, y: pd.Series, t1: pd.Series,
                         folds: list[Fold], target: str,
                         grid: list[dict[str, Any]] | None = None) -> WalkForwardPredictions:
    """
    Nested walk-forward: tune on inner folds of each outer training set,
    refit, predict. `grid` overrides `model`'s grid.
    """
    classes = classes_for(target)
    t0 = pd.Series(X.index, index=X.index)
    pieces, fold_numbers, selected, trials = [], [], [], []
    for fold in folds:
        train = fold.train
        tuned = tune(model, X.iloc[train], y.iloc[train], t0.iloc[train], t1.iloc[train],
                     classes, grid=grid)
        proba = fit_predict_proba(model, tuned.params, X.iloc[train], y.iloc[train],
                                  X.iloc[fold.test], classes)
        index = X.index[fold.test]
        pieces.append(pd.DataFrame(proba, index=index, columns=list(classes)))
        fold_numbers.append(pd.Series(fold.number, index=index))
        selected.append({"fold": fold.number, "config": config_id(model, tuned.params),
                         "params": tuned.params})
        for trial in tuned.trials:
            trials.append({"outer_fold": fold.number, "n_train": len(train), **trial})
    return WalkForwardPredictions(model=model, proba=pd.concat(pieces),
                                  fold=pd.concat(fold_numbers), selected=selected,
                                  trials=trials)
