"""
ML-driven direction strategy: trains a classifier on historical features
to predict next-day direction, then converts predictions into the same
long/flat signal interface every other strategy in this project uses.

Out-of-sample only. The model is fitted once, on the rows before the
split date (qbt.timesplit.split_date(df.index, train_frac): the first
`train_frac` of ALL rows, warm-up included), and predicts only from the
split date on. The training period gets no prediction: its signal is
flat (0), exactly like warm-up, so a full-period backtest sits in cash
with zero return and no cost there, and its first trade is the entry on
the first out-of-sample prediction. Full-period metrics therefore
describe "cash, then the model": Sharpe, volatility and annualized
return average over the flat period as well. For the model's own
performance read evaluate_out_of_sample()'s out_sample half (which
splits on the same date) or rolling_walk_forward(refit_per_fold=True).

Each training row's label is the next day's direction, so it is known
only at the next day's close. The last row before the split date has a
label that ends ON the split date, the same close at which the first
out-of-sample prediction is made; it is purged from training, so every
training label was known strictly before the first prediction.

Passing a `regimes` label series switches on regime awareness:
regime_mode="feature" one-hot encodes the regime as extra input columns
into a single model, with the columns chosen from the regimes seen in
the training rows only (a regime first seen out of sample encodes as all
zeros, like an unknown day). regime_mode="conditional" trains a separate
model per regime, which increases flexibility but also overfitting risk
since each model sees a fraction of the data at the same parameter
count; regimes with too little history fall back to a model fitted on
everything, and model_report() reports which did. The regime labels
must themselves be out of sample from the split date on;
qbt.adaptive.ml_regime_conditional arranges that.

Monthly SPY prediction (next-month direction and regime) lives in the
submodules, separate from this daily strategy: qbt.ml.features and
qbt.ml.labels build the point-in-time dataset, qbt.ml.splits the purged
walk-forward folds, qbt.ml.baselines and qbt.ml.evaluate the baselines
and their out-of-sample scoring.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qbt.features import build_features, build_labels
from qbt.timesplit import split_index

MODEL_TYPES = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
}

REGIME_MODES = ("feature", "conditional")

# Below this many training rows, a per-regime model is not a model, it's
# a memorized sample. Those regimes borrow the global model instead.
MIN_REGIME_TRAIN_ROWS = 150


def _regime_dummies(regimes: pd.Series, index: pd.Index,
                    train_index: pd.Index | None = None) -> pd.DataFrame:
    """
    One-hot encodes regime labels on `index`. The columns are the
    regimes present in `train_index` (default: all of `index`); a row
    whose regime has no column, or is unknown (-1), is all zeros rather
    than being lumped in with regime 0.
    """
    aligned = regimes.reindex(index).fillna(-1).astype(int)
    source = aligned if train_index is None else aligned.loc[train_index]
    present = sorted(r for r in source.unique() if r >= 0)
    dummies = pd.DataFrame(0.0, index=index, columns=[f"regime_{r}" for r in present])
    for regime_id in present:
        dummies.loc[aligned == regime_id, f"regime_{regime_id}"] = 1.0
    return dummies


def _prepare_data(df: pd.DataFrame, regimes: pd.Series | None = None):
    """
    Builds features + labels and drops any row with a NaN in either
    (warm-up rows, and the last row, whose next-day label doesn't exist
    yet). When `regimes` is supplied the regime label is carried along
    in a "regime" column so downstream code can split by it; it is not a
    feature here (train_ml_model() encodes it after the split).
    """
    features = build_features(df)
    data = features.copy()
    data["label"] = build_labels(df)
    if regimes is not None:
        data["regime"] = regimes.reindex(data.index).fillna(-1).astype(int)

    feature_cols = features.columns.tolist()
    data = data.dropna(subset=feature_cols + ["label"])
    data["label"] = data["label"].astype(int)
    return data, feature_cols


def _split(df: pd.DataFrame, data: pd.DataFrame,
           train_frac: float) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    """
    (split_date, train_mask, test_mask) over the rows of `data`.
    split_date is the first out-of-sample day, shared with
    evaluate_out_of_sample(). Training rows are those whose label (the
    next day's close) is known strictly before split_date, which purges
    the one row whose label ends on it. Test rows are split_date onward.
    """
    position = split_index(len(df), train_frac)
    split_date = pd.Timestamp(df.index[position])
    label_end = pd.Series(df.index[1:], index=df.index[:-1]).reindex(data.index)
    train_mask = (label_end < split_date).to_numpy()
    test_mask = (data.index >= split_date)
    if not train_mask.any():
        raise ValueError(
            f"No training rows before {split_date.date()}: train_frac={train_frac} leaves "
            "nothing after the feature warm-up. Raise train_frac or widen the date range."
        )
    if not test_mask.any():
        raise ValueError(f"No out-of-sample rows from {split_date.date()} on.")
    return split_date, train_mask, np.asarray(test_mask)


def train_ml_model(df: pd.DataFrame, train_frac: float = 0.7, model_type: str = "logistic",
                   regimes: pd.Series | None = None):
    """
    Fits a classifier on the training rows (see _split()). Returns
    (model, feature_cols, split_date, data): split_date is the first
    out-of-sample day, `data` every valid row with its features.

    With `regimes`, the regime label is one-hot encoded into the feature
    matrix, with columns chosen from the training rows only.
    """
    data, feature_cols = _prepare_data(df, regimes)
    split_date, train_mask, _test_mask = _split(df, data, train_frac)

    if regimes is not None:
        dummies = _regime_dummies(regimes, data.index, train_index=data.index[train_mask])
        data = pd.concat([data, dummies], axis=1)
        feature_cols = feature_cols + dummies.columns.tolist()

    train_data = data.iloc[np.flatnonzero(train_mask)]
    model = MODEL_TYPES[model_type]()
    model.fit(train_data[feature_cols], train_data["label"])

    return model, feature_cols, split_date, data


def train_regime_conditional_models(df: pd.DataFrame, train_frac: float = 0.7,
                                    model_type: str = "logistic", regimes: pd.Series | None = None,
                                    min_train_rows: int = MIN_REGIME_TRAIN_ROWS):
    """
    One model per regime, each fitted only on training rows in that
    regime, plus a global fallback for regimes without enough history to
    justify their own. Splitting the data this way reduces the rows
    available per model at the same parameter count, which increases
    overfitting risk.
    """
    if regimes is None:
        raise ValueError("train_regime_conditional_models() needs a regime label series.")

    data, feature_cols = _prepare_data(df, regimes)
    split_date, train_mask, _test_mask = _split(df, data, train_frac)
    train_data = data.iloc[np.flatnonzero(train_mask)]

    fallback = MODEL_TYPES[model_type]()
    fallback.fit(train_data[feature_cols], train_data["label"])

    models, train_counts = {}, {}
    for regime_id in sorted(r for r in train_data["regime"].unique() if r >= 0):
        subset = train_data[train_data["regime"] == regime_id]
        train_counts[int(regime_id)] = len(subset)
        # Needs enough rows AND both classes present; a regime where every
        # training day was an up day can only teach "always long".
        if len(subset) >= min_train_rows and subset["label"].nunique() > 1:
            model = MODEL_TYPES[model_type]()
            model.fit(subset[feature_cols], subset["label"])
            models[int(regime_id)] = model

    return models, fallback, feature_cols, split_date, data, train_counts


def _conditional_predict(
    models: dict, fallback, data: pd.DataFrame, feature_cols: list
) -> pd.Series:
    """Routes each row to its regime's model, falling back where there isn't one."""
    predictions = pd.Series(index=data.index, dtype=float)
    for regime_id, model in models.items():
        rows = data["regime"] == regime_id
        if rows.any():
            predictions[rows] = model.predict(data.loc[rows, feature_cols])

    missing = predictions.isna()
    if missing.any():
        predictions[missing] = fallback.predict(data.loc[missing, feature_cols])
    return predictions.astype(int)


def ml_direction_predictions(df: pd.DataFrame, train_frac: float = 0.7,
                             model_type: str = "logistic", regimes: pd.Series | None = None,
                             regime_mode: str = "feature") -> pd.Series:
    """
    The model's out-of-sample predictions (1 up, 0 down) on df.index:
    NaN on every day before the split date and on warm-up rows, which
    get no prediction at all.
    """
    if regimes is not None and regime_mode not in REGIME_MODES:
        raise ValueError(f"regime_mode must be one of {REGIME_MODES}, got {regime_mode!r}")

    if regimes is not None and regime_mode == "conditional":
        models, fallback, feature_cols, split_date, data, _counts = (
            train_regime_conditional_models(df, train_frac, model_type, regimes))
        oos = data[data.index >= split_date]
        predictions = _conditional_predict(models, fallback, oos, feature_cols)
    else:
        model, feature_cols, split_date, data = train_ml_model(df, train_frac, model_type, regimes)
        oos = data[data.index >= split_date]
        predictions = pd.Series(model.predict(oos[feature_cols]), index=oos.index)

    return predictions.astype(float).reindex(df.index)


def ml_direction_signal(df: pd.DataFrame, train_frac: float = 0.7, model_type: str = "logistic",
                        regimes: pd.Series | None = None,
                        regime_mode: str = "feature") -> pd.Series:
    """
    Long (1) or flat (0) on each day: the out-of-sample prediction from
    the split date on, flat before it (training period) and on warm-up
    rows. See the module docstring for what that means for full-period
    metrics.
    """
    predictions = ml_direction_predictions(df, train_frac, model_type, regimes, regime_mode)
    return predictions.fillna(0).astype(int)


def _feature_importance(model, feature_cols: list, model_type: str) -> dict:
    if model_type == "logistic":
        clf = model.named_steps["logisticregression"]
        return dict(zip(feature_cols, clf.coef_[0], strict=True))
    return dict(zip(feature_cols, model.feature_importances_, strict=True))


def model_report(df: pd.DataFrame, train_frac: float = 0.7, model_type: str = "logistic",
                 regimes: pd.Series | None = None, regime_mode: str = "feature") -> dict:
    """
    Diagnostics on the model itself (accuracy, feature importance) --
    separate from the trading-strategy metrics in analytics.py. Train
    accuracy is scored on exactly the rows the model was fitted on, test
    accuracy on the out-of-sample rows; the two never share a row. With
    `regimes` supplied, the report also includes a per-regime accuracy
    breakdown.
    """
    conditional = regimes is not None and regime_mode == "conditional"

    if conditional:
        (models, fallback, feature_cols, split_date,
         data, train_counts) = train_regime_conditional_models(
            df, train_frac, model_type, regimes
        )
    else:
        model, feature_cols, split_date, data = train_ml_model(df, train_frac, model_type, regimes)
        train_counts = {}

    _split_date, train_mask, test_mask = _split(df, data, train_frac)
    train_data = data.iloc[np.flatnonzero(train_mask)]
    test_data = data.iloc[np.flatnonzero(test_mask)]

    if conditional:
        train_pred = _conditional_predict(models, fallback, train_data, feature_cols)
        test_pred = _conditional_predict(models, fallback, test_data, feature_cols)
        importance = _feature_importance(fallback, feature_cols, model_type)
    else:
        train_pred = model.predict(train_data[feature_cols])
        test_pred = model.predict(test_data[feature_cols])
        importance = _feature_importance(model, feature_cols, model_type)

    test_labels = test_data["label"]
    report = {
        "split_date": str(split_date.date()),
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "train_accuracy": accuracy_score(train_data["label"], train_pred),
        "test_accuracy": accuracy_score(test_labels, test_pred),
        "test_confusion_matrix": confusion_matrix(test_labels, test_pred).tolist(),
        "feature_importance": importance,
        # Accuracy a model predicting only the majority class would get;
        # test accuracy at or below this means the model added nothing.
        "test_base_rate": float(max(test_labels.mean(), 1 - test_labels.mean())),
        "regime_mode": regime_mode if regimes is not None else None,
    }

    if regimes is not None:
        test_regimes = test_data["regime"]
        rows = []
        for regime_id in sorted(r for r in test_regimes.unique() if r >= 0):
            mask = (test_regimes == regime_id).to_numpy()
            if mask.sum() < 20:
                continue
            actual = test_labels.to_numpy()[mask]
            test_pred_array: np.ndarray = np.asarray(test_pred)
            predicted = test_pred_array[mask]
            rows.append({
                "regime": int(regime_id),
                "test_days": int(mask.sum()),
                "test_accuracy": accuracy_score(actual, predicted),
                "base_rate": float(max(actual.mean(), 1 - actual.mean())),
                "train_rows": train_counts.get(int(regime_id)),
                "own_model": train_counts.get(int(regime_id), 0) >= MIN_REGIME_TRAIN_ROWS,
            })
        report["by_regime"] = rows

    return report
