"""
ML-driven direction strategy: trains a classifier on historical features
to predict next-day direction, then converts predictions into the same
long/flat signal interface every other strategy in this project uses.

IMPORTANT (overfitting risk): predictions on the portion of data the
model was TRAINED on are in-sample and will look better than they
honestly should. This strategy's train_frac defaults to 0.7, matching
walk_forward.py's default split, specifically so that running this
strategy through evaluate_out_of_sample() gives an honest read on the
out-of-sample 30% -- that's the only part of this strategy's performance
that should be trusted or quoted.

REGIME-AWARE MODELS (added alongside regime.py). Passing a `regimes`
label series switches on one of two ways of telling the model what kind
of market it is in:

    regime_mode="feature"      one model, with the regime one-hot encoded
                               as extra input columns. The model can
                               learn "in a crisis, ignore momentum" if
                               the data supports it. Costs a few
                               parameters; keeps all the training rows.

    regime_mode="conditional"  a SEPARATE model per regime, each trained
                               only on days in that regime. Maximum
                               flexibility -- and a direct route to
                               overfitting, because a 3-regime split
                               leaves each model roughly a third of the
                               data while the parameter count stays the
                               same. Regimes with too little history fall
                               back to a model fitted on everything;
                               model_report() tells you which did.

Which is better is an empirical question, and a genuinely good exercise.
The honest answer on index data is usually "neither beats plain logistic
regression", and finding that out yourself is worth more than being told.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features, build_labels

MODEL_TYPES = {
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
}

REGIME_MODES = ("feature", "conditional")

# Below this many training rows, a per-regime model is not a model, it's
# a memorized sample. Those regimes borrow the global model instead.
MIN_REGIME_TRAIN_ROWS = 150


def _regime_dummies(regimes: pd.Series, index: pd.Index) -> pd.DataFrame:
    """
    One-hot encodes regime labels. Unknown (-1) rows become all-zeros,
    which is the honest encoding of "we don't know what regime this is"
    rather than silently lumping them in with regime 0.
    """
    aligned = regimes.reindex(index).fillna(-1).astype(int)
    present = sorted(r for r in aligned.unique() if r >= 0)
    dummies = pd.DataFrame(0.0, index=index, columns=[f"regime_{r}" for r in present])
    for regime_id in present:
        dummies.loc[aligned == regime_id, f"regime_{regime_id}"] = 1.0
    return dummies


def _prepare_data(df: pd.DataFrame, regimes: pd.Series = None, encode_regime: bool = False):
    """
    Builds features + labels and drops any row with a NaN in either
    (warm-up rows). When `regimes` is supplied the regime label is
    carried along so downstream code can split by it; it only enters the
    FEATURE list when encode_regime is True.
    """
    features = build_features(df)
    labels = build_labels(df)

    if encode_regime and regimes is not None:
        features = pd.concat([features, _regime_dummies(regimes, features.index)], axis=1)

    data = features.copy()
    data["label"] = labels
    if regimes is not None:
        data["regime"] = regimes.reindex(data.index).fillna(-1).astype(int)

    feature_cols = features.columns.tolist()
    data = data.dropna(subset=feature_cols + ["label"])
    return data, feature_cols


def train_ml_model(df: pd.DataFrame, train_frac: float = 0.7, model_type: str = "logistic",
                   regimes: pd.Series = None):
    """
    Fits a classifier on the first `train_frac` of valid rows,
    chronologically -- never on rows after the split, so the model never
    sees the "future" relative to any row it'll later be tested on.

    With `regimes`, the regime label is one-hot encoded into the feature
    matrix. The return signature is unchanged either way.
    """
    data, feature_cols = _prepare_data(df, regimes, encode_regime=regimes is not None)

    split_idx = int(len(data) * train_frac)
    split_date = data.index[split_idx]
    train_data = data.iloc[:split_idx]

    model = MODEL_TYPES[model_type]()
    model.fit(train_data[feature_cols], train_data["label"])

    return model, feature_cols, split_date, data


def train_regime_conditional_models(df: pd.DataFrame, train_frac: float = 0.7,
                                    model_type: str = "logistic", regimes: pd.Series = None,
                                    min_train_rows: int = MIN_REGIME_TRAIN_ROWS):
    """
    One model per regime, each fitted only on training-period days in
    that regime, plus a global fallback for regimes without enough
    history to justify their own.

    The data-splitting cost is real and worth stating plainly: with 2,500
    rows, a 70% train split and 3 regimes, each model gets roughly 580
    rows to fit a dozen features on. That is exactly how a "smarter"
    model ends up generalizing worse than the dumb one it replaced.
    """
    if regimes is None:
        raise ValueError("train_regime_conditional_models() needs a regime label series.")

    data, feature_cols = _prepare_data(df, regimes, encode_regime=False)
    split_idx = int(len(data) * train_frac)
    split_date = data.index[split_idx]
    train_data = data.iloc[:split_idx]

    fallback = MODEL_TYPES[model_type]()
    fallback.fit(train_data[feature_cols], train_data["label"])

    models, train_counts = {}, {}
    for regime_id in sorted(r for r in train_data["regime"].unique() if r >= 0):
        subset = train_data[train_data["regime"] == regime_id]
        train_counts[int(regime_id)] = len(subset)
        # A model needs enough rows AND both classes present -- a regime
        # where every training day was an up day can't teach a classifier
        # anything except "always long".
        if len(subset) >= min_train_rows and subset["label"].nunique() > 1:
            model = MODEL_TYPES[model_type]()
            model.fit(subset[feature_cols], subset["label"])
            models[int(regime_id)] = model

    return models, fallback, feature_cols, split_date, data, train_counts


def _conditional_predict(models: dict, fallback, data: pd.DataFrame, feature_cols: list) -> pd.Series:
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


def ml_direction_signal(df: pd.DataFrame, train_frac: float = 0.7, model_type: str = "logistic",
                        regimes: pd.Series = None, regime_mode: str = "feature") -> pd.Series:
    """
    Trains once on the first `train_frac` of history, then predicts
    direction for every valid row (both the training period and the
    held-out period). Rows without enough history for features default
    to flat (0) -- same warm-up convention as every other strategy here.

    With regimes=None this behaves exactly as it always has, which is why
    it stays registered in STRATEGIES unchanged.
    """
    if regimes is not None and regime_mode not in REGIME_MODES:
        raise ValueError(f"regime_mode must be one of {REGIME_MODES}, got {regime_mode!r}")

    if regimes is not None and regime_mode == "conditional":
        models, fallback, feature_cols, _split, data, _counts = train_regime_conditional_models(
            df, train_frac, model_type, regimes
        )
        predictions = _conditional_predict(models, fallback, data, feature_cols)
    else:
        model, feature_cols, _split_date, data = train_ml_model(df, train_frac, model_type, regimes)
        predictions = pd.Series(model.predict(data[feature_cols]), index=data.index)

    signal = pd.Series(0, index=df.index)
    signal.loc[predictions.index] = predictions
    return signal.astype(int)


def _feature_importance(model, feature_cols: list, model_type: str) -> dict:
    if model_type == "logistic":
        clf = model.named_steps["logisticregression"]
        return dict(zip(feature_cols, clf.coef_[0]))
    return dict(zip(feature_cols, model.feature_importances_))


def model_report(df: pd.DataFrame, train_frac: float = 0.7, model_type: str = "logistic",
                 regimes: pd.Series = None, regime_mode: str = "feature") -> dict:
    """
    Diagnostics on the MODEL itself (accuracy, feature importance) --
    separate from the trading-strategy metrics in analytics.py, which
    should be read from evaluate_out_of_sample(), not from here.

    With regimes supplied the report gains a per-regime accuracy
    breakdown, which is often the most informative table in the whole
    dashboard: it shows whether an unimpressive 51% overall model is 56%
    in one regime and 46% in another (worth conditioning on) or 51%
    everywhere (nothing there to condition on).
    """
    conditional = regimes is not None and regime_mode == "conditional"

    if conditional:
        models, fallback, feature_cols, split_date, data, train_counts = train_regime_conditional_models(
            df, train_frac, model_type, regimes
        )
        train_data, test_data = data.loc[:split_date], data.loc[split_date:]
        train_pred = _conditional_predict(models, fallback, train_data, feature_cols)
        test_pred = _conditional_predict(models, fallback, test_data, feature_cols)
        importance = _feature_importance(fallback, feature_cols, model_type)
    else:
        model, feature_cols, split_date, data = train_ml_model(df, train_frac, model_type, regimes)
        train_data, test_data = data.loc[:split_date], data.loc[split_date:]
        train_pred = model.predict(train_data[feature_cols])
        test_pred = model.predict(test_data[feature_cols])
        importance = _feature_importance(model, feature_cols, model_type)
        train_counts = {}

    test_labels = test_data["label"]
    report = {
        "split_date": str(split_date.date()),
        "train_accuracy": accuracy_score(train_data["label"], train_pred),
        "test_accuracy": accuracy_score(test_labels, test_pred),
        "test_confusion_matrix": confusion_matrix(test_labels, test_pred).tolist(),
        "feature_importance": importance,
        # Accuracy a model that always predicts the majority class would
        # get. Test accuracy at or below this means the model added
        # nothing at all -- and on daily equity data the base rate is
        # around 53%, which is why "54% accuracy!" is not the win it looks.
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
            predicted = np.asarray(test_pred)[mask]
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
