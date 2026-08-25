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
"""

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


def _prepare_data(df: pd.DataFrame):
    """Builds features + labels and drops any row with a NaN in either (warm-up rows)."""
    features = build_features(df)
    labels = build_labels(df)

    data = features.copy()
    data["label"] = labels
    data = data.dropna()
    return data, features.columns.tolist()


def train_ml_model(df: pd.DataFrame, train_frac: float = 0.7, model_type: str = "logistic"):
    """
    Fits a classifier on the first `train_frac` of valid rows,
    chronologically -- never on rows after the split, so the model never
    sees the "future" relative to any row it'll later be tested on.
    """
    data, feature_cols = _prepare_data(df)

    split_idx = int(len(data) * train_frac)
    split_date = data.index[split_idx]
    train_data = data.iloc[:split_idx]

    model = MODEL_TYPES[model_type]()
    model.fit(train_data[feature_cols], train_data["label"])

    return model, feature_cols, split_date, data


def ml_direction_signal(df: pd.DataFrame, train_frac: float = 0.7, model_type: str = "logistic") -> pd.Series:
    """
    Trains once on the first `train_frac` of history, then predicts
    direction for every valid row (both the training period and the
    held-out period). Rows without enough history for features default
    to flat (0) -- same warm-up convention as every other strategy here.
    """
    model, feature_cols, _split_date, data = train_ml_model(df, train_frac, model_type)

    predictions = pd.Series(model.predict(data[feature_cols]), index=data.index)

    signal = pd.Series(0, index=df.index)
    signal.loc[predictions.index] = predictions
    return signal.astype(int)


def model_report(df: pd.DataFrame, train_frac: float = 0.7, model_type: str = "logistic") -> dict:
    """
    Diagnostics on the MODEL itself (accuracy, feature importance) --
    separate from the trading-strategy metrics in analytics.py, which
    should be read from evaluate_out_of_sample(), not from here.
    """
    model, feature_cols, split_date, data = train_ml_model(df, train_frac, model_type)

    train_data = data.loc[:split_date]
    test_data = data.loc[split_date:]

    train_pred = model.predict(train_data[feature_cols])
    test_pred = model.predict(test_data[feature_cols])

    if model_type == "logistic":
        clf = model.named_steps["logisticregression"]
        importance = dict(zip(feature_cols, clf.coef_[0]))
    else:
        importance = dict(zip(feature_cols, model.feature_importances_))

    return {
        "split_date": str(split_date.date()),
        "train_accuracy": accuracy_score(train_data["label"], train_pred),
        "test_accuracy": accuracy_score(test_data["label"], test_pred),
        "test_confusion_matrix": confusion_matrix(test_data["label"], test_pred).tolist(),
        "feature_importance": importance,
    }
