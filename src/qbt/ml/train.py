"""
Training a monthly model for serving (C3), as a pure function of a price
frame and a configuration: qbt_service.train stores the result in the
model registry.

train_for_serving() does two things with the same procedure C2 used:

1. Evaluate. The C2 nested purged walk-forward (qbt.ml.models.
   walk_forward_predict) over the frame gives out-of-sample predictions,
   scored against every C1/C2 baseline on the same months, and against
   the target's bar (qbt.ml.evaluate.BAR) with a paired stationary-
   bootstrap interval and an exact McNemar test. These are the numbers
   the registry and every prediction carry.
2. Fit. The same tuning (purged k-fold, lowest log loss) over every
   labelled month, then a refit on all of them. The out-of-sample
   numbers above estimate how this procedure does on months it hasn't
   seen; the served model is the procedure applied once more, to all of
   them.

It also computes the serving input: the features of the latest complete
month in the frame (its last decision date, whose label isn't known yet,
so the model never trained on it), with a SHA-256 over their exact bits.
Those are what POST /predict scores. The HMM features take tens of
seconds to build, so they are computed here once, not per request.

Deterministic: fixed seeds, deterministic LightGBM, a seeded bootstrap.
The same frame and configuration give a byte-identical artifact on the
same library versions (LightGBM writes its text format deterministically
and has no timestamps in it).
"""

import hashlib
import importlib.metadata
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from qbt.ml.evaluate import BAR, build_dataset, mcnemar_exact
from qbt.ml.evaluate import baseline_predictions as baseline_table
from qbt.ml.features import month_end_dates, monthly_features
from qbt.ml.metrics import auc, brier, log_loss, paired_mean_bootstrap, predicted_class
from qbt.ml.models import (
    INNER_SPLITS,
    LIGHTGBM_FIXED,
    PROB_FLOOR,
    SEED,
    classes_for,
    fit,
    grid_for,
    to_artifact,
    tune,
    walk_forward_predict,
)
from qbt.ml.splits import EMBARGO_MONTHS, walk_forward_splits
from qbt.regime import RULES_REGIME_NAMES

FEATURE_SETS = {"base": False, "base+hmm": True}
SERVED_MODELS = ("lightgbm",)
CLASS_NAMES = {
    "direction": {0.0: "down or flat", 1.0: "up"},
    "regime": {float(k): v for k, v in RULES_REGIME_NAMES.items()},
}
PREDICTS = {
    "direction": "whether the close at the next month end is above the close on as_of",
    "regime": "the most common daily `rules` regime over the trading days after as_of up to "
              "and including the next month end",
}
BAR_DESCRIPTIONS = {
    "majority_class": "always predict the most common class of the training months "
                      "(for direction: always up, i.e. buy-and-hold)",
    "rules_regime_lookup": "the most frequent next-month regime among training months that "
                           "had the same rules regime on their decision date",
}
FEATURES_HASH_VERSION = b"qbt-features-v1\0"


@dataclass(frozen=True)
class TrainingConfig:
    target: str = "regime"
    feature_set: str = "base+hmm"
    model: str = "lightgbm"
    seed: int = SEED
    # The C2 evaluation (scripts/ml_c2_report.py); tests shrink them.
    min_train: int = 60
    test_size: int = 24
    embargo: int = EMBARGO_MONTHS
    hmm_kwargs: dict[str, Any] | None = None
    n_boot: int = 2000
    mean_block: float = 6.0

    def __post_init__(self) -> None:
        if self.target not in CLASS_NAMES:
            raise ValueError(f"target must be one of {list(CLASS_NAMES)}, got {self.target!r}")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}, "
                             f"got {self.feature_set!r}")
        if self.model not in SERVED_MODELS:
            raise ValueError(f"only {SERVED_MODELS} can be served, got {self.model!r}")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")


@dataclass(frozen=True)
class TrainedModel:
    config: TrainingConfig
    artifact: bytes
    artifact_sha256: bytes
    features: list[str]
    classes: list[float]
    hyperparameters: dict[str, Any]
    train_first_decision: date
    train_last_decision: date
    train_last_label_end: date
    n_train: int
    metrics: dict[str, Any]
    serving_as_of: date
    serving_features: dict[str, float]
    serving_features_sha256: bytes
    library_versions: dict[str, str] = field(default_factory=dict)


def features_hash(as_of: date, row: pd.Series) -> bytes:
    """SHA-256 over the decision date, the column names and every value's exact bits."""
    h = hashlib.sha256(FEATURES_HASH_VERSION)
    h.update(as_of.isoformat().encode() + b"\0")
    for name in row.index:
        h.update(str(name).encode() + b"\0")
    h.update(row.to_numpy(dtype="<f8").tobytes())
    return h.digest()


def serving_features(df: pd.DataFrame, columns: list[str], include_hmm: bool,
                     hmm_kwargs: dict[str, Any] | None) -> tuple[date, pd.Series]:
    """The latest complete month's decision date in `df` and its feature row."""
    dates = month_end_dates(pd.DatetimeIndex(df.index))
    if len(dates) == 0:
        raise ValueError("the prices contain no complete month")
    as_of = dates[-1]
    table = monthly_features(df, include_hmm=include_hmm, hmm_kwargs=hmm_kwargs)
    row = pd.Series(table.loc[as_of, columns].to_numpy(dtype=float), index=columns)
    missing = [name for name, value in row.items() if np.isnan(value)]
    if missing:
        raise ValueError(f"features on {as_of.date()} are incomplete: {missing}")
    return as_of.date(), row


def library_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name)
            for name in ("qbt", "lightgbm", "numpy", "pandas", "scikit-learn", "scipy")}


def _evaluate(ds: Any, config: TrainingConfig) -> dict[str, Any]:
    classes = classes_for(config.target)
    folds = walk_forward_splits(ds.t0, ds.t1, min_train=config.min_train,
                                test_size=config.test_size, embargo=config.embargo)
    wf = walk_forward_predict(config.model, ds.X, ds.y, ds.t1, folds, config.target,
                              seed=config.seed)
    base = baseline_table(ds, folds, PROB_FLOOR)
    y = base["y"].to_numpy()
    proba = wf.proba.loc[base.index].to_numpy()
    correct = predicted_class(proba, classes) == y
    baselines = [c for c in base.columns if c not in ("fold", "y")]
    hits = {b: base[b].to_numpy() == y for b in baselines}
    boot = paired_mean_bootstrap({"model": correct, **hits}, "model", n_boot=config.n_boot,
                                 mean_block=config.mean_block)
    bar = BAR[config.target]
    vs_bar = boot["vs"][bar]
    mcnemar_p = mcnemar_exact(correct, hits[bar])
    beats = bool(vs_bar["difference_ci"][0] > 0 and mcnemar_p < 0.05)
    value, n_classes = auc(y, proba, classes)
    return {
        "evaluation": "nested purged walk-forward on this snapshot (the C2 procedure): "
                      "hyperparameters tuned on inner purged folds of each outer training set "
                      "only; every month below is predicted by a model that never saw it",
        "walk_forward": {"min_train": config.min_train, "test_size": config.test_size,
                         "embargo": config.embargo, "inner_splits": INNER_SPLITS,
                         "n_folds": len(folds)},
        "out_of_sample": {
            "n_months": len(y),
            "first_test_decision": base.index[0].date().isoformat(),
            "last_test_decision": base.index[-1].date().isoformat(),
            "accuracy": boot["mean"], "accuracy_ci": boot["ci"],
            "auc": None if np.isnan(value) else value, "auc_classes_averaged": n_classes,
            "log_loss": log_loss(y, proba, classes), "brier": brier(y, proba, classes),
        },
        "baselines": {b: boot["vs"][b]["other_mean"] for b in baselines},
        "bar": {
            "name": bar, "description": BAR_DESCRIPTIONS[bar],
            "accuracy": vs_bar["other_mean"],
            "difference": vs_bar["difference"], "difference_ci": vs_bar["difference_ci"],
            "mcnemar_p": mcnemar_p, "beats": beats,
            "rule": "beats = paired 95% stationary-bootstrap interval of the accuracy "
                    "difference above 0 and exact McNemar p < 0.05",
            "bootstrap": {"n_boot": config.n_boot, "mean_block_months": config.mean_block,
                          "seed": 0},
        },
        "selected_per_fold": [s["config"] for s in wf.selected],
    }


def train_for_serving(df: pd.DataFrame, config: TrainingConfig | None = None) -> TrainedModel:
    config = TrainingConfig() if config is None else config
    include_hmm = FEATURE_SETS[config.feature_set]
    classes = classes_for(config.target)
    ds = build_dataset(df, config.target, include_hmm=include_hmm, hmm_kwargs=config.hmm_kwargs)
    metrics = _evaluate(ds, config)

    tuned = tune(config.model, ds.X, ds.y, ds.t0, ds.t1, classes, seed=config.seed)
    fitted = fit(config.model, tuned.params, ds.X, ds.y, classes, config.seed)
    artifact = to_artifact(fitted)

    columns = list(ds.X.columns)
    as_of, row = serving_features(df, columns, include_hmm, config.hmm_kwargs)
    return TrainedModel(
        config=config, artifact=artifact, artifact_sha256=hashlib.sha256(artifact).digest(),
        features=columns, classes=list(classes),
        hyperparameters={
            "selected": tuned.params,
            "fixed": LIGHTGBM_FIXED | {"random_state": config.seed},
            "grid": grid_for(config.model),
            "selection": f"lowest pooled log loss over {INNER_SPLITS} purged inner folds "
                         f"of all {len(ds.y)} labelled months",
            "inner_trials": [{k: t[k] for k in ("config", "inner_log_loss", "selected")}
                             for t in tuned.trials],
        },
        train_first_decision=ds.t0.iloc[0].date(), train_last_decision=ds.t0.iloc[-1].date(),
        train_last_label_end=ds.t1.iloc[-1].date(), n_train=len(ds.y),
        metrics=metrics, serving_as_of=as_of,
        serving_features={str(k): float(v) for k, v in row.items()},
        serving_features_sha256=features_hash(as_of, row),
        library_versions=library_versions(),
    )


def config_dict(config: TrainingConfig) -> dict[str, Any]:
    return asdict(config)
