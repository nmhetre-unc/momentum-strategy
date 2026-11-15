"""
Assembles the monthly dataset and scores the baselines out of sample
across walk-forward folds.

Built so that a weak result is still credible:
- every test prediction comes from a fold that trained only on earlier,
  purged samples (qbt.ml.splits);
- accuracy is reported with its sample size and a Wilson 95% interval.
  With about 200 monthly samples, the standard error of an accuracy
  near 55% is about 3.5 points, so an edge smaller than that is noise;
- two predictors are compared with an exact McNemar test on the same
  test months, not by eyeballing two accuracies;
- the trading comparison charges cost_bps on every position change and
  reports buy-and-hold alongside.

Caveat on the regime target: monthly regimes are strongly
autocorrelated, so successive test months are not independent and the
intervals here are too narrow for it. Read them as a lower bound on the
uncertainty.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from qbt.ml.baselines import (
    conditional_prior_proba,
    predict_buy_and_hold,
    predict_majority,
    predict_previous,
    predict_rules_regime,
)
from qbt.ml.features import month_end_dates, monthly_features
from qbt.ml.labels import (
    direction_labels,
    forward_returns,
    label_end_times,
    previous_labels,
    regime_labels,
)
from qbt.ml.splits import Fold, fold_table

TARGETS = ("direction", "regime")
REGIME_CLASSES = (0.0, 1.0, 2.0, 3.0)
# The baseline a model has to beat (C2): for direction, always up (the
# majority class, i.e. buy-and-hold); for regime, the rules-regime lookup
# table, which beats both the raw rules regime and every C2 model.
BAR = {"direction": "majority_class", "regime": "rules_regime_lookup"}
MONTHS_PER_YEAR = 12
WILSON_Z = 1.959963984540054  # two-sided 95%


@dataclass(frozen=True)
class MonthlyDataset:
    target: str
    X: pd.DataFrame               # features, indexed by decision date t0
    y: pd.Series                  # label for the month (t0, t1]
    t1: pd.Series                 # label end time (the next decision date)
    previous: pd.Series           # label of the month ending at t0 (known at t0)
    forward_return: pd.Series     # close-to-close return over (t0, t1]

    @property
    def t0(self) -> pd.Series:
        return pd.Series(self.X.index, index=self.X.index)


def build_dataset(df: pd.DataFrame, target: str = "direction", include_hmm: bool = True,
                  hmm_kwargs: dict | None = None,
                  daily_regimes: pd.Series | None = None) -> MonthlyDataset:
    """
    One row per complete decision month whose features, label and
    previous label all exist. `previous` is computed on the full monthly
    grid before any rows are dropped, so it is always the immediately
    preceding month even when warm-up rows are removed.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    dates = month_end_dates(pd.DatetimeIndex(df.index))
    X = monthly_features(df, include_hmm=include_hmm, hmm_kwargs=hmm_kwargs)
    if target == "direction":
        y = direction_labels(df, dates)
    else:
        y = regime_labels(df, dates, daily_regimes=daily_regimes)
    previous = previous_labels(y)
    t1 = label_end_times(dates)
    fwd = forward_returns(df, dates)

    keep = X.notna().all(axis=1) & y.notna() & previous.notna()
    return MonthlyDataset(target=target, X=X[keep], y=y[keep], t1=t1[keep],
                          previous=previous[keep], forward_return=fwd[keep])


def baseline_predictions(ds: MonthlyDataset, folds: list[Fold],
                         prob_floor: float = 1e-4) -> pd.DataFrame:
    """
    Per test month: fold, y, and every classification baseline's
    prediction, each from what its fold may see. Regime adds the rules
    regime on the decision date and rules_regime_lookup: the most frequent
    next-month regime among training months with the same rules regime on
    their decision date (ties to the lower class; an unseen regime falls
    back to the training class frequencies).
    """
    pieces = []
    for fold in folds:
        cols = {"fold": fold.number,
                "y": ds.y.iloc[fold.test].to_numpy(dtype=float),
                "majority_class": predict_majority(ds.y, fold),
                "previous_label": predict_previous(ds.previous, fold)}
        if ds.target == "regime":
            cols["rules_regime_at_decision"] = predict_rules_regime(ds.X, fold)
            rules = ds.X["regime_rules"]
            lookup = conditional_prior_proba(ds.y.iloc[fold.train], rules.iloc[fold.train],
                                             rules.iloc[fold.test], REGIME_CLASSES, prob_floor)
            cols["rules_regime_lookup"] = np.asarray(REGIME_CLASSES)[lookup.argmax(axis=1)]
        pieces.append(pd.DataFrame(cols, index=ds.y.index[fold.test]))
    out = pd.concat(pieces)
    if not out.index.is_unique:
        raise ValueError("test blocks overlap; each month must be tested once")
    return out


# ---- statistics ----

def wilson_interval(correct: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = correct / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall over the classes present in y_true."""
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(recalls)) if recalls else np.nan


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for paired correct/incorrect outcomes."""
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    if a_only + b_only == 0:
        return 1.0
    return float(binomtest(a_only, a_only + b_only, 0.5).pvalue)


# ---- trading ----

def strategy_returns(positions: pd.Series, forward_return: pd.Series,
                     cost_bps: float = 5.0) -> pd.Series:
    """
    Monthly net return of holding `positions` (1 long, 0 flat) from each
    decision date to the next, paying cost_bps on every change in
    position, including the first entry from flat.
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    turnover = positions.diff().abs().fillna(positions.abs())
    return positions * forward_return - turnover * cost_bps / 10_000


def trading_summary(returns: pd.Series, positions: pd.Series) -> dict:
    equity = (1 + returns).cumprod()
    n = len(returns)
    std = returns.std(ddof=1)
    return {
        "n_months": n,
        "total_return": float(equity.iloc[-1] - 1) if n else np.nan,
        "ann_return": float(equity.iloc[-1] ** (MONTHS_PER_YEAR / n) - 1) if n else np.nan,
        "ann_vol": float(std * np.sqrt(MONTHS_PER_YEAR)) if n > 1 else np.nan,
        "sharpe": (float(returns.mean() / std * np.sqrt(MONTHS_PER_YEAR))
                   if n > 1 and std > 0 else np.nan),
        "max_drawdown": float((equity / equity.cummax() - 1).min()) if n else np.nan,
        "exposure": float(positions.mean()) if n else np.nan,
    }


# ---- evaluation ----

@dataclass(frozen=True)
class BaselineReport:
    folds: pd.DataFrame           # per fold: sizes, dates, per-baseline accuracy
    predictions: pd.DataFrame     # per test month: fold, y, each baseline's prediction
    summary: pd.DataFrame         # per baseline: n, accuracy, Wilson CI, balanced accuracy
    mcnemar_p: float              # previous_label vs majority_class
    trading: pd.DataFrame | None  # direction only: per strategy, trading_summary()


def evaluate_baselines(ds: MonthlyDataset, folds: list[Fold],
                       cost_bps: float = 5.0) -> BaselineReport:
    classifiers = ["majority_class", "previous_label"]
    pieces = []
    for fold in folds:
        pieces.append(pd.DataFrame({
            "fold": fold.number,
            "y": ds.y.iloc[fold.test].to_numpy(dtype=float),
            "majority_class": predict_majority(ds.y, fold),
            "previous_label": predict_previous(ds.previous, fold),
            "buy_and_hold": predict_buy_and_hold(fold),
        }, index=ds.y.index[fold.test]))
    predictions = pd.concat(pieces)
    if not predictions.index.is_unique:
        raise ValueError("test blocks overlap; each month must be tested once")

    table = fold_table(folds, ds.t0)
    for name in classifiers:
        correct = predictions[name] == predictions["y"]
        table[f"acc_{name}"] = correct.groupby(predictions["fold"]).mean().to_numpy()

    y_true = predictions["y"].to_numpy()
    rows = []
    for name in classifiers:
        y_pred = predictions[name].to_numpy()
        n_correct, n = int((y_pred == y_true).sum()), len(y_true)
        low, high = wilson_interval(n_correct, n)
        rows.append({"predictor": name, "n": n, "accuracy": n_correct / n if n else np.nan,
                     "ci_low": low, "ci_high": high,
                     "balanced_accuracy": balanced_accuracy(y_true, y_pred)})
    summary = pd.DataFrame(rows).set_index("predictor")

    mcnemar_p = mcnemar_exact((predictions["previous_label"] == predictions["y"]).to_numpy(),
                              (predictions["majority_class"] == predictions["y"]).to_numpy())

    trading = None
    if ds.target == "direction":
        fwd = ds.forward_return.loc[predictions.index]
        trading = pd.DataFrame({
            name: trading_summary(strategy_returns(predictions[name], fwd, cost_bps),
                                  predictions[name])
            for name in ["buy_and_hold", *classifiers]
        }).T
    else:
        predictions = predictions.drop(columns="buy_and_hold")

    return BaselineReport(folds=table, predictions=predictions, summary=summary,
                          mcnemar_p=mcnemar_p, trading=trading)
