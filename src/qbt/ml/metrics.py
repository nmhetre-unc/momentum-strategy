"""
Scoring for C2's out-of-sample probability forecasts, and the
resampling used to put intervals on them.

- auc(): binary ROC AUC for two classes; for more, the macro average of
  one-vs-rest AUCs over the classes that both occur and don't occur in
  y (others are undefined and skipped; the count used is returned).
- log_loss / brier(): multiclass Brier is the mean over rows of the sum
  over classes of (p_k - 1[y = k])^2, so for two classes it is twice
  the familiar binary Brier score. brier_binary() gives the familiar one.
- calibration_bins(): equal-count bins of the forecast probability with
  the observed frequency in each, for reliability plots.
- paired_mean_bootstrap(): a per-month score's mean for a model (accuracy
  from 1/0 correctness, or mean log loss), and its difference from each
  baseline on the same test months, with stationary-bootstrap
  intervals. Months are resampled in blocks (qbt.bootstrap) because
  monthly labels, regimes especially, are autocorrelated; an i.i.d.
  interval would be too narrow. One shared index set per replicate is
  applied to every predictor, so the differences are paired.
- paired_sharpe_bootstrap(): the same for a strategy's Sharpe ratio and
  its difference from buy-and-hold over identical months.
"""

from typing import Any

import numpy as np
import pandas as pd

from qbt.bootstrap import stationary_bootstrap_indices

MONTHS_PER_YEAR = 12
N_BOOT = 2000
MEAN_BLOCK = 6       # months
CI_LEVEL = 0.95


def _rows(y: np.ndarray, classes: tuple[float, ...]) -> np.ndarray:
    return np.array([classes.index(float(v)) for v in y])


def log_loss_per_month(y: np.ndarray, proba: np.ndarray,
                       classes: tuple[float, ...]) -> np.ndarray:
    return -np.log(proba[np.arange(len(y)), _rows(y, classes)])


def log_loss(y: np.ndarray, proba: np.ndarray, classes: tuple[float, ...]) -> float:
    return float(np.mean(log_loss_per_month(y, proba, classes)))


def brier(y: np.ndarray, proba: np.ndarray, classes: tuple[float, ...]) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), _rows(y, classes)] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def brier_binary(y: np.ndarray, p_positive: np.ndarray) -> float:
    return float(np.mean((p_positive - y) ** 2))


def _binary_auc(positive: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as half."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc(y: np.ndarray, proba: np.ndarray, classes: tuple[float, ...]) -> tuple[float, int]:
    """(AUC, number of classes it averages over); NaN if undefined."""
    if len(classes) == 2:
        value = _binary_auc(y == classes[1], proba[:, 1])
        return value, 0 if np.isnan(value) else 2
    values = [_binary_auc(y == c, proba[:, k]) for k, c in enumerate(classes)]
    defined = [v for v in values if not np.isnan(v)]
    return (float(np.mean(defined)) if defined else np.nan), len(defined)


def predicted_class(proba: np.ndarray, classes: tuple[float, ...]) -> np.ndarray:
    return np.asarray(classes)[proba.argmax(axis=1)]


def calibration_bins(p: np.ndarray, outcome: np.ndarray, n_bins: int = 5) -> list[dict]:
    """
    Equal-count bins of forecast probability `p` (ties kept together),
    each with its mean forecast, observed frequency of `outcome` and size.
    """
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        rows.append({"mean_forecast": float(p[chunk].mean()),
                     "observed_frequency": float(outcome[chunk].mean()),
                     "n": int(len(chunk))})
    return rows


def _interval(values: np.ndarray) -> list[float]:
    tail = (1 - CI_LEVEL) / 2 * 100
    lo, hi = np.percentile(values, [tail, 100 - tail])
    return [float(lo), float(hi)]


def paired_mean_bootstrap(values: dict[str, np.ndarray], focus: str,
                          higher_is_better: bool = True, n_boot: int = N_BOOT,
                          mean_block: float = MEAN_BLOCK, seed: int = 0) -> dict:
    """
    `values` maps predictor name -> a per-month score (1.0/0.0 for
    correct/incorrect gives accuracy; per-month log loss gives mean log
    loss), all over the same months in the same order. Returns the focus
    predictor's mean with its interval and, for every other predictor,
    the paired difference focus - other with its interval and the share
    of replicates in which the focus predictor is not better.
    """
    names = list(values)
    matrix = np.vstack([np.asarray(values[name], dtype=float) for name in names])
    n = matrix.shape[1]
    boot = np.empty((n_boot, len(names)))
    indices = stationary_bootstrap_indices(n, n_boot, mean_block,
                                           np.random.default_rng(seed))
    for i, idx in enumerate(indices):
        boot[i] = matrix[:, idx].mean(axis=1)
    f = names.index(focus)
    means = matrix.mean(axis=1)
    out: dict[str, Any] = {"mean": float(means[f]), "ci": _interval(boot[:, f]), "vs": {}}
    for j, name in enumerate(names):
        if j == f:
            continue
        diff = boot[:, f] - boot[:, j]
        not_better = diff <= 0 if higher_is_better else diff >= 0
        out["vs"][name] = {
            "other_mean": float(means[j]),
            "difference": float(means[f] - means[j]),
            "difference_ci": _interval(diff),
            "share_of_replicates_not_better": float(np.mean(not_better)),
        }
    return out


def monthly_sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe of monthly returns (no risk-free rate, like C1)."""
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(MONTHS_PER_YEAR)) if std > 0 else np.nan


def paired_sharpe_bootstrap(strategy: np.ndarray, benchmark: np.ndarray,
                            n_boot: int = N_BOOT, mean_block: float = MEAN_BLOCK,
                            seed: int = 0) -> dict:
    if len(strategy) != len(benchmark):
        raise ValueError("strategy and benchmark must cover the same months")
    s_boot, d_boot = np.empty(n_boot), np.empty(n_boot)
    indices = stationary_bootstrap_indices(len(strategy), n_boot, mean_block,
                                           np.random.default_rng(seed))
    for i, idx in enumerate(indices):
        s_boot[i] = monthly_sharpe(strategy[idx])
        d_boot[i] = s_boot[i] - monthly_sharpe(benchmark[idx])
    s, b = monthly_sharpe(strategy), monthly_sharpe(benchmark)
    return {
        "sharpe": s, "sharpe_ci": _interval(s_boot[~np.isnan(s_boot)]),
        "benchmark_sharpe": b,
        "difference": s - b, "difference_ci": _interval(d_boot[~np.isnan(d_boot)]),
        "share_of_replicates_not_better": float(np.mean(~(d_boot > 0))),
        "n_degenerate_replicates": int(np.isnan(s_boot).sum()),
    }
