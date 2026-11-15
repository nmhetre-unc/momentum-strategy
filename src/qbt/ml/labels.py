"""
Targets for monthly prediction, one row per decision date t (see
qbt.ml.features.month_end_dates()). Unlike features, labels look
forward on purpose: the label on row t describes the month (t, t_next],
where t_next is the next decision date. Every label row carries that
end time (label_end_times()), because the splits in qbt.ml.splits need
to know when each label became known in order to purge it.

- direction_labels(): 1.0 if the close at t_next is above the close at
  t, else 0.0 (a flat month counts as not up).
- regime_labels(): the most common daily `rules` regime over the
  trading days in (t, t_next], ties going to the regime seen latest.
  `rules` is used because it fits nothing: a fitted regime model would
  make the target itself depend on which window it was fitted on, so
  the "answer" would move between folds.

  Much of this label is already determined at t. Each daily `rules`
  regime classifies on a 60-day trend and a 20-day volatility ranked
  against its expanding history, so on day k of month t+1, 60-k of the
  trend's daily returns and 20-k of the volatility's are known at the
  decision date. The label therefore often equals the regime already
  observable at t: the regime_rules feature, or previous_labels(). A
  model has to beat that persistence, not the majority class, before it
  can claim to predict regime changes. scripts/ml_c1_report.py measures
  how often the label equals each (results/ml/c1_baselines.json,
  "regime_label_determinacy").

The last decision date has no next month, so its label is NaN, never
filled.

previous_labels() shifts a label one row back: the label of the month
that ended at t, known at t's close. It is what the "last month" and
"current regime" baselines predict with.
"""

import numpy as np
import pandas as pd

from qbt.regime import UNKNOWN, detect_regimes


def label_end_times(dates: pd.DatetimeIndex) -> pd.Series:
    """t_next for every decision date t; NaT for the last one."""
    return pd.Series(dates, index=dates).shift(-1)


def forward_returns(df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Close-to-close return from each decision date to the next."""
    closes = df["Close"].astype(float).loc[dates]
    return closes.shift(-1) / closes - 1


def direction_labels(df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    ret = forward_returns(df, dates)
    return (ret > 0).astype(float).where(ret.notna())


def daily_rules_regimes(df: pd.DataFrame) -> pd.Series:
    """Unsmoothed daily `rules` regime labels; UNKNOWN during warm-up."""
    return detect_regimes(df, method="rules", smooth="none").labels


def _modal_regime(window: np.ndarray) -> float:
    known = window[window != UNKNOWN]
    if len(known) == 0:
        return np.nan
    values, counts = np.unique(known, return_counts=True)
    tied = values[counts == counts.max()]
    if len(tied) == 1:
        return float(tied[0])
    # Ties go to whichever tied regime appeared latest in the month.
    last_seen = {v: np.flatnonzero(known == v)[-1] for v in tied}
    return float(max(tied, key=lambda v: last_seen[v]))


def regime_labels(df: pd.DataFrame, dates: pd.DatetimeIndex,
                  daily_regimes: pd.Series | None = None) -> pd.Series:
    """
    Modal daily regime over (t, t_next] for each decision date t.
    `daily_regimes` overrides the `rules` labels (tests use it to supply
    a known regime path).
    """
    daily = daily_rules_regimes(df) if daily_regimes is None else daily_regimes
    values = daily.to_numpy()
    positions = daily.index.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("every decision date must be a day in the regime series")
    out = np.full(len(dates), np.nan)
    for i in range(len(dates) - 1):
        out[i] = _modal_regime(values[positions[i] + 1:positions[i + 1] + 1])
    return pd.Series(out, index=dates)


def previous_labels(labels: pd.Series) -> pd.Series:
    """The label of the month ending at each decision date (known at t)."""
    return labels.shift(1)
