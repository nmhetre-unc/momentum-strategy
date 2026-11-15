"""
Point-in-time features for monthly SPY prediction.

Decisions are made once a month, at the close of the last trading day
of each complete calendar month (month_end_dates()). Every feature is
first built as a daily series in which row t uses only prices up to
and including day t's close (daily_features()), then sampled on those
decision dates (monthly_features()). Building daily and sampling keeps
the causality claim checkable: tests/test_ml_monthly.py truncates the
price history at several cutoffs and requires every column to be
unchanged on every surviving day, which is the lookahead canary each
feature gets.

Regime features come from the existing models in qbt.regime, used only
in their causal modes: `rules` fits nothing, and the HMM runs through
detect_regimes_walk_forward(), which labels each day with a model
fitted only on earlier days and decodes with the forward filter.
"""

import numpy as np
import pandas as pd

from qbt.regime import UNKNOWN, detect_regimes, detect_regimes_walk_forward
from qbt.regime_features import realized_volatility

TRADING_DAYS_PER_MONTH = 21

# Trailing return horizons, in trading days.
RETURN_HORIZONS = {"ret_1m": 21, "ret_3m": 63, "ret_6m": 126, "ret_12m": 252}

BASE_FEATURES = [
    *RETURN_HORIZONS,
    "mom_12_1",       # 12-month return skipping the latest month (classic momentum)
    "sma200_gap",     # close / 200-day SMA - 1
    "vol_1m",         # annualized realized volatility, 21 days
    "vol_3m",         # annualized realized volatility, 63 days
    "drawdown_12m",   # close / trailing 252-day high - 1
    "regime_rules",   # qbt.regime "rules" label (0-3), nothing fitted
]

HMM_STATES = 3
HMM_FEATURES = ["regime_hmm", *(f"hmm_p{k}" for k in range(HMM_STATES))]


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    The last trading day in `index` of each calendar month, i.e. the
    decision dates. The final month is dropped unless the data reaches
    that month's last business day: data ending on 13 November would
    otherwise make 13 November look like November's close. (A month
    whose true last trading day falls before its last business day, such
    as a Good Friday on the 31st, is dropped too when it is the final
    month; losing one month is the conservative error.)
    """
    if len(index) == 0:
        return pd.DatetimeIndex([])
    periods = index.to_period("M")
    is_last = np.append(periods[1:] != periods[:-1], True)
    ends = index[is_last]
    last = index[-1]
    if last != last + pd.offsets.BMonthEnd(0):
        ends = ends[:-1]
    return pd.DatetimeIndex(ends)


def _regime_rules(df: pd.DataFrame) -> pd.Series:
    labels = detect_regimes(df, method="rules", smooth="none").labels
    return labels.where(labels != UNKNOWN).astype(float)


def _regime_hmm(df: pd.DataFrame, hmm_kwargs: dict | None) -> pd.DataFrame:
    kwargs = {"n_regimes": HMM_STATES, **(hmm_kwargs or {})}
    if kwargs["n_regimes"] != HMM_STATES:
        raise ValueError(f"hmm features are built for n_regimes={HMM_STATES}")
    result = detect_regimes_walk_forward(df, method="hmm", **kwargs)
    out = pd.DataFrame(index=df.index)
    out["regime_hmm"] = result.labels.where(result.labels != UNKNOWN).astype(float)
    probabilities = result.probabilities
    for k in range(HMM_STATES):
        if probabilities is not None and k in probabilities.columns:
            out[f"hmm_p{k}"] = probabilities[k].astype(float)
        else:
            out[f"hmm_p{k}"] = np.nan
    return out


def daily_features(df: pd.DataFrame, include_hmm: bool = True,
                   hmm_kwargs: dict | None = None) -> pd.DataFrame:
    """
    One row per trading day; row t uses only prices up to day t's close.
    Warm-up rows are NaN, never back-filled. `hmm_kwargs` is passed to
    detect_regimes_walk_forward() (e.g. initial_train, refit_every).
    """
    close = df["Close"].astype(float)
    features = pd.DataFrame(index=df.index)
    for name, days in RETURN_HORIZONS.items():
        features[name] = close.pct_change(days, fill_method=None)
    features["mom_12_1"] = close.shift(TRADING_DAYS_PER_MONTH) / close.shift(252) - 1
    features["sma200_gap"] = close / close.rolling(200).mean() - 1
    features["vol_1m"] = realized_volatility(close, 21)
    features["vol_3m"] = realized_volatility(close, 63)
    features["drawdown_12m"] = close / close.rolling(252).max() - 1
    features["regime_rules"] = _regime_rules(df)
    if include_hmm:
        features = features.join(_regime_hmm(df, hmm_kwargs))
    return features


def monthly_features(df: pd.DataFrame, include_hmm: bool = True,
                     hmm_kwargs: dict | None = None) -> pd.DataFrame:
    """daily_features() sampled on the decision dates (month_end_dates())."""
    daily = daily_features(df, include_hmm=include_hmm, hmm_kwargs=hmm_kwargs)
    return daily.loc[month_end_dates(pd.DatetimeIndex(df.index))]
