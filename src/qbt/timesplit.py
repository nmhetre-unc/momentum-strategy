"""
The one chronological split point shared by every fitted strategy and
every evaluator that splits history into in-sample and out-of-sample.

`frac` is a fraction of ALL rows of the price frame, warm-up rows
included. A model that drops warm-up rows before fitting still splits
on this date, so the strategy and evaluate_out_of_sample() (and the
per-fold override in rolling_walk_forward()) always agree on which day
is the first out-of-sample day.
"""

import math

import pandas as pd


def split_index(n: int, frac: float) -> int:
    """
    Position of the first out-of-sample row when the first `frac` of `n`
    rows are in-sample. Equal to int(n * frac), except that a product a
    hair below an integer through floating-point rounding (882 * (756 /
    882)) still lands on that integer.
    """
    if not 0 < frac < 1:
        raise ValueError(f"split fraction must be strictly between 0 and 1, got {frac}")
    index = math.floor(n * frac + 1e-9)
    if not 0 < index < n:
        raise ValueError(f"a {frac} split of {n} rows leaves an empty side")
    return index


def split_date(index: pd.Index, frac: float) -> pd.Timestamp:
    """The first out-of-sample date: index[split_index(len(index), frac)]."""
    return pd.Timestamp(index[split_index(len(index), frac)])


# A strategy that learns anything from history (a fitted model, a regime
# model, a choice selected on past performance) holds no position until
# every day it learned from is behind it, and records the first day it
# may trade under this key in the returned signal's `attrs`. Evaluators
# score it from that day on only, never on data it learned from.
OUT_OF_SAMPLE_ATTR = "out_of_sample_from"


def mark_out_of_sample(signal: pd.Series, start: pd.Timestamp) -> pd.Series:
    """
    `signal` flattened before `start`, with `start` recorded as its first
    out-of-sample day. A position decided on `start` earns from the next
    day (run_backtest shifts it), so the strategy's return on `start`
    itself is zero.
    """
    out = signal.where(signal.index >= start, 0.0).astype(float)
    out.attrs[OUT_OF_SAMPLE_ATTR] = pd.Timestamp(start)
    return out


def out_of_sample_start(signal: pd.Series) -> pd.Timestamp | None:
    """The first day `signal` may hold a position, or None for a strategy
    that learns nothing (every day is out of sample)."""
    start = signal.attrs.get(OUT_OF_SAMPLE_ATTR)
    return None if start is None else pd.Timestamp(start)


def scoring_split(index: pd.Index, frac: float, signal: pd.Series) -> pd.Timestamp:
    """
    Where an evaluator's out-of-sample half starts: the shared split date,
    or the strategy's own first out-of-sample day if that is later, so
    the out-of-sample half never contains a day the strategy learned from.
    """
    shared = split_date(index, frac)
    own = out_of_sample_start(signal)
    return shared if own is None or own <= shared else own
