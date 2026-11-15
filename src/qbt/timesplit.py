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
