"""
Train/test splits for samples whose labels span time (Lopez de Prado,
Advances in Financial Machine Learning, ch. 7).

Each sample i has a decision time t0[i] (when its features are known
and the prediction is made) and a label end time t1[i] (when its label
becomes known). For the monthly targets, t1[i] is the next month-end,
which is also t0[i + 1].

Why plain k-fold leaks here
---------------------------
Plain k-fold puts samples from after a test fold into the training set,
and those samples are built from the test fold's own prices:

1. Features look back. ret_3m, ret_12m, vol_3m, drawdown_12m and the
   regime features on a sample dated a few months after a test month are
   computed from that test month's returns. A model fitted on them has
   been shown the test period.
2. Labels overlap, or are serially dependent. The training sample just
   before a test month has a label that ends exactly where the test
   month begins (they share the close at t0), and the regime label is a
   function of 60-day trend and volatility windows that span several
   months. Neighbouring labels are therefore nearly the same, and both
   features and labels are strongly autocorrelated. A model with
   neighbours on both sides of a test sample can interpolate its answer
   rather than predict it.

Either way, the test score measures recall of the test period rather
than prediction of it, and it is optimistic by an amount you can't
know.

The fixes
---------
- Purging drops every training sample whose label interval
  [t0, t1] intersects the test fold's span [first test t0, last test t1].
  Intervals are closed, so a label ending at the first test t0 is purged
  too. This is the conservative reading: that label is known only at the
  same close where the test prediction is made.
- An embargo additionally drops the `embargo` samples immediately after
  the test fold. This covers the feature look-back in point 1, which
  purging by label interval doesn't see. The default, EMBARGO_MONTHS =
  12, matches the longest fixed look-back (ret_12m, mom_12_1,
  drawdown_12m). Some features have no fixed look-back: the `rules`
  regime ranks volatility against its whole expanding history, and the
  HMM features are standardized on an expanding window. So no finite
  embargo makes k-fold fully clean, which is one more reason the
  out-of-sample estimate here is walk-forward.
- Walk-forward trains only on samples before each test fold, so no
  training sample ever comes after the test data. Purging still matters
  at the train/test boundary. The embargo, by construction, has nothing
  after the test fold to remove: walk_forward_splits() accepts it so
  both splitters share one rule, and reports zero embargoed samples.
  purged_kfold_splits() is where the embargo does work. It is here as
  the reference construction and for comparing against walk-forward.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Longest fixed feature look-back in qbt.ml.features, in months (252 days).
EMBARGO_MONTHS = 12


@dataclass(frozen=True)
class Fold:
    number: int
    train: np.ndarray       # positions of training samples, after purge and embargo
    test: np.ndarray        # positions of test samples (one contiguous block)
    purged: np.ndarray      # positions removed because their label overlaps the test span
    embargoed: np.ndarray   # positions removed by the post-test embargo


def _validate(t0: pd.Series, t1: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if len(t0) != len(t1):
        raise ValueError("t0 and t1 must have the same length")
    start = pd.DatetimeIndex(t0).to_numpy()
    end = pd.DatetimeIndex(t1).to_numpy()
    if np.isnat(start).any() or np.isnat(end).any():
        raise ValueError("t0 and t1 must not contain NaT (drop unlabelled samples first)")
    if not (start[1:] > start[:-1]).all():
        raise ValueError("t0 must be strictly increasing")
    if (end < start).any():
        raise ValueError("every label must end at or after its decision time")
    return start, end


def purge_and_embargo(t0: pd.Series, t1: pd.Series, candidates: np.ndarray,
                      test: np.ndarray, embargo: int = 0) -> Fold:
    """
    Removes from `candidates` (training positions) every sample whose
    [t0, t1] intersects the test span, then the `embargo` positions right
    after the test block. Returns a Fold numbered 0; callers renumber.
    """
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    start, end = _validate(t0, t1)
    test = np.asarray(test, dtype=int)
    if len(test) == 0 or not (np.diff(test) == 1).all():
        raise ValueError("test must be a non-empty contiguous block of positions")
    candidates = np.setdiff1d(np.asarray(candidates, dtype=int), test)

    span_start, span_end = start[test[0]], end[test].max()
    overlaps = (start[candidates] <= span_end) & (end[candidates] >= span_start)
    purged = candidates[overlaps]
    kept = candidates[~overlaps]

    after = np.arange(test[-1] + 1, min(test[-1] + 1 + embargo, len(start)))
    embargoed = np.intersect1d(kept, after)
    train = np.setdiff1d(kept, embargoed)
    return Fold(0, train, test, purged, embargoed)


def walk_forward_splits(t0: pd.Series, t1: pd.Series, min_train: int = 60,
                        test_size: int = 24, embargo: int = EMBARGO_MONTHS) -> list[Fold]:
    """
    Expanding-window walk-forward: each fold trains on every purged
    sample before its test block, and test blocks of `test_size` follow
    each other to the end of the data (the last one may be shorter).
    The first test block starts at the earliest position that leaves at
    least `min_train` training samples after purging.
    """
    if min_train < 1 or test_size < 1:
        raise ValueError("min_train and test_size must be >= 1")
    n = len(t0)
    start = None
    for s in range(min_train, n):
        fold = purge_and_embargo(t0, t1, np.arange(s), np.arange(s, min(s + test_size, n)),
                                 embargo)
        if len(fold.train) >= min_train:
            start = s
            break
    if start is None:
        raise ValueError(
            f"{n} samples can't give {min_train} purged training samples plus a test block"
        )

    folds = []
    for number, s in enumerate(range(start, n, test_size)):
        test = np.arange(s, min(s + test_size, n))
        fold = purge_and_embargo(t0, t1, np.arange(s), test, embargo)
        folds.append(Fold(number, fold.train, fold.test, fold.purged, fold.embargoed))
    return folds


def purged_kfold_splits(t0: pd.Series, t1: pd.Series, n_splits: int = 5,
                        embargo: int = EMBARGO_MONTHS) -> list[Fold]:
    """
    Contiguous k-fold with training on both sides of each test block,
    purged and embargoed. Not walk-forward: it trains on later data, so
    use it only as a comparison or for model selection inside a training
    window, never as the out-of-sample estimate.
    """
    n = len(t0)
    if n_splits < 2 or n_splits > n:
        raise ValueError(f"n_splits must be between 2 and {n}, got {n_splits}")
    folds = []
    for number, test in enumerate(np.array_split(np.arange(n), n_splits)):
        fold = purge_and_embargo(t0, t1, np.arange(n), test, embargo)
        folds.append(Fold(number, fold.train, fold.test, fold.purged, fold.embargoed))
    return folds


def fold_table(folds: list[Fold], t0: pd.Series) -> pd.DataFrame:
    """Sample counts and date ranges per fold, for reporting."""
    dates = pd.DatetimeIndex(t0)
    rows = []
    for fold in folds:
        rows.append({
            "fold": fold.number,
            "train_start": dates[fold.train[0]].date() if len(fold.train) else None,
            "train_end": dates[fold.train[-1]].date() if len(fold.train) else None,
            "test_start": dates[fold.test[0]].date(),
            "test_end": dates[fold.test[-1]].date(),
            "n_train": len(fold.train),
            "n_purged": len(fold.purged),
            "n_embargoed": len(fold.embargoed),
            "n_test": len(fold.test),
        })
    return pd.DataFrame(rows)
