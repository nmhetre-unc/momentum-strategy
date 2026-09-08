"""
Bootstrap confidence intervals for strategy statistics.
"""

import numpy as np
import pandas as pd

from qbt.analytics import sharpe_ratio


def iid_bootstrap_sharpe(returns: pd.Series, n_boot: int = 2000, seed: int = 0):
    """
    Naive i.i.d. bootstrap: resamples `returns` with replacement n_boot
    times and computes the Sharpe ratio on each draw. Returns
    (point, lo, hi) -- the Sharpe on the original series, and the
    5th/95th percentiles of the bootstrap distribution. Draws rows
    independently, so it ignores autocorrelation in `returns`.
    """
    values = np.asarray(returns)
    n = len(values)
    rng = np.random.default_rng(seed)

    point = sharpe_ratio(returns)
    boot_sharpes = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_sharpes[i] = sharpe_ratio(sample)

    lo, hi = np.percentile(boot_sharpes, [5, 95])
    return point, lo, hi


def stationary_bootstrap_sharpe(returns: pd.Series, n_boot: int = 2000,
                                mean_block: int = 20, seed: int = 0):
    """
    Politis-Romano stationary bootstrap: resamples `returns` in blocks of
    geometrically-distributed length (mean `mean_block`), wrapping around
    to the start of the series when a block runs past the end, so runs of
    consecutive observations -- and the autocorrelation they carry -- are
    preserved rather than reshuffled. Same return signature as
    iid_bootstrap_sharpe.
    """
    values = np.asarray(returns)
    n = len(values)
    p = 1.0 / mean_block
    rng = np.random.default_rng(seed)

    point = sharpe_ratio(returns)
    boot_sharpes = np.empty(n_boot)
    for i in range(n_boot):
        pieces = []
        total = 0
        while total < n:
            start = rng.integers(0, n)
            length = rng.geometric(p)
            pieces.append(values[(start + np.arange(length)) % n])
            total += length
        boot_sharpes[i] = sharpe_ratio(np.concatenate(pieces)[:n])

    lo, hi = np.percentile(boot_sharpes, [5, 95])
    return point, lo, hi
