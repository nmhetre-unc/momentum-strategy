"""
Bootstrap confidence intervals and multiple-testing-aware significance
statistics for strategy performance.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from qbt.analytics import sharpe_ratio

# Euler-Mascheroni constant, used in the expected-maximum-Sharpe term below.
EULER_MASCHERONI = 0.5772156649015329


def iid_bootstrap_sharpe(returns: pd.Series, n_boot: int = 2000, seed: int = 0):
    """
    Naive i.i.d. bootstrap: resamples `returns` with replacement n_boot
    times and computes the Sharpe ratio on each draw. Returns
    (point, lo, hi) -- the Sharpe on the original series, and the
    5th/95th percentiles of the bootstrap distribution. Draws rows
    independently, so it ignores autocorrelation in `returns`.
    """
    values: np.ndarray = np.asarray(returns, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)

    point = sharpe_ratio(returns)
    boot_sharpes: np.ndarray = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_sharpes[i] = sharpe_ratio(sample)

    lo, hi = np.percentile(boot_sharpes, [5, 95])
    return point, lo, hi


def deflated_sharpe_ratio(sharpe: float, n_trials: int, skew: float, kurtosis: float,
                          n_obs: int, sharpe_benchmark: float = 0.0) -> float:
    """
    P(true Sharpe > sharpe_benchmark), after correcting for selection bias
    from testing `n_trials` independent strategies and for non-normal
    returns. Bailey, D.H. and Lopez de Prado, M. (2014), "The Deflated
    Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and
    Non-Normality," Journal of Portfolio Management, 40(5), 94-107.

    `sharpe` must be the per-period (non-annualized) Sharpe ratio,
    consistent with `n_obs` observations at that period length --
    annualizing one without the other breaks the formula. `kurtosis` is
    on the Pearson scale (3.0 for a normal distribution, not excess).
    At `n_trials <= 1` there is no multiple-testing correction to apply,
    so this reduces to the plain probabilistic Sharpe ratio versus
    `sharpe_benchmark`.

    `sharpe_benchmark` defaults to 0.0 -- "beats doing nothing." Pass a
    benchmark's own per-period Sharpe (same units, same period) to ask
    the harder "beats holding the benchmark" question instead; only the
    final threshold shifts; the selection-bias correction (`sr0`) is
    still computed from the STRATEGY's own moments, per Bailey & Lopez de
    Prado's derivation, not the benchmark's. Do not call this with the
    benchmark's own Sharpe as `sharpe` and itself as `sharpe_benchmark` --
    the numerator collapses to `-sr0`, which is a category error (a fixed
    reference point isn't a hypothesis that multiple-testing correction
    applies to); leave that cell unset instead.
    """
    sr_std = np.sqrt((1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe ** 2) / (n_obs - 1))

    if n_trials <= 1:
        sr0 = 0.0
    else:
        sr0 = sr_std * (
            (1 - EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n_trials)
            + EULER_MASCHERONI * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )

    return float(norm.cdf((sharpe - sharpe_benchmark - sr0) / sr_std))


def newey_west_se(returns: pd.Series, lags: int | None = None) -> float:
    """
    Newey-West HAC standard error of the sample mean of `returns`,
    robust to serial correlation and heteroskedasticity. `lags` defaults
    to floor(4 * (n/100)**(2/9)), the automatic bandwidth rule from
    Newey & West (1994).
    """
    values: np.ndarray = np.asarray(returns, dtype=float)
    n = len(values)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    demeaned: np.ndarray = values - values.mean()
    variance = np.dot(demeaned, demeaned) / n
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        autocovariance = np.dot(demeaned[lag:], demeaned[:-lag]) / n
        variance += 2 * weight * autocovariance

    return float(np.sqrt(variance / n))


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
    values: np.ndarray = np.asarray(returns, dtype=float)
    n = len(values)
    p = 1.0 / mean_block
    rng = np.random.default_rng(seed)

    point = sharpe_ratio(returns)
    boot_sharpes: np.ndarray = np.empty(n_boot)
    for i in range(n_boot):
        pieces: list[np.ndarray] = []
        total = 0
        while total < n:
            start = rng.integers(0, n)
            length = rng.geometric(p)
            pieces.append(values[(start + np.arange(length)) % n])
            total += length
        block: np.ndarray = np.concatenate(pieces)
        boot_sharpes[i] = sharpe_ratio(block[:n])

    lo, hi = np.percentile(boot_sharpes, [5, 95])
    return point, lo, hi
