"""
Confirms the stationary bootstrap accounts for autocorrelation that the
naive i.i.d. bootstrap ignores, and checks the deflated Sharpe ratio and
Newey-West standard error against known directional properties.
"""

import numpy as np

from qbt.stats import (
    deflated_sharpe_ratio,
    iid_bootstrap_sharpe,
    newey_west_se,
    stationary_bootstrap_sharpe,
)


def test_stationary_interval_wider_than_iid_under_autocorrelation():
    rng = np.random.default_rng(7)
    n = 1000
    phi = 0.6
    mu = 0.0005
    sigma = 0.01

    returns = np.empty(n)
    returns[0] = mu
    for t in range(1, n):
        returns[t] = mu + phi * (returns[t - 1] - mu) + rng.normal(0, sigma)

    iid_point, iid_lo, iid_hi = iid_bootstrap_sharpe(returns, seed=0)
    stat_point, stat_lo, stat_hi = stationary_bootstrap_sharpe(returns, mean_block=20, seed=0)

    iid_width = iid_hi - iid_lo
    stat_width = stat_hi - stat_lo

    print(
        f"IID bootstrap:        point={iid_point:.3f}  "
        f"[{iid_lo:.3f}, {iid_hi:.3f}]  width={iid_width:.3f}"
    )
    print(
        f"Stationary bootstrap: point={stat_point:.3f}  "
        f"[{stat_lo:.3f}, {stat_hi:.3f}]  width={stat_width:.3f}"
    )

    assert stat_width > iid_width, (
        f"stationary bootstrap width {stat_width:.4f} not wider than iid width {iid_width:.4f}"
    )


def test_dsr_decreases_monotonically_with_n_trials():
    n_trials_list = [1, 3, 5, 7, 11, 21, 51, 101]
    dsrs = [
        deflated_sharpe_ratio(sharpe=0.05, n_trials=n, skew=-0.3, kurtosis=4.5, n_obs=1250)
        for n in n_trials_list
    ]
    assert all(a > b for a, b in zip(dsrs, dsrs[1:], strict=False)), (
        f"DSR not strictly decreasing: {dsrs}"
    )


def test_dsr_of_very_high_sharpe_at_one_trial_is_near_one():
    dsr = deflated_sharpe_ratio(sharpe=0.5, n_trials=1, skew=0.0, kurtosis=3.0, n_obs=500)
    assert dsr > 0.999, f"expected DSR close to 1.0, got {dsr:.6f}"


def test_newey_west_se_exceeds_naive_se_under_autocorrelation():
    rng = np.random.default_rng(11)
    n = 2000
    phi = 0.8
    sigma = 0.01

    returns = np.empty(n)
    returns[0] = 0.0
    for t in range(1, n):
        returns[t] = phi * returns[t - 1] + rng.normal(0, sigma)

    naive_se = returns.std(ddof=1) / np.sqrt(n)
    nw_se = newey_west_se(returns)

    assert nw_se > naive_se, f"Newey-West SE {nw_se:.6f} not larger than naive SE {naive_se:.6f}"


def test_dsr_sensitive_to_skew():
    dsr_symmetric = deflated_sharpe_ratio(
        sharpe=0.06, n_trials=5, skew=0.0, kurtosis=3.0, n_obs=1000
    )
    dsr_neg_skew = deflated_sharpe_ratio(
        sharpe=0.06, n_trials=5, skew=-1.5, kurtosis=3.0, n_obs=1000
    )
    assert dsr_neg_skew < dsr_symmetric, (
        f"negatively skewed DSR ({dsr_neg_skew:.4f}) not lower than symmetric DSR "
        f"({dsr_symmetric:.4f})"
    )
