"""
Confirms the stationary bootstrap accounts for autocorrelation that the
naive i.i.d. bootstrap ignores.
"""

import numpy as np

from qbt.stats import iid_bootstrap_sharpe, stationary_bootstrap_sharpe


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

    print(f"IID bootstrap:        point={iid_point:.3f}  [{iid_lo:.3f}, {iid_hi:.3f}]  width={iid_width:.3f}")
    print(f"Stationary bootstrap: point={stat_point:.3f}  [{stat_lo:.3f}, {stat_hi:.3f}]  width={stat_width:.3f}")

    assert stat_width > iid_width, (
        f"stationary bootstrap width {stat_width:.4f} not wider than iid width {iid_width:.4f}"
    )
