"""
Micro-benchmarks for the stationary bootstrap at the project's real size:
n=4278 (SPY 2008-2025 daily returns), mean_block=20, n_boot=2000.

Not collected by a bare `pytest` (testpaths = ["tests"]). Run explicitly:

    pytest benchmarks/bench_bootstrap.py --benchmark-only \
        --benchmark-json=benchmarks/results/bootstrap_micro.json

Groups:
  replicate         one replicate: next(indices) + values[idx] + sharpe_ratio,
                    from a long-lived iterator (the marginal cost per replicate)
  replicate-indices one replicate's indices only
  full-call         stationary_bootstrap_sharpe(returns, 2000, 20)
  index-matrix      all 2000 replicates' indices, comparing impl="cpp"'s
                    one-call-per-replicate design with one batched fill call
The return series is synthetic: timing doesn't depend on the values.
"""

import numpy as np
import pytest

from qbt import bootstrap
from qbt.analytics import sharpe_ratio
from qbt.bootstrap import stationary_bootstrap_indices
from qbt.stats import stationary_bootstrap_sharpe

N, MEAN_BLOCK, N_BOOT = 4278, 20, 2000
IMPLS = ["python", "numpy", "cpp"]
FULL_CALL_ROUNDS = {"python": 15, "numpy": 25, "cpp": 60}
REPLICATE_ROUNDS = 3000


@pytest.fixture(scope="module", autouse=True)
def _require_extension():
    if not bootstrap.cpp_available():
        pytest.exit(f"C++ extension unavailable: {bootstrap._EXT_UNAVAILABLE}", returncode=2)


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    return np.random.default_rng(2008).normal(3e-4, 0.012, N)


def _endless(impl: str):
    return stationary_bootstrap_indices(N, 10**12, MEAN_BLOCK, np.random.default_rng(0), impl=impl)


@pytest.mark.parametrize("impl", IMPLS)
def test_replicate(benchmark, impl, returns):
    benchmark.group = "replicate"
    it = _endless(impl)
    benchmark.pedantic(lambda: sharpe_ratio(returns[next(it)]),
                       rounds=REPLICATE_ROUNDS, iterations=1, warmup_rounds=100)


@pytest.mark.parametrize("impl", IMPLS)
def test_replicate_indices(benchmark, impl):
    benchmark.group = "replicate-indices"
    it = _endless(impl)
    benchmark.pedantic(lambda: next(it), rounds=REPLICATE_ROUNDS, iterations=1,
                       warmup_rounds=100)


@pytest.mark.parametrize("impl", IMPLS)
def test_full_call(benchmark, impl, returns):
    benchmark.group = "full-call"
    result = benchmark.pedantic(
        stationary_bootstrap_sharpe, args=(returns,),
        kwargs={"n_boot": N_BOOT, "mean_block": MEAN_BLOCK, "seed": 0, "impl": impl},
        rounds=FULL_CALL_ROUNDS[impl], iterations=1, warmup_rounds=1,
    )
    # Same answer from every implementation, or the timing is meaningless.
    assert result == stationary_bootstrap_sharpe(returns, n_boot=N_BOOT, mean_block=MEAN_BLOCK,
                                                 seed=0, impl="python")


def test_index_matrix_per_replicate_calls(benchmark):
    """impl="cpp" as shipped: one extension call (and one row allocation) per replicate."""
    benchmark.group = "index-matrix"

    def run():
        for _ in stationary_bootstrap_indices(N, N_BOOT, MEAN_BLOCK,
                                              np.random.default_rng(0), impl="cpp"):
            pass

    benchmark.pedantic(run, rounds=60, iterations=1, warmup_rounds=2)


def test_index_matrix_batched_alloc_in_call(benchmark):
    """One fill call for all rows, including allocating the (n_boot, n) buffer."""
    benchmark.group = "index-matrix"
    ext = bootstrap._EXT
    assert ext is not None

    def run():
        out = np.empty((N_BOOT, N), dtype=np.int64)
        ext.fill_stationary_indices(np.random.default_rng(0), out, 1 / MEAN_BLOCK)

    benchmark.pedantic(run, rounds=60, iterations=1, warmup_rounds=2)


def test_index_matrix_batched_preallocated(benchmark):
    """One fill call for all rows into a buffer allocated (and touched) beforehand."""
    benchmark.group = "index-matrix"
    ext = bootstrap._EXT
    assert ext is not None
    out = np.zeros((N_BOOT, N), dtype=np.int64)
    benchmark.pedantic(
        lambda: ext.fill_stationary_indices(np.random.default_rng(0), out, 1 / MEAN_BLOCK),
        rounds=60, iterations=1, warmup_rounds=2,
    )
    rows = list(stationary_bootstrap_indices(N, N_BOOT, MEAN_BLOCK, np.random.default_rng(0),
                                             impl="python"))
    assert np.array_equal(out, np.stack(rows))
