"""
Pins the stationary bootstrap's exact outputs for fixed seeds, so any
reimplementation of the index generator (vectorized, compiled) is held to
bit-identical results rather than "statistically similar" ones.

The pinned values were captured from the original inline loops in
qbt.stats.stationary_bootstrap_sharpe and
scripts/regenerate_results.paired_stationary_bootstrap, before they were
refactored onto qbt.bootstrap. Comparisons use == on floats, not
approx. The pins are tied to numpy's Generator stream (PCG64 with its
integers/geometric/normal algorithms), which numpy doesn't promise to keep
stable across releases; if a numpy upgrade breaks every pin at once,
that's the cause, not this code.
"""

import hashlib
import importlib.util
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import qbt
import qbt.bootstrap as qb
from qbt.bootstrap import IMPLEMENTATIONS, stationary_bootstrap_indices, validate_mean_block
from qbt.stats import stationary_bootstrap_sharpe

# CI sets QBT_REQUIRE_CPP=1, so a missing extension fails there instead of
# silently skipping every "cpp" case.
REQUIRE_CPP = os.environ.get("QBT_REQUIRE_CPP") == "1"
_skip_cpp = pytest.mark.skipif(
    not qb.cpp_available() and not REQUIRE_CPP,
    reason=f"C++ extension unavailable: {qb._EXT_UNAVAILABLE}",
)
IMPLS = [pytest.param(i, marks=_skip_cpp) if i == "cpp" else i for i in IMPLEMENTATIONS]
FAST_IMPLS = [i for i in IMPLS if i != "python"]

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "regenerate_results.py"


def _series(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(3e-4, 0.012, n)


@pytest.fixture(scope="module")
def regenerate_results():
    spec = importlib.util.spec_from_file_location("regenerate_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# (n, mean_block, n_boot, seed) -> (point, lo, hi); series is _series(n, 100 + seed).
SINGLE_PINS = [
    ((4278, 20, 2000, 0), (0.6806712175873201, 0.312081435638621, 1.068002028247365)),
    ((4278, 10, 500, 1), (0.10371185354460688, -0.28686181848499037, 0.495582429967299)),
    ((4278, 40, 500, 2), (0.0074667193268161515, -0.3929817727277836, 0.4070834483394127)),
    ((126, 20, 200, 3), (1.1175751539967542, -0.6860926831144191, 2.6141799133886736)),
    ((21, 20, 200, 4), (7.8810548101958835, 5.6095890275504114, 10.641669622336407)),
    ((1000, 1, 300, 5), (0.8254465877767547, 0.05524143019137032, 1.6140263262162644)),
    ((50, 50, 300, 6), (-1.6161511067318035, -3.628568899275504, 0.22399978044224156)),
]

# (n, mean_block, n_boot, seed) -> (strat_lo, strat_hi, diff_point, diff_lo, diff_hi);
# series are _series(n, 200 + seed) and _series(n, 300 + seed).
PAIRED_PINS = [
    ((4278, 20, 2000, 0), (-0.03788956171769544, 0.7065826099823561, -0.24279399606115903,
                           -0.8248473489706938, 0.3357165681927548)),
    ((126, 20, 200, 3), (-0.8106676300344645, 2.987422027697513, 1.7135323122489132,
                         -0.25136173295969055, 3.7333530499882186)),
    ((1000, 5, 300, 7), (-0.4328485601539149, 1.5085936451242432, -0.0715069948253802,
                         -1.2786676228782132, 1.1490425758008322)),
]

# (n, mean_block, n_boot, seed) -> sha256 of the concatenated int64 index rows.
INDEX_PINS = [
    ((4278, 20, 50, 0), "a4f9e090eb0c5cafe52c7c7291f22983d69a9dfdcb7fc3145e99495157f77b82"),
    ((1, 1, 20, 1), "b393978842a0fa3d3e1470196f098f473f9678e72463cb65ec4ab5581856c2e4"),
    ((2, 2, 20, 2), "920ab3aecd4b6b9126f3d6cd3ea405d80f7ba961439f72998f13a08ff4ab89c0"),
    ((126, 20, 100, 3), "43e6af93111569db030eb647657e7b84b717776b26abba9d91707502153998f1"),
    ((300, 300, 20, 4), "69ad819d3626cf1b307b178069fa56a8b9539902b0da701ce7e849d0655323ff"),
]


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize(("args", "expected"), SINGLE_PINS)
def test_stationary_bootstrap_sharpe_pinned(args, expected, impl):
    n, mean_block, n_boot, seed = args
    result = stationary_bootstrap_sharpe(
        _series(n, 100 + seed), n_boot=n_boot, mean_block=mean_block, seed=seed, impl=impl
    )
    assert tuple(float(x) for x in result) == expected


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize(("args", "expected"), PAIRED_PINS)
def test_paired_stationary_bootstrap_pinned(regenerate_results, args, expected, impl):
    n, mean_block, n_boot, seed = args
    result = regenerate_results.paired_stationary_bootstrap(
        _series(n, 200 + seed), _series(n, 300 + seed),
        n_boot=n_boot, mean_block=mean_block, seed=seed, impl=impl,
    )
    assert tuple(float(x) for x in result) == expected


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize(("args", "expected"), INDEX_PINS)
def test_index_stream_pinned(args, expected, impl):
    n, mean_block, n_boot, seed = args
    digest = hashlib.sha256()
    rows = stationary_bootstrap_indices(
        n, n_boot, mean_block, np.random.default_rng(seed), impl=impl
    )
    for row in rows:
        assert row.dtype == np.int64 and row.shape == (n,)
        assert row.min() >= 0 and row.max() < n
        digest.update(row.tobytes())
    assert digest.hexdigest() == expected


def test_unknown_impl_raises():
    with pytest.raises(ValueError, match="Unknown impl"):
        stationary_bootstrap_indices(10, 1, 2, np.random.default_rng(0), impl="fortran")


def test_indices_are_wrapping_runs():
    # Within a replicate, each step either continues a block (i -> i+1 mod n)
    # or starts a new one; with mean_block=n there are few block starts.
    n = 200
    rows = list(stationary_bootstrap_indices(n, 50, n, np.random.default_rng(0)))
    steps = np.concatenate([np.diff(r) % n for r in rows])
    assert (steps == 1).mean() > 0.95


@pytest.mark.parametrize("impl", IMPLS)
def test_generator_draws_lazily(impl):
    # Consuming k replicates must leave the Generator exactly where the
    # original inline loop would after k iterations, not n_boot.
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    rows = stationary_bootstrap_indices(100, 1000, 10, a, impl=impl)
    first = [next(rows) for _ in range(3)]
    expected = list(stationary_bootstrap_indices(100, 3, 10, b))
    assert all(np.array_equal(x, y) for x, y in zip(first, expected, strict=True))
    assert a.bit_generator.state == b.bit_generator.state


@pytest.mark.parametrize("mean_block", [0, 0.5, -1, 101, 1e15, float("nan"), float("inf")])
def test_invalid_mean_block_raises(mean_block):
    with pytest.raises(ValueError, match="mean_block"):
        stationary_bootstrap_indices(100, 10, mean_block, np.random.default_rng(0))
    with pytest.raises(ValueError, match="mean_block"):
        stationary_bootstrap_sharpe(_series(100, 0), n_boot=10, mean_block=mean_block)


def test_invalid_mean_block_raises_before_drawing():
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(100, 10, 0, rng)
    assert rng.bit_generator.state == before


def test_empty_series_raises():
    with pytest.raises(ValueError, match="at least 1 observation"):
        validate_mean_block(0, 1)


@pytest.mark.parametrize(("n", "mean_block"), [(1, 1), (100, 1), (100, 100), (100, 99.5)])
def test_boundary_mean_block_accepted(n, mean_block):
    rows = list(stationary_bootstrap_indices(n, 5, mean_block, np.random.default_rng(0)))
    assert len(rows) == 5 and all(r.shape == (n,) for r in rows)


# --------------------------------------------------------------------------
# Equivalence: every implementation against the reference, over n,
# mean_block and seed. Bit-identical rows, and the same Generator state
# afterwards (so a caller sharing the Generator can't tell them apart).
# --------------------------------------------------------------------------
@st.composite
def bootstrap_case(draw, max_n: int):
    n = draw(st.integers(1, max_n))
    mean_block = draw(st.one_of(
        st.integers(1, n),
        st.floats(1.0, float(n), allow_nan=False, allow_infinity=False),
        st.sampled_from([1, n]),
    ))
    seed = draw(st.integers(0, 2**63 - 1))
    n_boot = draw(st.integers(0, 8))
    return n, mean_block, seed, n_boot


def _assert_matches_reference(n, mean_block, seed, n_boot, impl):
    ref_rng, rng = np.random.default_rng(seed), np.random.default_rng(seed)
    expected = list(stationary_bootstrap_indices(n, n_boot, mean_block, ref_rng, impl="python"))
    got = list(stationary_bootstrap_indices(n, n_boot, mean_block, rng, impl=impl))
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        assert g.dtype == e.dtype == np.int64
        assert np.array_equal(g, e)
    assert rng.bit_generator.state == ref_rng.bit_generator.state


@pytest.mark.parametrize("impl", FAST_IMPLS)
@settings(max_examples=300, deadline=None)
@given(case=bootstrap_case(max_n=300))
def test_impl_matches_reference_small_n(impl, case):
    _assert_matches_reference(*case, impl=impl)


@pytest.mark.parametrize("impl", FAST_IMPLS)
@settings(max_examples=40, deadline=None)
@given(case=bootstrap_case(max_n=5000))
def test_impl_matches_reference_large_n(impl, case):
    _assert_matches_reference(*case, impl=impl)


@pytest.mark.parametrize("impl", FAST_IMPLS)
@settings(max_examples=60, deadline=None)
@given(case=bootstrap_case(max_n=400), data_seed=st.integers(0, 2**32 - 1))
def test_sharpe_interval_matches_reference(impl, case, data_seed):
    n, mean_block, seed, n_boot = case
    returns = _series(n, data_seed)
    expected = stationary_bootstrap_sharpe(returns, n_boot=max(n_boot, 1),
                                           mean_block=mean_block, seed=seed, impl="python")
    got = stationary_bootstrap_sharpe(returns, n_boot=max(n_boot, 1),
                                      mean_block=mean_block, seed=seed, impl=impl)
    # == on floats, NaN-aware: a constant or length-1 series is legitimate input.
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


# --------------------------------------------------------------------------
# impl selection: "auto", the C++ extension's availability, and the
# numpy-version guard. Mismatches are simulated, not built.
# --------------------------------------------------------------------------
@pytest.fixture
def extension_unavailable(monkeypatch):
    """Puts qbt.bootstrap in the state it loads into when the extension is unusable."""
    reason = "simulated: extension unusable"
    monkeypatch.setattr(qb, "_EXT", None)
    monkeypatch.setattr(qb, "_EXT_UNAVAILABLE", reason)
    return reason


def _reference_rows(n=200, n_boot=5, mean_block=10, seed=0):
    return list(stationary_bootstrap_indices(n, n_boot, mean_block,
                                             np.random.default_rng(seed), impl="python"))


def test_extension_present_when_required():
    if not REQUIRE_CPP:
        pytest.skip("QBT_REQUIRE_CPP not set")
    assert qb.cpp_available(), qb._EXT_UNAVAILABLE


@_skip_cpp
def test_auto_uses_cpp_when_available():
    assert qb.resolve_impl("auto") == "cpp"
    assert qb.resolve_impl("cpp") == "cpp"


@pytest.mark.parametrize("impl", ["python", "numpy"])
def test_explicit_pure_impls_resolve_to_themselves(impl):
    assert qb.resolve_impl(impl) == impl


def test_auto_falls_back_to_numpy_with_warning(extension_unavailable):
    with pytest.warns(qb.BootstrapFallbackWarning, match=extension_unavailable):
        assert qb.resolve_impl("auto") == "numpy"
    with pytest.warns(qb.BootstrapFallbackWarning):
        rows = list(stationary_bootstrap_indices(200, 5, 10, np.random.default_rng(0)))
    assert all(np.array_equal(a, b) for a, b in zip(rows, _reference_rows(), strict=True))
    with pytest.warns(qb.BootstrapFallbackWarning):
        got = stationary_bootstrap_sharpe(_series(300, 1), n_boot=50)
    assert got == stationary_bootstrap_sharpe(_series(300, 1), n_boot=50, impl="python")


def test_explicit_cpp_raises_when_unavailable(extension_unavailable):
    with pytest.raises(RuntimeError, match=extension_unavailable):
        qb.resolve_impl("cpp")
    with pytest.raises(RuntimeError, match=extension_unavailable):
        stationary_bootstrap_indices(200, 5, 10, np.random.default_rng(0), impl="cpp")


def test_load_extension_rejects_numpy_version_mismatch(monkeypatch):
    if not qb.cpp_available():
        pytest.skip("needs a built extension to compare versions against")
    monkeypatch.setattr(np, "__version__", "0.0.0-simulated")
    ext, reason = qb._load_extension()
    assert ext is None
    assert reason is not None
    assert "built against numpy" in reason and "0.0.0-simulated" in reason


def test_load_extension_reports_missing_module(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.delattr(qbt, "_bootstrap_ext", raising=False)
    monkeypatch.setitem(sys.modules, "qbt._bootstrap_ext", None)
    ext, reason = qb._load_extension()
    assert ext is None
    assert reason is not None and "not available" in reason


def test_load_extension_accepts_matching_build():
    if not qb.cpp_available():
        pytest.skip("extension not built")
    ext, reason = qb._load_extension()
    assert ext is not None and reason is None
    assert np.__version__ == ext.NUMPY_BUILD_VERSION


# --------------------------------------------------------------------------
# The extension's own guards: it writes into a caller buffer, so it must
# reject anything it would otherwise silently copy or overrun.
# --------------------------------------------------------------------------
@pytest.fixture
def ext():
    if not qb.cpp_available():
        if REQUIRE_CPP:
            pytest.fail(f"QBT_REQUIRE_CPP=1 but {qb._EXT_UNAVAILABLE}")
        pytest.skip(f"C++ extension unavailable: {qb._EXT_UNAVAILABLE}")
    return qb._EXT


def test_ext_fills_several_rows_like_the_reference(ext):
    # The multi-row path (one call, one lock hold, n_boot rows) matches the
    # reference row for row and leaves the Generator in the same state.
    rng, ref_rng = np.random.default_rng(5), np.random.default_rng(5)
    out = np.empty((7, 300), dtype=np.int64)
    ext.fill_stationary_indices(rng, out, 1 / 12)
    expected = list(stationary_bootstrap_indices(300, 7, 12, ref_rng, impl="python"))
    assert np.array_equal(out, np.stack(expected))
    assert rng.bit_generator.state == ref_rng.bit_generator.state


@pytest.mark.parametrize("bad", [
    np.empty((2, 10), dtype=np.int32),          # wrong dtype: would be a converted copy
    np.empty((2, 10), dtype=np.float64),
    np.empty((10, 2), dtype=np.int64).T,        # not C-contiguous
])
def test_ext_rejects_buffers_it_would_copy(ext, bad):
    with pytest.raises(TypeError):
        ext.fill_stationary_indices(np.random.default_rng(0), bad, 0.1)


def test_ext_rejects_read_only_buffer(ext):
    out = np.zeros((2, 10), dtype=np.int64)
    out.flags.writeable = False
    with pytest.raises(ValueError):
        ext.fill_stationary_indices(np.random.default_rng(0), out, 0.1)
    assert not out.any()


@pytest.mark.parametrize("shape", [(10,), (1, 0), (2, 3, 4)])
def test_ext_rejects_bad_shapes(ext, shape):
    with pytest.raises(ValueError):
        ext.fill_stationary_indices(np.random.default_rng(0), np.empty(shape, np.int64), 0.1)


@pytest.mark.parametrize("p", [0.0, -0.1, 1.0000001, float("nan"), float("inf")])
def test_ext_rejects_bad_p(ext, p):
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    with pytest.raises(ValueError):
        ext.fill_stationary_indices(rng, np.empty((1, 10), np.int64), p)
    assert rng.bit_generator.state == before


def test_ext_rejects_non_generator(ext):
    with pytest.raises(AttributeError):
        ext.fill_stationary_indices(object(), np.empty((1, 10), np.int64), 0.1)


def test_ext_zero_rows_draws_nothing(ext):
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    ext.fill_stationary_indices(rng, np.empty((0, 10), np.int64), 0.1)
    assert rng.bit_generator.state == before


def test_ext_holds_and_releases_bit_generator_lock(ext):
    rng = np.random.default_rng(0)
    lock = rng.bit_generator.lock
    out = np.zeros((50, 2000), dtype=np.int64)
    done = threading.Event()

    def fill():
        ext.fill_stationary_indices(rng, out, 0.05)
        done.set()

    with lock:  # held by this thread: the fill must wait for it
        worker = threading.Thread(target=fill)
        worker.start()
        assert not done.wait(0.3), "fill ran without acquiring bit_generator.lock"
    worker.join(10)
    assert done.is_set()
    assert lock.acquire(blocking=False), "fill did not release bit_generator.lock"
    lock.release()
