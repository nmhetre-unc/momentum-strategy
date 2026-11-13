"""
Resampling indices for the Politis-Romano stationary bootstrap, shared by
every block-bootstrap resampler in the project (qbt.stats and
scripts/regenerate_results.py) so they all consume the random stream in
exactly the same order.

The pure-Python implementation (impl="python") is the reference: every
faster implementation must reproduce its output bit-for-bit for the same
Generator state, and leave the Generator in the same state afterwards,
not merely match it statistically.

The C++ implementation (impl="cpp", the qbt._bootstrap_ext extension)
draws through numpy's own C distribution routines, linked from the numpy
it was built against. numpy doesn't promise an identical stream across
releases, so the extension is only used when the installed numpy is that
same version: impl="auto" falls back to "numpy" with a
BootstrapFallbackWarning otherwise, and an explicit impl="cpp" raises.
"""

import warnings
from collections.abc import Iterator
from types import ModuleType

import numpy as np

IMPLEMENTATIONS = ("python", "numpy", "cpp")


class BootstrapFallbackWarning(RuntimeWarning):
    """impl="auto" couldn't use the C++ extension and fell back to NumPy."""


def _load_extension() -> tuple[ModuleType | None, str | None]:
    """(module, None) if the C++ extension is usable, else (None, reason)."""
    try:
        from qbt import _bootstrap_ext
    except (ImportError, OSError) as exc:  # OSError: a DLL blocked or unloadable
        return None, f"the C++ extension qbt._bootstrap_ext is not available ({exc})"
    built = _bootstrap_ext.NUMPY_BUILD_VERSION
    if built != np.__version__:
        return None, (
            f"qbt._bootstrap_ext was built against numpy {built}, but numpy "
            f"{np.__version__} is installed, and the random stream is only guaranteed "
            "identical within one numpy release; rebuild the extension "
            "(pip install -e .) against the installed numpy"
        )
    return _bootstrap_ext, None


_EXT, _EXT_UNAVAILABLE = _load_extension()


def cpp_available() -> bool:
    """True if impl="cpp" (and therefore impl="auto" -> "cpp") is usable."""
    return _EXT is not None


def resolve_impl(impl: str) -> str:
    """
    Maps `impl` to a concrete implementation name. "auto" is "cpp" when
    the extension is usable, else "numpy" plus a BootstrapFallbackWarning
    saying why. An explicit "cpp" that isn't usable raises RuntimeError
    rather than silently running something else.
    """
    if impl == "auto":
        if _EXT is not None:
            return "cpp"
        warnings.warn(
            f"stationary bootstrap impl='auto' is using the NumPy implementation: "
            f"{_EXT_UNAVAILABLE}.",
            BootstrapFallbackWarning, stacklevel=3,
        )
        return "numpy"
    if impl == "cpp" and _EXT is None:
        raise RuntimeError(f"impl='cpp' was requested, but {_EXT_UNAVAILABLE}.")
    if impl not in IMPLEMENTATIONS:
        raise ValueError(f"Unknown impl {impl!r}; choose from {IMPLEMENTATIONS + ('auto',)}.")
    return impl


def validate_mean_block(n: int, mean_block: float) -> None:
    """
    Raises ValueError unless 1 <= mean_block <= n. Below 1, the geometric
    block-length parameter 1/mean_block exceeds 1 and is invalid. Above n,
    the expected block is longer than the series itself, so the bootstrap
    degenerates into rotations of the original series. At extreme values
    (p = 1/mean_block below ~1e-15) numpy's geometric draw also saturates
    at INT64_MAX, and the block it implies can't be allocated.
    """
    if n < 1:
        raise ValueError(f"Stationary bootstrap needs at least 1 observation; got n={n}.")
    if not 1 <= mean_block <= n:
        raise ValueError(
            f"mean_block must satisfy 1 <= mean_block <= n (n={n}); got {mean_block!r}."
        )


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float,
                                 rng: np.random.Generator,
                                 impl: str = "auto") -> Iterator[np.ndarray]:
    """
    Yields `n_boot` int64 index arrays of length `n`, one per bootstrap
    replicate, for resampling a length-`n` series. Each replicate is built
    from blocks: a uniform start rng.integers(0, n), then a geometric
    length rng.geometric(1 / mean_block), drawn in that order, block after
    block, until at least `n` indices exist. Blocks wrap around past the
    end of the series, and the last block is truncated to fit.

    `impl` picks the implementation; every one yields bit-identical
    arrays and leaves `rng` in the same state:
      "python" -- the reference: builds each block as its own array.
      "numpy"  -- same scalar draws in the same order, but builds a
                  replicate's indices in one vectorized step. The scalar
                  draws themselves can't be batched without reordering
                  the stream, so they remain a Python loop.
      "cpp"    -- the whole replicate (every draw and every index) in one
                  C++ call; see src/cpp/bootstrap_ext.cpp.
      "auto"   -- (default) "cpp" if usable, else "numpy"; see resolve_impl.

    Validates eagerly: a bad `mean_block` or `impl` raises here (and
    "auto" warns here), before the first replicate is requested. Draws lazily: replicate i is drawn
    only when it's requested, so callers sharing `rng` with other work see
    the same stream position as the original inline loop.
    """
    validate_mean_block(n, mean_block)
    resolved = resolve_impl(impl)
    p = 1.0 / mean_block
    if resolved == "cpp":
        return _replicates_cpp(n, n_boot, p, rng)
    if resolved == "numpy":
        return _replicates_numpy(n, n_boot, p, rng)
    return _replicates_python(n, n_boot, p, rng)


def _replicates_cpp(n: int, n_boot: int, p: float,
                    rng: np.random.Generator) -> Iterator[np.ndarray]:
    # One C++ call per replicate, not one for all n_boot: drawing ahead
    # would leave `rng` past where the reference leaves it whenever a
    # caller stops early, breaking the lazy-draw guarantee. Each call
    # holds bit_generator.lock across that replicate's whole block loop
    # and writes into the row allocated for it here.
    assert _EXT is not None  # guaranteed by resolve_impl
    fill = _EXT.fill_stationary_indices
    for _ in range(n_boot):
        out = np.empty((1, n), dtype=np.int64)
        fill(rng, out, p)
        yield out[0]


def _replicates_numpy(n: int, n_boot: int, p: float,
                      rng: np.random.Generator) -> Iterator[np.ndarray]:
    integers, geometric = rng.integers, rng.geometric
    positions = np.arange(n)
    for _ in range(n_boot):
        starts: list[np.int64] = []
        lengths: list[int] = []
        total = 0
        while total < n:
            starts.append(integers(0, n))
            length = geometric(p)
            lengths.append(length)
            total += length
        # Truncate the last block so the replicate is exactly n long;
        # it still has length >= 1, since total was < n before it.
        lengths[-1] -= total - n
        length_arr = np.asarray(lengths, dtype=np.int64)
        block_offset = np.subtract(np.cumsum(length_arr), length_arr)
        # Position j inside block k maps to starts[k] + (j - block_offset[k]),
        # the same value the reference computes as start + arange(length).
        # (Ufuncs rather than operators for the same mypy reason as below.)
        shift = np.repeat(np.subtract(np.asarray(starts, dtype=np.int64), block_offset),
                          length_arr)
        yield np.remainder(np.add(shift, positions), n)


def _replicates_python(n: int, n_boot: int, p: float,
                       rng: np.random.Generator) -> Iterator[np.ndarray]:
    for _ in range(n_boot):
        pieces: list[np.ndarray] = []
        total = 0
        while total < n:
            start = rng.integers(0, n)
            length = rng.geometric(p)
            # np.remainder is what `%` dispatches to; spelled out because
            # mypy 1.10 can't resolve numpy 2.x's 1-D array stubs and
            # mistypes `start + arange` as a scalar.
            pieces.append(np.remainder(start + np.arange(length), n))
            total += length
        replicate: np.ndarray = np.concatenate(pieces)
        yield replicate[:n]
