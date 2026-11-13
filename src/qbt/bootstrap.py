"""
Resampling indices for the Politis-Romano stationary bootstrap, shared by
every block-bootstrap resampler in the project (qbt.stats and
scripts/regenerate_results.py) so they all consume the random stream in
exactly the same order.

The pure-Python implementation (impl="python") is the reference: every
faster implementation must reproduce its output bit-for-bit for the same
Generator state, and leave the Generator in the same state afterwards,
not merely match it statistically.
"""

from collections.abc import Iterator

import numpy as np


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


IMPLEMENTATIONS = ("python", "numpy")


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float,
                                 rng: np.random.Generator,
                                 impl: str = "python") -> Iterator[np.ndarray]:
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

    Validates eagerly: a bad `mean_block` or `impl` raises here, before
    the first replicate is requested. Draws lazily: replicate i is drawn
    only when it's requested, so callers sharing `rng` with other work see
    the same stream position as the original inline loop.
    """
    validate_mean_block(n, mean_block)
    if impl == "python":
        return _replicates_python(n, n_boot, 1.0 / mean_block, rng)
    if impl == "numpy":
        return _replicates_numpy(n, n_boot, 1.0 / mean_block, rng)
    raise ValueError(f"Unknown impl {impl!r}; choose from {IMPLEMENTATIONS}.")


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
