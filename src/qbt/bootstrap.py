"""
Resampling indices for the Politis-Romano stationary bootstrap, shared by
every block-bootstrap resampler in the project (qbt.stats and
scripts/regenerate_results.py) so they all consume the random stream in
exactly the same order.

This pure-Python version is the reference implementation: anything
faster must reproduce its output bit-for-bit for the same Generator
state, not merely match it statistically.
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


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float,
                                 rng: np.random.Generator) -> Iterator[np.ndarray]:
    """
    Yields `n_boot` int64 index arrays of length `n`, one per bootstrap
    replicate, for resampling a length-`n` series. Each replicate is built
    from blocks: a uniform start rng.integers(0, n), then a geometric
    length rng.geometric(1 / mean_block), drawn in that order, block after
    block, until at least `n` indices exist. Blocks wrap around past the
    end of the series, and the last block is truncated to fit.

    Validates eagerly: a bad `mean_block` raises here, before the first
    replicate is requested. Draws lazily: replicate i is drawn only when
    it's requested, so callers sharing `rng` with other work see the same
    stream position as the original inline loop.
    """
    validate_mean_block(n, mean_block)
    return _replicates(n, n_boot, 1.0 / mean_block, rng)


def _replicates(n: int, n_boot: int, p: float, rng: np.random.Generator) -> Iterator[np.ndarray]:
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
