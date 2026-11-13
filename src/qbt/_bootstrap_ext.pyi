# Type stub for the C++ extension built from src/cpp/bootstrap_ext.cpp.
# `out` must be a writable, C-contiguous (rows, n) int64 array; it's
# filled in place. (np.ndarray rather than npt.NDArray[np.int64]: mypy
# 1.10 can't resolve numpy 2.x's NDArray alias.)
import numpy as np

NUMPY_BUILD_VERSION: str

def fill_stationary_indices(rng: np.random.Generator, out: np.ndarray, p: float) -> None: ...
