# Type stub for the C++ extension built from src/cpp/bootstrap_ext.cpp.
# `out` must be a writable, C-contiguous (rows, n) int64 array; it's
# filled in place.
import numpy as np
import numpy.typing as npt

NUMPY_BUILD_VERSION: str
COMPILER: str

def fill_stationary_indices(
    rng: np.random.Generator, out: npt.NDArray[np.int64], p: float
) -> None: ...
