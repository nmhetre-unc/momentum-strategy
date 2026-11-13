// qbt._bootstrap_ext: the stationary-bootstrap index generator in C++.
//
// Must reproduce qbt.bootstrap's pure-Python reference bit-for-bit. It
// does so by drawing from the caller's own numpy Generator. The bitgen_t*
// comes out of `rng.bit_generator.capsule`, and the two distributions are
// numpy's own C routines from libnpyrandom (linked from the build-time
// numpy), called exactly as Generator.integers(0, n) and
// Generator.geometric(p) call them for a scalar draw:
//   integers(0, n) -> random_bounded_uint64_fill(state, off=0, rng=n-1, cnt=1,
//                                                use_masked=false, &out)
//   geometric(p)   -> random_geometric(state, p)
// Per block, one start then one length, in that order: the same stream
// consumption as the reference loop.
//
// Memory layout. `out` is a caller-allocated, C-contiguous (rows, n) int64
// array; row r occupies out[r*n .. r*n + n). Each block is written as a
// sequential run of `take` indices at the row's fill position, wrapping
// from n-1 to 0 by compare-and-reset rather than a per-element modulo. The
// last block is truncated to the space left in the row. Nothing is
// allocated after the argument checks: no vectors, no temporaries, no
// Python objects. The buffer is written in place and nothing is returned.
//
// Locking. numpy's own scalar draws run under `with bit_generator.lock`
// (a threading.Lock) with the GIL released. This does the same, but once
// for the whole fill rather than once per draw. It acquires the lock with
// the GIL held, releases the GIL for the loop (which touches no Python
// objects), then re-takes the GIL and releases the lock. RAII destruction
// order guarantees that sequence even if something throws.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <utility>

extern "C" {
#include "numpy/random/distributions.h"
}

#ifndef QBT_NUMPY_BUILD_VERSION
#error "QBT_NUMPY_BUILD_VERSION must be defined by the build (the numpy version compiled against)"
#endif
#ifndef QBT_COMPILER
#define QBT_COMPILER "unknown"
#endif

namespace py = pybind11;

namespace {

bitgen_t *bitgen_from(const py::object &bit_generator) {
    py::object capsule = bit_generator.attr("capsule");
    void *ptr = PyCapsule_GetPointer(capsule.ptr(), "BitGenerator");
    if (ptr == nullptr) {
        throw py::error_already_set();
    }
    return static_cast<bitgen_t *>(ptr);
}

// Holds a Python threading.Lock for its lifetime. Constructed and
// destroyed with the GIL held.
class LockHold {
  public:
    explicit LockHold(py::object lock) : lock_(std::move(lock)) { lock_.attr("acquire")(); }
    ~LockHold() {
        try {
            lock_.attr("release")();
        } catch (py::error_already_set &err) {
            err.discard_as_unraisable("qbt._bootstrap_ext: releasing bit_generator.lock");
        }
    }
    LockHold(const LockHold &) = delete;
    LockHold &operator=(const LockHold &) = delete;

  private:
    py::object lock_;
};

void fill_rows(bitgen_t *state, int64_t *data, int64_t rows, int64_t n, double p) {
    const auto bound = static_cast<uint64_t>(n - 1);
    for (int64_t r = 0; r < rows; ++r) {
        int64_t *row = data + r * n;
        int64_t filled = 0;
        while (filled < n) {
            uint64_t start = 0;
            random_bounded_uint64_fill(state, 0, bound, 1, false, &start);
            const int64_t length = random_geometric(state, p);
            const int64_t remaining = n - filled;
            const int64_t take = length < remaining ? length : remaining;
            auto pos = static_cast<int64_t>(start);
            for (int64_t j = 0; j < take; ++j) {
                row[filled + j] = pos;
                if (++pos == n) {
                    pos = 0;
                }
            }
            // The reference adds the untruncated length; `filled < n` is
            // the same test as its `total < n` either way.
            filled += take;
        }
    }
}

void fill_stationary_indices(const py::object &rng,
                             py::array_t<int64_t, py::array::c_style> out, double p) {
    if (out.ndim() != 2) {
        throw py::value_error("out must be a 2-D (rows, n) int64 array");
    }
    const int64_t rows = out.shape(0);
    const int64_t n = out.shape(1);
    if (n < 1) {
        throw py::value_error("out must have at least one column (n >= 1)");
    }
    if (!(p > 0.0 && p <= 1.0) || !std::isfinite(p)) {
        throw py::value_error("p must satisfy 0 < p <= 1");
    }
    py::object bit_generator = rng.attr("bit_generator");
    bitgen_t *state = bitgen_from(bit_generator);
    int64_t *data = out.mutable_data();  // raises if `out` is read-only

    LockHold hold(bit_generator.attr("lock"));
    py::gil_scoped_release nogil;
    fill_rows(state, data, rows, n, p);
}

}  // namespace

PYBIND11_MODULE(_bootstrap_ext, m) {
    m.doc() = "C++ stationary-bootstrap index generator; see qbt.bootstrap.";
    m.attr("NUMPY_BUILD_VERSION") = QBT_NUMPY_BUILD_VERSION;
    m.attr("COMPILER") = QBT_COMPILER;  // recorded for benchmark provenance
    m.def("fill_stationary_indices", &fill_stationary_indices, py::arg("rng"),
          py::arg("out").noconvert(), py::arg("p"),
          "Fill every row of `out` (C-contiguous (rows, n) int64) with one stationary-"
          "bootstrap replicate drawn from `rng`, holding rng.bit_generator.lock throughout.");
}
