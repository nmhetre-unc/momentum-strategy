"""
Interleaved micro-benchmarks: the source of the micro numbers in the
results table. bench_bootstrap.py (pytest-benchmark) runs each
implementation's rounds back to back, so if the CPU's speed drifts between
tests, a ratio between them drifts too. Here every repeat measures every
implementation once, in an order that rotates each repeat, and speedups are
computed as paired ratios within each repeat before being summarized.

    python benchmarks/micro_interleaved.py --out benchmarks/results/micro_interleaved.json

Benchmarks (n=4278, mean_block=20, n_boot=2000, synthetic returns):
  replicate          K consecutive replicates (indices + values[idx] +
                     sharpe_ratio) from a long-lived iterator, timed
                     together; reported per replicate
  replicate_indices  the same, indices only
  full_call          stationary_bootstrap_sharpe(returns, 2000, 20)
  index_matrix       all 2000 replicates' indices: impl="cpp" as shipped
                     (one call per replicate) vs one batched fill call,
                     with and without the buffer allocation
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import machine  # noqa: E402

from qbt import bootstrap  # noqa: E402
from qbt.analytics import sharpe_ratio  # noqa: E402
from qbt.bootstrap import stationary_bootstrap_indices  # noqa: E402
from qbt.stats import stationary_bootstrap_sharpe  # noqa: E402

N, MEAN_BLOCK, N_BOOT = 4278, 20, 2000
IMPLS = ("python", "numpy", "cpp")


def describe(xs: list[float]) -> dict:
    q1, med, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return {"median": med, "q1": q1, "q3": q3, "iqr": q3 - q1,
            "min": min(xs), "max": max(xs), "n": len(xs)}


def rotated(items, repeat: int):
    k = repeat % len(items)
    return items[k:] + items[:k]


def run_interleaved(variants: dict, repeats: int) -> dict:
    """variants: name -> zero-arg callable returning seconds for one measurement."""
    names = list(variants)
    times: dict[str, list[float]] = {name: [] for name in names}
    for r in range(repeats):
        for name in rotated(names, r):
            times[name].append(variants[name]())
    return times


def timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def paired_ratios(times: dict, slow: str, fast: str) -> dict:
    return describe([s / f for s, f in zip(times[slow], times[fast], strict=True)])


def main(out: Path, repeats_full: int, repeats_rep: int, k: int) -> None:
    if not bootstrap.cpp_available():
        sys.exit(f"C++ extension unavailable: {bootstrap._EXT_UNAVAILABLE}")
    info_start = machine.machine_info()
    returns = np.random.default_rng(2008).normal(3e-4, 0.012, N)
    ext = bootstrap._EXT
    assert ext is not None

    # Correctness first: every implementation gives the reference answer.
    ref = stationary_bootstrap_sharpe(returns, n_boot=N_BOOT, mean_block=MEAN_BLOCK, impl="python")
    for impl in IMPLS[1:]:
        got = stationary_bootstrap_sharpe(returns, n_boot=N_BOOT, mean_block=MEAN_BLOCK, impl=impl)
        assert got == ref, (impl, got, ref)

    results: dict = {}

    iters = {impl: stationary_bootstrap_indices(N, 10**12, MEAN_BLOCK, np.random.default_rng(0),
                                                impl=impl) for impl in IMPLS}

    def replicate_block(impl: str, with_sharpe: bool):
        it = iters[impl]
        if with_sharpe:
            return lambda: timed(lambda: [sharpe_ratio(returns[next(it)]) for _ in range(k)]) / k
        return lambda: timed(lambda: [next(it) for _ in range(k)]) / k

    for name, with_sharpe in (("replicate", True), ("replicate_indices", False)):
        for impl in IMPLS:  # warm-up
            replicate_block(impl, with_sharpe)()
        times = run_interleaved({i: replicate_block(i, with_sharpe) for i in IMPLS}, repeats_rep)
        results[name] = times

    def full_call(impl: str):
        return lambda: timed(lambda: stationary_bootstrap_sharpe(
            returns, n_boot=N_BOOT, mean_block=MEAN_BLOCK, seed=0, impl=impl))

    for impl in IMPLS:
        full_call(impl)()
    results["full_call"] = run_interleaved({i: full_call(i) for i in IMPLS}, repeats_full)

    prealloc = np.zeros((N_BOOT, N), dtype=np.int64)

    def per_replicate_calls():
        for _ in stationary_bootstrap_indices(N, N_BOOT, MEAN_BLOCK, np.random.default_rng(0),
                                              impl="cpp"):
            pass

    def batched_alloc():
        out = np.empty((N_BOOT, N), dtype=np.int64)
        ext.fill_stationary_indices(np.random.default_rng(0), out, 1 / MEAN_BLOCK)

    matrix_variants = {
        "cpp_per_replicate_calls": lambda: timed(per_replicate_calls),
        "cpp_batched_alloc_in_call": lambda: timed(batched_alloc),
        "cpp_batched_preallocated": lambda: timed(lambda: ext.fill_stationary_indices(
            np.random.default_rng(0), prealloc, 1 / MEAN_BLOCK)),
    }
    for fn in matrix_variants.values():
        fn()
    results["index_matrix"] = run_interleaved(matrix_variants, repeats_rep)

    summary: dict = {}
    for bench, times in results.items():
        summary[bench] = {name: describe(ts) for name, ts in times.items()}
        if bench != "index_matrix":
            summary[bench]["ratio_numpy_over_cpp"] = paired_ratios(times, "numpy", "cpp")
            summary[bench]["ratio_python_over_cpp"] = paired_ratios(times, "python", "cpp")
            summary[bench]["ratio_python_over_numpy"] = paired_ratios(times, "python", "numpy")
    m = results["index_matrix"]
    summary["index_matrix"]["per_replicate_overhead_vs_preallocated_s"] = describe(
        [a - b for a, b in zip(m["cpp_per_replicate_calls"], m["cpp_batched_preallocated"],
                               strict=True)])

    doc = {
        "benchmark": "stationary bootstrap micro-benchmarks, interleaved",
        "params": {"n": N, "mean_block": MEAN_BLOCK, "n_boot": N_BOOT,
                   "replicates_per_measurement": k, "repeats_replicate": repeats_rep,
                   "repeats_full_call": repeats_full,
                   "order": "every repeat runs every variant once; start rotates per repeat",
                   "ratios": "paired within each repeat, then summarized"},
        "units": "seconds (replicate benchmarks: per replicate)",
        "machine_start": info_start,
        "machine_end": machine.machine_info(),
        "summary": summary,
        "raw": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    for bench, s in summary.items():
        for name, d in s.items():
            print(f"{bench:18} {name:32} median {d['median']:.6g}  IQR {d['iqr']:.3g}  "
                  f"[{d['min']:.6g}, {d['max']:.6g}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "results/micro_interleaved.json")
    ap.add_argument("--repeats-full", type=int, default=20)
    ap.add_argument("--repeats-replicate", type=int, default=60)
    ap.add_argument("--k", type=int, default=50, help="replicates per replicate measurement")
    args = ap.parse_args()
    main(args.out, args.repeats_full, args.repeats_replicate, args.k)
