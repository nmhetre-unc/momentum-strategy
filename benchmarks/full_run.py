"""
Wall time of scripts/regenerate_results.py with every stationary-bootstrap
call forced to one implementation, `--runs` times each, interleaved
(every run measures every implementation once, and the starting
implementation rotates per run) so slow drift on the machine is spread
across implementations rather than landing on one. Speedups are paired
ratios within each run.

Each run also times its own bootstrap calls (the script's
stationary_bootstrap_sharpe and paired_stationary_bootstrap, outermost
call only), so the bootstrap's share of the run is measured directly
rather than inferred.

Each run is a fresh subprocess: imports, data load and all, timed two
ways: main() alone, and the whole process. The summary.md each run
produces goes to a temp dir (never results/), and its sha256 is recorded:
every run of every implementation must produce the same file.

    python benchmarks/full_run.py --runs 3 --out benchmarks/results/full_run.json

Uses the normal data cache (data_cache/, gitignored); the first-ever run
downloads SPY 2008-2025 once, before any timing starts.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

IMPLS = ("python", "numpy", "cpp")


def child(impl: str, out_dir: str) -> None:
    """Runs regenerate_results.main() with `impl` forced; prints one JSON line."""
    t_import = time.perf_counter()
    import regenerate_results as rr

    import qbt.bootstrap
    import qbt.stats

    real = qbt.bootstrap.stationary_bootstrap_indices
    calls = {"n": 0}

    def forced(*args, **kwargs):
        calls["n"] += 1
        kwargs["impl"] = impl  # explicit "cpp" raises rather than falling back
        return real(*args, **kwargs)

    qbt.stats.stationary_bootstrap_indices = forced
    rr.stationary_bootstrap_indices = forced
    bootstrap_s = {"total": 0.0}

    def timed_call(fn):
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                bootstrap_s["total"] += time.perf_counter() - t
        return wrapper

    # Neither calls the other, so summing them never double-counts.
    rr.stationary_bootstrap_sharpe = timed_call(rr.stationary_bootstrap_sharpe)
    rr.paired_stationary_bootstrap = timed_call(rr.paired_stationary_bootstrap)
    rr.RESULTS_DIR = Path(out_dir)
    rr.RESULTS_PATH = rr.RESULTS_DIR / "summary.md"
    import_s = time.perf_counter() - t_import

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        rr.main()
    main_s = time.perf_counter() - t0

    digest = hashlib.sha256(rr.RESULTS_PATH.read_bytes()).hexdigest()
    print(json.dumps({"impl": impl, "main_s": main_s, "import_s": import_s,
                      "bootstrap_s": bootstrap_s["total"],
                      "bootstrap_share_of_main": bootstrap_s["total"] / main_s,
                      "bootstrap_calls": calls["n"], "summary_sha256": digest}))


def _busy_cpu_seconds() -> float | None:
    """CPU-seconds spent non-idle by the whole system since boot (Linux only)."""
    try:
        fields = Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]
    except OSError:
        return None
    user, nice, system, idle, iowait, irq, softirq, steal = (int(x) for x in fields[:8])
    return (user + nice + system + irq + softirq + steal) / os.sysconf("SC_CLK_TCK")


def _children_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def parent(runs: int, out: Path) -> None:
    import machine
    from regenerate_results import END, START, TICKER

    from qbt.data import fetch_ohlcv

    fetch_ohlcv(TICKER, START, END)  # warm the cache before any timing
    info = machine.machine_info()
    if not info["extension_available"]:
        sys.exit("C++ extension unavailable; build it first (pip install -e .)")

    records = []
    for run in range(1, runs + 1):
        k = (run - 1) % len(IMPLS)
        for impl in IMPLS[k:] + IMPLS[:k]:
            with tempfile.TemporaryDirectory() as tmp:
                busy0, child0 = _busy_cpu_seconds(), _children_cpu_seconds()
                t0 = time.perf_counter()
                proc = subprocess.run([sys.executable, __file__, "--child", impl, tmp],
                                      capture_output=True, text=True, check=True)
                process_s = time.perf_counter() - t0
                busy1, child_cpu_s = _busy_cpu_seconds(), _children_cpu_seconds() - child0
            # Contamination check: CPUs' worth of work done by everything
            # *other* than this run, averaged over the run (0.0 = quiet machine).
            other_cpus = (None if busy0 is None or busy1 is None
                          else max(0.0, (busy1 - busy0 - child_cpu_s) / process_s))
            rec = json.loads(proc.stdout.strip().splitlines()[-1])
            rec.update({"run": run, "process_s": process_s, "child_cpu_s": child_cpu_s,
                        "other_processes_avg_cpus": other_cpus})
            records.append(rec)
            print(f"run {run} {impl:6}  main {rec['main_s']:6.2f}s  "
                  f"bootstrap {rec['bootstrap_s']:6.2f}s ({rec['bootstrap_share_of_main']:.0%})  "
                  f"process {process_s:6.2f}s  child cpu {child_cpu_s:6.2f}s  "
                  f"other load {other_cpus:.2f} cpus  calls {rec['bootstrap_calls']}", flush=True)

    digests = {r["summary_sha256"] for r in records}
    def spread(xs):
        return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}

    def per_run(impl, key):
        return [r[key] for r in sorted(records, key=lambda r: r["run"]) if r["impl"] == impl]

    summary: dict = {}
    for impl in IMPLS:
        summary[impl] = {key: spread(per_run(impl, key)) for key in
                         ("main_s", "process_s", "bootstrap_s", "bootstrap_share_of_main",
                          "child_cpu_s", "other_processes_avg_cpus")}
    for slow, fast in (("python", "cpp"), ("numpy", "cpp"), ("python", "numpy")):
        for key in ("main_s", "bootstrap_s"):
            pairs = zip(per_run(slow, key), per_run(fast, key), strict=True)
            summary[f"ratio_{slow}_over_{fast}_{key}"] = spread([a / b for a, b in pairs])
    result = {
        "benchmark": "scripts/regenerate_results.py full run, impl forced",
        "machine": info,
        "loadavg_1_5_15_at_end": list(os.getloadavg()),
        "runs_per_impl": runs,
        "order": "interleaved python, numpy, cpp per run; fresh subprocess each",
        "all_summaries_identical": len(digests) == 1,
        "summary_sha256": sorted(digests),
        "summary": summary,
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"identical summaries across all runs: {len(digests) == 1}; wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("IMPL", "OUT_DIR"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks/results/full_run.json")
    args = ap.parse_args()
    if args.child:
        child(*args.child)
    else:
        parent(args.runs, args.out)
