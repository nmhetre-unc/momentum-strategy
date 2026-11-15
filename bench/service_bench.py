"""
Measures the backtest service under Docker Compose. Results, each with the
environment they ran in (bench/environment.py), go to bench/results/service/.

    python bench/service_bench.py all            # everything, about an hour
    python bench/service_bench.py throughput     # or one measurement
    python bench/service_bench.py report         # the results table, from the JSON files

Runs as its own Compose project (qbt-bench, ports 56432/58100) built from
compose.yaml as is: production settings, no test delay, real backtests.
SPY 2010-2025 is ingested once; every job runs one of the four strategies
with default parameters (bench/loadgen.py: a fixed round-robin mix, one
seed per job), n_boot 2000.

Each measurement is repeated 3 times and reported as median [min-max].
Every repetition first checks the host is on AC power (it aborts if not),
empties the jobs table and VACUUMs, and the repetitions of the throughput
measurement rotate the order of worker counts so slow drift (heat, other
load) doesn't line up with one worker count.

throughput  Pause the W workers, queue a backlog through the API (300 jobs
            per worker), unpause, wait until every job is done. Rate is
            measured over the middle 80% of completions (first and last 10%
            are ramp-up and the tail). CPU use per container and for the
            whole VM comes from cgroup cpu.stat over the same window.
control     The same backtests with no service at all: W one-off containers
            of the image, each running the strategy mix in a loop on the SPY
            frame for 30 s (one BLAS thread, as in the worker's compute
            process), started together. Its jobs/min is the ceiling the CPU
            allows; the service at W workers can't beat it, and the gap is
            what the queue, Postgres and the result writes cost.
latency     4 workers, Poisson submissions at 50% and 75% of the median
            4-worker service throughput for 150 s. Submit-to-start and
            submit-to-done from the database's own timestamps (created_at,
            started_at, finished_at) for jobs submitted after a 30 s warm-up.
            (Not 90%: that rate is measured draining a backlog with nothing
            else running, and the API and the load generator take CPU too,
            so 90% may exceed what the service sustains.) The backlog
            (submitted minus finished) is sampled once a second while
            submitting; if it grows, the queue isn't stable, its p99 only
            measures how long the run was, and the growth rate is reported
            instead.
get         GET /backtests/{id} of a succeeded job at a fixed 100 req/s for
            60 s, open loop, latency from the scheduled send time: once with
            the workers idle, once while 4 workers drain a backlog.
kill        4 workers, Poisson submissions at 60% of capacity. 40 s in,
            `docker kill` (SIGKILL) a random worker container; if it was
            between jobs, try another 20 s later. Times from the kill until
            the job it held is requeued (by the reaper, after its lease
            expires), claimed again, and finished. Then every submitted job
            is checked: present once, succeeded, one result, from the right
            attempt; worker logs are searched for fencing rejections; the
            killed job's metrics are compared with a direct qbt.runner run.
"""

import argparse
import asyncio
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import loadgen  # noqa: E402
from compose_smoke import DIRECT_RUN  # noqa: E402
from environment import KeepAwake, environment, on_ac_power, power_events_since  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "service"
PROJECT, API_PORT, DB_PORT = "qbt-bench", 58100, 56432
API = f"http://127.0.0.1:{API_PORT}"
DSN = f"postgresql://qbt:qbt@127.0.0.1:{DB_PORT}/qbt"
ENV = os.environ | {"QBT_API_PORT": str(API_PORT), "QBT_DB_PORT": str(DB_PORT)}
for name in ("COMPOSE_PROJECT_NAME", "COMPOSE_FILE", "QBT_TEST_COMPUTE_DELAY_SECONDS"):
    ENV.pop(name, None)

REPS = 3
WORKER_COUNTS = (1, 2, 4, 8)
BACKLOG_PER_WORKER = 300
LATENCY_WORKERS, LATENCY_LOADS, LATENCY_SECONDS, WARMUP_SECONDS = 4, (0.5, 0.75), 150, 30
# A queue counts as growing if its backlog rises by more than 2% of the
# arrival rate (jobs/s per s of run) over the measured window.
GROWTH_THRESHOLD = 0.02
GET_RATE, GET_SECONDS, GET_POOL = 100, 60, 1000
KILL_WORKERS, KILL_LOAD, KILL_SECONDS, KILL_AT, KILL_RETRY = 4, 0.6, 150, 40, 20
COOLDOWN_SECONDS = 10
CONTROL_SECONDS = 30
BASE_CLOCK_GHZ = 1.8  # Intel Core 7 150U; the "% Processor Performance" reference


def log(message: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


# ---- the stack ----

# Which Compose file(s) to run; bench/before_after.py points this at variants.
COMPOSE_FILES = ["-f", "compose.yaml"]


def compose(*args: str, capture: bool = False) -> str:
    result = subprocess.run(["docker", "compose", "-p", PROJECT, *COMPOSE_FILES,
                             "--project-directory", str(ROOT), *args],
                            cwd=ROOT, env=ENV, text=True, check=True,
                            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                            stderr=subprocess.STDOUT if capture else subprocess.DEVNULL)
    return result.stdout or ""


def sql(query: str, *params: Any) -> list[tuple]:
    with psycopg.connect(DSN, autocommit=True) as conn:
        cursor = conn.execute(query, params or None)
        return cursor.fetchall() if cursor.description else []


def setup() -> None:
    log("stack: build, up, ingest SPY")
    compose("build")
    compose("up", "-d", "--wait", "--wait-timeout", "180")
    if not sql("SELECT 1 FROM price_series WHERE ticker = 'SPY'"):
        compose("run", "--rm", "ingest", loadgen.TICKER, "--start", loadgen.START,
                "--end", loadgen.END)


def worker_ids() -> list[str]:
    return compose("ps", "-q", "worker", capture=True).split()


def scale(workers: int) -> list[str]:
    compose("up", "-d", "--wait", "--wait-timeout", "180", "--no-recreate",
            "--scale", f"worker={workers}", "worker")
    ids = worker_ids()
    assert len(ids) == workers, (workers, ids)
    return ids


def reset() -> None:
    if not on_ac_power():
        sys.exit("not on AC power: stopping (results on battery are not comparable)")
    sql("DELETE FROM jobs")
    sql("VACUUM ANALYZE jobs, results, metrics")
    time.sleep(COOLDOWN_SECONDS)


def wait_drained(timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        (pending,) = sql("SELECT count(*) FROM jobs WHERE status IN ('queued', 'running')")[0]
        if pending == 0:
            return
        time.sleep(0.5)
    raise RuntimeError("jobs still pending after the timeout")


def container_ids() -> dict[str, str]:
    """Full container id -> service name."""
    out = compose("ps", "--format", "{{.ID}} {{.Service}}", "--no-trunc", capture=True)
    return dict(line.split() for line in out.splitlines() if line.strip())


# ---- CPU, from cgroup v2 cpu.stat (visible from the WSL distro) ----

def _usage_usec(path: Path) -> int | None:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


# Windows' "% Processor Performance": the effective clock as a percentage of
# the base clock (1.8 GHz on this CPU), across all logical processors.
# Streamed once a second by one powershell.exe for the whole sample.
_PERF_COUNTER = (
    "Get-Counter '\\Processor Information(_Total)\\% Processor Performance' "
    "-SampleInterval 1 -Continuous | ForEach-Object { "
    "$_.CounterSamples[0].CookedValue; [Console]::Out.Flush() }")


class CpuSampler(threading.Thread):
    """Samples CPU time once a second: the whole VM and each container, plus
    the host's effective clock (when powershell.exe is reachable)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.containers = container_ids()
        self.samples: list[tuple[float, int, dict[str, int]]] = []
        self.perf: list[tuple[float, float]] = []
        self._halt = threading.Event()
        self._perf_process: subprocess.Popen | None = None

    def _read_perf(self) -> None:
        try:
            self._perf_process = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-Command", _PERF_COUNTER],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        assert self._perf_process.stdout is not None
        for line in self._perf_process.stdout:
            try:
                self.perf.append((time.time(), float(line)))
            except ValueError:
                continue

    def run(self) -> None:
        threading.Thread(target=self._read_perf, daemon=True).start()
        while not self._halt.is_set():
            vm = _usage_usec(Path("/sys/fs/cgroup/cpu.stat")) or 0
            per = {}
            for cid in self.containers:
                usage = _usage_usec(Path(f"/sys/fs/cgroup/docker/{cid}/cpu.stat"))
                if usage is not None:
                    per[cid] = usage
            self.samples.append((time.time(), vm, per))
            self._halt.wait(1.0)

    def stop(self) -> None:
        self._halt.set()
        self.join()
        if self._perf_process is not None:
            self._perf_process.kill()
            self._perf_process.wait()

    def cores(self, t0: float, t1: float) -> dict[str, float]:
        """Mean CPUs busy between two wall-clock times: vm, and per service."""
        inside = [s for s in self.samples if t0 <= s[0] <= t1]
        if len(inside) < 2:
            return {}
        (ta, vma, pa), (tb, vmb, pb) = inside[0], inside[-1]
        seconds = tb - ta
        result = {"vm": (vmb - vma) / 1e6 / seconds}
        for cid, service in self.containers.items():
            if cid in pa and cid in pb:
                result[service] = result.get(service, 0.0) + (pb[cid] - pa[cid]) / 1e6 / seconds
        perf = [v for t, v in self.perf if t0 <= t <= t1]
        if perf:
            result["host_clock_ghz"] = statistics.mean(perf) / 100 * BASE_CLOCK_GHZ
        return {k: round(v, 2) for k, v in result.items()}


# ---- statistics ----

def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")


def summarize(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def save(name: str, data: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
    log(f"wrote {RESULTS / f'{name}.json'}")


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


class PowerChanged(RuntimeError):
    pass


def guarded(label: str, fn: Any, *args: Any) -> dict:
    """
    Runs one repetition and keeps it only if the machine stayed awake and on
    AC power throughout: no sleep, wake, Modern Standby or power-source event
    in the Windows System log between its start and end, and AC at the end.
    A spoiled repetition is discarded and rerun (up to twice) if the machine
    is back on AC; otherwise the benchmark stops, keeping what it saved.
    """
    for attempt in range(3):
        started = datetime.now(UTC)
        result = fn(*args)
        events = power_events_since(started)
        if events == [] and on_ac_power():
            return result | {"started": started.isoformat(timespec="seconds"),
                             "ended": datetime.now(UTC).isoformat(timespec="seconds"),
                             "power_events": [], "discarded_attempts": attempt}
        log(f"  {label}: DISCARDED (power events {events}, AC {on_ac_power()})")
        if not on_ac_power():
            raise PowerChanged(f"{label}: not on AC power")
    raise PowerChanged(f"{label}: power events on three attempts in a row")


def checkpoint(name: str, runs: list[dict], config: dict) -> None:
    """Saves the repetitions so far, so a stopped run loses nothing."""
    save(name, {"measurement": name, "complete": False, "config": config, "runs": runs})


def job_rows() -> list[tuple]:
    return sql("""
        SELECT extract(epoch FROM j.created_at)::float8, extract(epoch FROM j.started_at)::float8,
               extract(epoch FROM j.finished_at)::float8, r.duration_ms, j.attempts, j.status::text,
               r.attempt
        FROM jobs j LEFT JOIN results r ON r.job_id = j.id""")


# ---- throughput ----

def throughput_once(workers: int, first_index: int) -> dict:
    reset()
    scale(workers)
    compose("pause", "worker")
    backlog = BACKLOG_PER_WORKER * workers
    posted = asyncio.run(loadgen.submit_burst(API, backlog, first_index=first_index))
    assert posted.errors == 0, f"{posted.errors} POSTs failed"
    post_rate = backlog / (posted.finished - posted.started)

    sampler = CpuSampler()
    sampler.start()
    compose("unpause", "worker")
    wait_drained()
    sampler.stop()

    rows = job_rows()
    assert all(r[5] == "succeeded" for r in rows), "a job did not succeed"
    finished = sorted(r[2] for r in rows)
    lo, hi = int(0.1 * len(finished)), int(0.9 * len(finished)) - 1
    t0, t1 = finished[lo], finished[hi]
    rate = (hi - lo) / (t1 - t0)
    busy = sum(max(0.0, min(r[2], t1) - max(r[1], t0)) for r in rows) / (workers * (t1 - t0))
    in_window = [r for r in rows if t0 <= r[2] <= t1]
    wall = [(r[2] - r[1]) * 1000 for r in in_window]
    compute = [float(r[3]) for r in in_window]
    return {
        "workers": workers, "jobs": len(rows), "jobs_per_minute": rate * 60,
        "window_seconds": t1 - t0, "backlog_post_rate_per_s": post_rate,
        "worker_busy_fraction": busy,
        "job_wall_ms_p50": pct(wall, 50), "job_compute_ms_p50": pct(compute, 50),
        "job_overhead_ms_p50": pct([w - c for w, c in zip(wall, compute, strict=True)], 50),
        "cpus_busy": sampler.cores(t0, t1),
    }


def throughput() -> dict:
    runs: list[dict] = []
    index = 0
    for rep in range(REPS):
        shift = rep % len(WORKER_COUNTS)
        order = list(WORKER_COUNTS[shift:] + WORKER_COUNTS[:shift])
        for workers in order:
            log(f"throughput rep {rep + 1}/{REPS}: {workers} workers")
            result = guarded(f"throughput {workers}", throughput_once, workers, index)
            result["rep"] = rep + 1
            index += result["jobs"]
            log(f"  {result['jobs_per_minute']:.0f} jobs/min, cpus {result['cpus_busy']}")
            runs.append(result)
            checkpoint("throughput", runs, {"worker_counts": WORKER_COUNTS})
    summary = {}
    for workers in WORKER_COUNTS:
        mine = [r for r in runs if r["workers"] == workers]
        summary[str(workers)] = {key: summarize([r[key] for r in mine]) for key in (
            "jobs_per_minute", "worker_busy_fraction", "job_wall_ms_p50", "job_compute_ms_p50",
            "job_overhead_ms_p50", "backlog_post_rate_per_s")} | {
            "cpus_busy": {svc: summarize([r["cpus_busy"].get(svc, 0.0) for r in mine])
                          for svc in ("vm", "worker", "db", "api", "host_clock_ghz")}}
    data = {"measurement": "throughput", "environment": environment(),
            "config": {"worker_counts": WORKER_COUNTS, "backlog_per_worker": BACKLOG_PER_WORKER,
                       "mix": loadgen.STRATEGIES, "reps": REPS},
            "summary": summary, "runs": runs}
    save("throughput", data)
    return data


# ---- compute-only control ----

CONTROL_SCRIPT = """
import json, statistics, sys, time
from datetime import date
from qbt.runner import run
from qbt_service.db import make_engine
from qbt_service.models import parse_request
from qbt_service.prices import find_series, load_frame
from qbt_service.settings import Settings

start_at, seconds, worker = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
strategies = sys.argv[4].split(",")
engine = make_engine(Settings.from_env().database_url)
with engine.connect() as conn:
    series = find_series(conn, "SPY", date(2010, 1, 1), date(2025, 1, 1))
    frame = load_frame(conn, series, date(2010, 1, 1), date(2025, 1, 1))
engine.dispose()
requests = [parse_request({"strategy": s, "ticker": "SPY", "start": "2010-01-01",
                           "end": "2025-01-01"}) for s in strategies]

def once(i):
    r = requests[i % len(requests)]
    run(frame, r.strategy, r.params.model_dump(), cost_bps=r.cost_bps, n_boot=r.n_boot,
        seed=1000 * worker + i)

once(0)  # warm up imports and caches
late = time.time() - start_at
time.sleep(max(0.0, start_at - time.time()))
durations, i = [], 0
while time.time() < start_at + seconds:
    t = time.perf_counter()
    once(i)
    durations.append((time.perf_counter() - t) * 1000)
    i += 1
print(json.dumps({"count": i, "median_ms": statistics.median(durations), "late_s": late}))
"""
THREADS_ONE = [f"{name}=1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")]


def control_once(workers: int) -> dict:
    reset()
    compose("pause", "worker")
    start_at = time.time() + 15 + 2 * workers  # time for every container to start and warm up
    sampler = CpuSampler()
    sampler.start()
    processes = []
    for w in range(workers):
        env_args = [arg for pair in THREADS_ONE for arg in ("-e", pair)]
        processes.append(subprocess.Popen(
            ["docker", "compose", "-p", PROJECT, *COMPOSE_FILES, "--project-directory", str(ROOT),
             "run", "--rm", "--no-deps",
             *env_args, "--entrypoint", "python", "api", "-c", CONTROL_SCRIPT,
             str(start_at), str(CONTROL_SECONDS), str(w), ",".join(loadgen.STRATEGIES)],
            cwd=ROOT, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
    outputs = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in processes]
    sampler.stop()
    compose("unpause", "worker")
    assert all(o["late_s"] < 0 for o in outputs), f"a container started late: {outputs}"
    total = sum(o["count"] for o in outputs)
    return {"workers": workers, "backtests": total,
            "jobs_per_minute": total / CONTROL_SECONDS * 60,
            "compute_ms_p50": statistics.median(o["median_ms"] for o in outputs),
            "cpus_busy": sampler.cores(start_at + 2, start_at + CONTROL_SECONDS - 1)}


def control() -> dict:
    runs = []
    for rep in range(REPS):
        shift = rep % len(WORKER_COUNTS)
        for workers in WORKER_COUNTS[shift:] + WORKER_COUNTS[:shift]:
            log(f"control rep {rep + 1}/{REPS}: {workers} processes")
            result = guarded(f"control {workers}", control_once, workers) | {"rep": rep + 1}
            log(f"  {result['jobs_per_minute']:.0f} backtests/min, "
                f"{result['compute_ms_p50']:.0f} ms each, cpus {result['cpus_busy']}")
            runs.append(result)
            checkpoint("control", runs, {"worker_counts": WORKER_COUNTS})
    summary = {}
    for workers in WORKER_COUNTS:
        mine = [r for r in runs if r["workers"] == workers]
        summary[str(workers)] = {k: summarize([r[k] for r in mine])
                                 for k in ("jobs_per_minute", "compute_ms_p50")} | {
            "cpus_busy": {k: summarize([r["cpus_busy"].get(k, float("nan")) for r in mine])
                          for k in ("vm", "host_clock_ghz")}}
    data = {"measurement": "control", "environment": environment(),
            "config": {"worker_counts": WORKER_COUNTS, "seconds": CONTROL_SECONDS,
                       "mix": loadgen.STRATEGIES, "reps": REPS,
                       "what": "qbt.runner.run in W parallel containers; no API, queue or DB"},
            "summary": summary, "runs": runs}
    save("control", data)
    return data


def capacity_per_second(workers: int) -> float:
    return load("throughput")["summary"][str(workers)]["jobs_per_minute"]["median"] / 60


# ---- queue latency ----

def _rate_in(events: np.ndarray, times: np.ndarray) -> float | None:
    """Events per second between the first and last of `times`."""
    if len(times) < 2:
        return None
    t0, t1 = times[0], times[-1]
    return float(((events >= t0) & (events <= t1)).sum() / (t1 - t0))


def latency_once(rep: int, load_fraction: float, rate: float) -> dict:
    reset()
    scale(LATENCY_WORKERS)
    sampler = CpuSampler()
    sampler.start()
    posted = asyncio.run(loadgen.submit_poisson(API, rate, LATENCY_SECONDS,
                                                seed=1000 * rep + int(100 * load_fraction)))
    wait_drained()
    sampler.stop()
    rows = job_rows()
    assert all(r[5] == "succeeded" for r in rows)
    start = min(r[0] for r in rows)
    measured = [r for r in rows if r[0] >= start + WARMUP_SECONDS]
    to_start = [(r[1] - r[0]) * 1000 for r in measured]
    to_done = [(r[2] - r[0]) * 1000 for r in measured]
    post_ms = [s.latency_ms for s in posted.samples]

    # Backlog = submitted - finished, once a second over the measured window
    # (warm-up to the last submission); its least-squares slope is the growth.
    created = np.sort([r[0] for r in rows])
    finished = np.sort([r[2] for r in rows])
    times = np.arange(start + WARMUP_SECONDS, created[-1], 1.0)
    backlog = (np.searchsorted(created, times, side="right")
               - np.searchsorted(finished, times, side="right"))
    slope = float(np.polyfit(times - times[0], backlog, 1)[0]) if len(times) > 2 else 0.0
    growing = slope > GROWTH_THRESHOLD * rate
    return {
        "rep": rep + 1, "load": load_fraction, "target_rate_per_s": rate,
        "backlog_growth_jobs_per_s": slope, "queue_growing": bool(growing),
        "backlog_at_warmup_end": int(backlog[0]) if len(backlog) else None,
        "backlog_at_last_submit": int(backlog[-1]) if len(backlog) else None,
        "backlog_max": int(backlog.max()) if len(backlog) else None,
        # Over the measured window: arrivals vs completions (equal when stable).
        "arrival_rate_per_s": _rate_in(created, times),
        "completion_rate_per_s": _rate_in(finished, times),
        "submitted": len(posted.samples), "post_errors": posted.errors,
        "measured_jobs": len(measured),
        "achieved_rate_per_s": len(rows) / (max(r[0] for r in rows) - start),
        "submit_to_start_ms_p50": pct(to_start, 50),
        "submit_to_start_ms_p99": pct(to_start, 99),
        "submit_to_done_ms_p50": pct(to_done, 50),
        "submit_to_done_ms_p99": pct(to_done, 99),
        "post_ms_p50": pct(post_ms, 50), "post_ms_p99": pct(post_ms, 99),
        "max_send_lag_ms": posted.max_send_lag_ms,
        "drain_after_last_submit_s": max(r[2] for r in rows) - max(r[0] for r in rows),
        "cpus_busy": sampler.cores(start + WARMUP_SECONDS, max(r[2] for r in rows)),
    }


def latency() -> dict:
    capacity = capacity_per_second(LATENCY_WORKERS)
    runs = []
    for rep in range(REPS):
        for load_fraction in LATENCY_LOADS:
            rate = load_fraction * capacity
            log(f"latency rep {rep + 1}/{REPS}: {load_fraction:.0%} load = {rate:.1f} jobs/s")
            run = guarded(f"latency {load_fraction:.0%}", latency_once, rep, load_fraction, rate)
            log(f"  backlog growth {run['backlog_growth_jobs_per_s']:+.3f} jobs/s "
                f"(growing: {run['queue_growing']}, max {run['backlog_max']})")
            log(f"  start p50/p99 {run['submit_to_start_ms_p50']:.0f}/"
                f"{run['submit_to_start_ms_p99']:.0f} ms, done p50/p99 "
                f"{run['submit_to_done_ms_p50']:.0f}/{run['submit_to_done_ms_p99']:.0f} ms")
            runs.append(run)
            checkpoint("latency", runs, {"capacity_per_s": capacity})
    keys = ("submit_to_start_ms_p50", "submit_to_start_ms_p99", "submit_to_done_ms_p50",
            "submit_to_done_ms_p99", "post_ms_p50", "post_ms_p99", "achieved_rate_per_s",
            "backlog_growth_jobs_per_s", "backlog_max")
    summary = {str(f): {k: summarize([r[k] for r in runs if r["load"] == f]) for k in keys}
               for f in LATENCY_LOADS}
    data = {"measurement": "latency", "environment": environment(),
            "config": {"workers": LATENCY_WORKERS, "capacity_per_s": capacity,
                       "loads": LATENCY_LOADS, "seconds": LATENCY_SECONDS,
                       "warmup_seconds": WARMUP_SECONDS, "arrivals": "Poisson", "reps": REPS},
            "summary": summary, "runs": runs}
    save("latency", data)
    return data


# ---- GET ----

def get_once(rep: int, condition: str) -> dict:
    reset()
    scale(LATENCY_WORKERS)
    pool = asyncio.run(loadgen.submit_burst(API, GET_POOL))
    wait_drained()
    ids = [s.job_id for s in pool.samples if s.job_id]
    queued_after = None
    if condition == "under_load":
        compose("pause", "worker")
        backlog = int(capacity_per_second(LATENCY_WORKERS) * (GET_SECONDS + 30))
        asyncio.run(loadgen.submit_burst(API, backlog, first_index=GET_POOL))
        compose("unpause", "worker")
        time.sleep(3)
    sampler = CpuSampler()
    sampler.start()
    t0 = time.time()
    got = asyncio.run(loadgen.get_fixed_rate(API, ids, GET_RATE, GET_SECONDS, seed=rep))
    sampler.stop()
    if condition == "under_load":
        (queued_after,) = sql("SELECT count(*) FROM jobs WHERE status = 'queued'")[0]
        assert queued_after > 0, "the backlog ran out before the GET run ended"
        wait_drained()
    latency_ms = [s.latency_ms for s in got.samples]
    return {"rep": rep + 1, "condition": condition, "requests": len(got.samples),
            "errors": got.errors,
            "achieved_rate_per_s": len(got.samples) / (got.finished - got.started),
            "get_ms_p50": pct(latency_ms, 50), "get_ms_p99": pct(latency_ms, 99),
            "service_ms_p50": pct([s.service_ms for s in got.samples], 50),
            "max_send_lag_ms": got.max_send_lag_ms, "queued_at_end": queued_after,
            "cpus_busy": sampler.cores(t0, time.time())}


def get() -> dict:
    runs = []
    for rep in range(REPS):
        for condition in ("idle", "under_load"):
            log(f"get rep {rep + 1}/{REPS}: {condition}")
            run = guarded(f"get {condition}", get_once, rep, condition)
            log(f"  p50/p99 {run['get_ms_p50']:.1f}/{run['get_ms_p99']:.1f} ms, "
                f"{run['errors']} errors")
            runs.append(run)
            checkpoint("get", runs, {"rate_per_s": GET_RATE})
    summary = {c: {k: summarize([r[k] for r in runs if r["condition"] == c])
                   for k in ("get_ms_p50", "get_ms_p99", "achieved_rate_per_s")}
               for c in ("idle", "under_load")}
    data = {"measurement": "get", "environment": environment(),
            "config": {"rate_per_s": GET_RATE, "seconds": GET_SECONDS, "workers": LATENCY_WORKERS,
                       "pool_of_succeeded_jobs": GET_POOL, "reps": REPS,
                       "latency_from": "scheduled send time (open loop)"},
            "summary": summary, "runs": runs}
    save("get", data)
    return data


# ---- kill a worker ----

def _victims(container: str) -> list[str]:
    """Jobs still marked running under the killed container's worker id
    (worker ids are hostname:pid, and the hostname is the short container id)."""
    rows = sql("SELECT id::text FROM jobs WHERE status = 'running' AND lease_owner LIKE %s",
               f"{container[:12]}:%")
    return [r[0] for r in rows]


def _follow(job_id: str, killed_at: float, timeout: float = 180) -> dict:
    """Polls the job every 0.1 s: times from the kill until requeued, reclaimed, finished."""
    seen: dict[str, float] = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, attempts, last_error = sql(
            "SELECT status::text, attempts, last_error FROM jobs WHERE id = %s", job_id)[0]
        now = time.monotonic() - killed_at
        if status == "queued" and "requeued" not in seen:
            seen["requeued"] = now
            seen["requeue_reason"] = last_error
        if status == "running" and attempts >= 2 and "reclaimed" not in seen:
            seen.setdefault("requeued", now)
            seen["reclaimed"] = now
        if status in ("succeeded", "failed"):
            seen.setdefault("requeued", now)
            seen.setdefault("reclaimed", now)
            seen["finished"] = now
            seen["final_status"] = status
            seen["attempts"] = attempts
            return seen
        time.sleep(0.1)
    raise RuntimeError(f"job {job_id} not finished {timeout}s after the kill")


def _direct_metrics(job_id: str) -> dict:
    request, series_id = sql("SELECT request, price_series_id FROM jobs WHERE id = %s", job_id)[0]
    out = compose("run", "--rm", "--no-deps", "--entrypoint", "python", "api", "-c", DIRECT_RUN,
                  json.dumps({"request": request, "price_series_id": series_id}), capture=True)
    return json.loads(out.strip().splitlines()[-1])["metrics"]


def _stored_metrics(job_id: str) -> dict:
    rows = sql("SELECT name, value FROM metrics WHERE job_id = %s", job_id)
    return {name: None if value != value else value for name, value in rows}  # NaN -> None


def kill_once(rep: int) -> dict:
    reset()
    scale(KILL_WORKERS)
    since = datetime.now(UTC).isoformat()
    rate = KILL_LOAD * capacity_per_second(KILL_WORKERS)
    rng = random.Random(rep)
    box: dict[str, Any] = {}
    load_thread = threading.Thread(target=lambda: box.setdefault("run", asyncio.run(
        loadgen.submit_poisson(API, rate, KILL_SECONDS, seed=500 + rep))))
    load_thread.start()

    time.sleep(KILL_AT)
    attempts: list[dict] = []
    victims: list[str] = []
    killed_at = 0.0
    while not victims and len(attempts) < 5:
        container = rng.choice(worker_ids())
        subprocess.run(["docker", "kill", container], check=True, capture_output=True)
        killed_at = time.monotonic()
        victims = _victims(container)
        attempts.append({"container": container[:12], "jobs_in_flight": len(victims)})
        log(f"  killed {container[:12]}: {len(victims)} job(s) in flight")
        if not victims:
            time.sleep(KILL_RETRY)
    followed = {v: _follow(v, killed_at) for v in victims}
    load_thread.join()
    wait_drained()
    posted = box["run"]

    # Integrity: every accepted POST is one job, done once, by the right attempt.
    submitted = {s.job_id for s in posted.samples if s.job_id}
    rows = sql("""SELECT j.id::text, j.status::text, j.attempts, r.attempt,
                         (SELECT count(*) FROM metrics m WHERE m.job_id = j.id)
                  FROM jobs j LEFT JOIN results r ON r.job_id = j.id""")
    stored = {r[0] for r in rows}
    metric_counts = {r[4] for r in rows}
    logs = compose("logs", "--no-log-prefix", "--since", since, "worker", capture=True)
    restarted = [docker_inspect(a["container"]) for a in attempts]
    checks = {
        "posts_accepted": len(submitted), "post_errors": posted.errors,
        "jobs_in_db": len(rows),
        "lost": len(submitted - stored), "unexpected_extra_jobs": len(stored - submitted),
        "not_succeeded": sum(1 for r in rows if r[1] != "succeeded"),
        "without_result": sum(1 for r in rows if r[3] is None),
        "result_attempt_mismatch": sum(1 for r in rows if r[3] is not None and r[3] != r[2]),
        "jobs_with_more_than_one_attempt": sorted(r[0] for r in rows if r[2] > 1),
        "metric_rows_per_job": sorted(metric_counts),
        "fencing_rejections_in_logs": logs.count("lease_lost"),
        "results_discarded_in_logs": logs.count("result discarded"),
        "reaper_requeues_in_logs": logs.count("its lease expired"),
        "killed_job_metrics_equal_direct_run": all(
            _stored_metrics(v) == _direct_metrics(v) for v in victims),
    }
    ok = (checks["lost"] == checks["unexpected_extra_jobs"] == checks["not_succeeded"]
          == checks["without_result"] == checks["result_attempt_mismatch"] == 0
          and checks["jobs_with_more_than_one_attempt"] == sorted(victims)
          and len(checks["metric_rows_per_job"]) == 1
          and checks["killed_job_metrics_equal_direct_run"])
    return {"rep": rep + 1, "load_rate_per_s": rate, "kill_attempts": attempts,
            "victims": followed, "container_restarted_by_docker": restarted,
            "checks": checks, "all_checks_pass": ok}


def docker_inspect(short_id: str) -> dict:
    out = subprocess.run(["docker", "inspect", "--format",
                          "{{.RestartCount}} {{.State.Status}} {{.State.Health.Status}}", short_id],
                         capture_output=True, text=True).stdout.split()
    return {"container": short_id, "restart_count": int(out[0]) if out else None,
            "state": out[1] if len(out) > 1 else None,
            "health": out[2] if len(out) > 2 else None}


def kill() -> dict:
    runs = []
    for rep in range(REPS):
        log(f"kill rep {rep + 1}/{REPS}")
        run = guarded("kill", kill_once, rep)
        for job_id, times in run["victims"].items():
            log(f"  job {job_id[:8]}: requeued {times['requeued']:.1f}s, reclaimed "
                f"{times['reclaimed']:.1f}s, finished {times['finished']:.1f}s after the kill")
        log(f"  all checks pass: {run['all_checks_pass']}")
        runs.append(run)
        checkpoint("kill", runs, {"workers": KILL_WORKERS})
    victims = [t for r in runs for t in r["victims"].values()]
    summary = {k: summarize([t[k] for t in victims]) for k in ("requeued", "reclaimed", "finished")}
    data = {"measurement": "kill", "environment": environment(),
            "config": {"workers": KILL_WORKERS, "load": KILL_LOAD, "seconds": KILL_SECONDS,
                       "kill_at_seconds": KILL_AT, "signal": "SIGKILL (docker kill)",
                       "lease_seconds": 30, "heartbeat_seconds": 10, "reaper_seconds": 15,
                       "retry_backoff_seconds": 5, "reps": REPS},
            "summary": summary, "all_checks_pass": all(r["all_checks_pass"] for r in runs),
            "runs": runs}
    save("kill", data)
    return data


# ---- report ----

def fmt(s: dict, digits: int = 0) -> str:
    return f"{s['median']:.{digits}f} [{s['min']:.{digits}f}-{s['max']:.{digits}f}]"


def _range(values: list[float], digits: int = 0) -> str:
    return fmt(summarize(values), digits)


def report() -> str:
    t, c, lat, g, k = (load(n) for n in ("throughput", "control", "latency", "get", "kill"))
    src = "bench/results/service/"
    env = t["environment"]
    one = t["summary"]["1"]["jobs_per_minute"]["median"]
    lines = ["| Measurement | Condition | Result: median [min-max] of 3 runs | Source |",
             "|---|---|---|---|"]
    for w in WORKER_COUNTS:
        s, cs = t["summary"][str(w)], c["summary"][str(w)]
        share = s["jobs_per_minute"]["median"] / cs["jobs_per_minute"]["median"]
        lines.append(
            f"| Throughput | {w} worker{'s' if w > 1 else ''} | "
            f"**{fmt(s['jobs_per_minute'])} jobs/min**, "
            f"{s['jobs_per_minute']['median'] / one:.2f}x of 1 worker, {share:.0%} of the "
            f"compute-only ceiling. Backtest {fmt(s['job_compute_ms_p50'])} ms, overhead "
            f"{fmt(s['job_overhead_ms_p50'])} ms/job, host clock "
            f"{fmt(s['cpus_busy']['host_clock_ghz'], 2)} GHz; CPUs busy: VM "
            f"{fmt(s['cpus_busy']['vm'], 1)} of {env['docker']['vm_cpus']}, Postgres "
            f"{fmt(s['cpus_busy']['db'], 2)}, API {fmt(s['cpus_busy']['api'], 2)} "
            f"| {src}throughput.json |")
    for w in WORKER_COUNTS:
        cs = c["summary"][str(w)]
        lines.append(
            f"| Compute-only control | {w} parallel container{'s' if w > 1 else ''}, "
            f"no API/queue/DB | {fmt(cs['jobs_per_minute'])} backtests/min, "
            f"{fmt(cs['compute_ms_p50'])} ms each, host clock "
            f"{fmt(cs['cpus_busy']['host_clock_ghz'], 2)} GHz | {src}control.json |")
    capacity = lat["config"]["capacity_per_s"]
    for f in LATENCY_LOADS:
        mine = [r for r in lat["runs"] if r["load"] == f]
        growing = [r for r in mine if r["queue_growing"]]
        condition = (f"4 workers, Poisson, {f:.0%} of 4-worker throughput "
                     f"({f * capacity:.1f} jobs/s)")
        if growing:
            result = (f"**queue grows** in {len(growing)} of {len(mine)} runs: backlog +"
                      f"{_range([r['backlog_growth_jobs_per_s'] for r in mine], 2)} jobs/s "
                      f"(arrivals {_range([r['arrival_rate_per_s'] for r in mine], 1)}/s, "
                      f"completions {_range([r['completion_rate_per_s'] for r in mine], 1)}/s); "
                      f"no p99 reported")
        else:
            result = (
                f"submit-to-start p50 {_range([r['submit_to_start_ms_p50'] for r in mine])} ms, "
                f"p99 {_range([r['submit_to_start_ms_p99'] for r in mine])} ms; submit-to-done "
                f"p50 {_range([r['submit_to_done_ms_p50'] for r in mine])} ms, p99 "
                f"{_range([r['submit_to_done_ms_p99'] for r in mine])} ms (backlog stable: "
                f"{_range([r['backlog_growth_jobs_per_s'] for r in mine], 3)} jobs/s)")
        lines.append(f"| Queue latency | {condition} | {result} | {src}latency.json |")
    for cond in ("idle", "under_load"):
        s = g["summary"][cond]
        lines.append(f"| GET /backtests/{{id}} | {GET_RATE} req/s for {GET_SECONDS} s, "
                     f"{'workers idle' if cond == 'idle' else '4 workers draining a backlog'} | "
                     f"p50 {fmt(s['get_ms_p50'], 1)} ms, p99 {fmt(s['get_ms_p99'], 1)} ms | "
                     f"{src}get.json |")
    s = k["summary"]
    checks = [r["checks"] for r in k["runs"]]
    lines.append(
        f"| Kill a worker (docker kill, SIGKILL) | 4 workers, Poisson at {KILL_LOAD:.0%} | "
        f"job requeued {fmt(s['requeued'], 1)} s, reclaimed {fmt(s['reclaimed'], 1)} s, "
        f"finished {fmt(s['finished'], 1)} s after the kill. Jobs lost "
        f"{sum(x['lost'] for x in checks)}, duplicated "
        f"{sum(x['unexpected_extra_jobs'] for x in checks)}, "
        f"without a result {sum(x['without_result'] for x in checks)}, fencing rejections "
        f"{sum(x['fencing_rejections_in_logs'] for x in checks)}; killed jobs' metrics equal a "
        f"direct run: {all(x['killed_job_metrics_equal_direct_run'] for x in checks)} "
        f"| {src}kill.json |")
    w = env["host"]["windows"]
    header = (
        f"{env['host']['cpu_model']} ({w.get('physical_cores')} cores / "
        f"{w.get('logical_processors')} threads, 2 P + 8 E cores, 15 W), {w.get('ram_gib')} GiB, "
        f"Windows power plan {w.get('power_plan')}, on AC power (every repetition guarded). "
        f"{env['docker']['operating_system']} {env['docker']['server_version']} (WSL2 backend, "
        f"kernel {env['docker']['kernel']}); its VM: {env['docker']['vm_cpus']} CPUs, "
        f"{env['docker']['vm_memory_gib']} GiB, shared with the load generator. "
        f"Fixed mix: {', '.join(loadgen.STRATEGIES)} round robin, SPY 2010-2025, n_boot 2000.")
    return header + "\n\n" + "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    names = ["throughput", "control", "latency", "get", "kill"]
    parser.add_argument("what", nargs="+", choices=["all", "report", *names],
                        help="measurements to run, in this order; 'all' runs every one")
    parser.add_argument("--down", action="store_true", help="remove the stack afterwards")
    args = parser.parse_args()
    if args.what == ["report"]:
        print(report())
        return 0
    if not on_ac_power():
        sys.exit("not on AC power")
    chosen = names if "all" in args.what else [n for n in names if n in args.what]
    steps = {"throughput": throughput, "control": control, "latency": latency, "get": get,
             "kill": kill}
    with KeepAwake():
        setup()
        try:
            for name in chosen:
                steps[name]()
        except PowerChanged as exc:
            log(f"STOPPED: {exc}. Saved repetitions are kept (complete: false).")
            return 1
    usage = resource.getrusage(resource.RUSAGE_SELF)
    log(f"load generator CPU: {usage.ru_utime + usage.ru_stime:.0f} s")
    if all((RESULTS / f"{n}.json").exists() and load(n).get("complete", True) for n in names):
        text = report()
        (RESULTS / "README.md").write_text("# Service measurements\n\n" + text)
        print(text)
    if args.down:
        compose("down", "-v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
