"""
End-to-end smoke test of the Compose stack. Standard library only; needs
Docker (Docker Desktop here) and network access to Yahoo Finance.

    python scripts/compose_smoke.py [--keep]

Runs as its own Compose project (qbt-smoke) on ports 55432/58000, so it
never touches the development stack or its data volume:

1. From scratch: `down -v`, build, `up --wait` (db -> migrate -> api,
   worker; waits until the api and worker healthchecks pass).
2. Ingest SPY with the one-shot ingest service.
3. POST a backtest, poll GET until it succeeds, and check its metrics are
   exactly those of qbt.runner.run on the same snapshot, computed in a
   separate one-off container from the same image.
4. Two workers, backtests slowed by QBT_TEST_COMPUTE_DELAY_SECONDS (the
   scripts/compose.smoke.yaml overlay). POST another backtest, wait for a
   worker to claim it, `docker stop` that worker container (SIGTERM, as
   `docker compose stop` sends), and check: it exits 0 inside its
   stop_grace_period, its log says the job was released, the other worker
   claims and finishes it, the attempt was given back (still attempt 1),
   and the metrics again equal a direct run.
5. `down -v`, unless --keep.

Exits non-zero, with the reason, on the first failed check.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "qbt-smoke"
API_PORT, DB_PORT = 58000, 55432
API = f"http://127.0.0.1:{API_PORT}"
TICKER, START, END = "SPY", "2010-01-01", "2025-01-01"
SLOW_SECONDS = 25  # longer than QBT_SHUTDOWN_GRACE_SECONDS (8), so the stop lands mid-job

ENV = os.environ | {"QBT_API_PORT": str(API_PORT), "QBT_DB_PORT": str(DB_PORT),
                    "QBT_TEST_COMPUTE_DELAY_SECONDS": "0"}

# Runs inside a one-off container of the image: the backtest done directly
# with qbt.runner on the job's price snapshot, metrics as JSON (NaN -> null,
# as the API returns them).
DIRECT_RUN = """
import json, math, sys
from qbt.runner import run
from qbt_service.db import make_engine
from qbt_service.models import parse_request
from qbt_service.prices import load_frame
from qbt_service.settings import Settings

job = json.loads(sys.argv[1])
request = parse_request(job["request"])
with make_engine(Settings.from_env().database_url).connect() as conn:
    frame = load_frame(conn, job["price_series_id"], request.start, request.end)
out = run(frame, request.strategy, request.params.model_dump(), cost_bps=request.cost_bps,
          n_boot=request.n_boot, seed=request.seed)
metrics = {k: None if v is None or math.isnan(v) else float(v)
           for k, v in out.metrics.items()}
print(json.dumps({"metrics": metrics, "bootstrap_impl": out.bootstrap_impl}))
"""


class SmokeFailure(AssertionError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)
    print(f"  ok: {message}")


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def compose(*args: str, capture: bool = False, check_rc: bool = True) -> str:
    command = ["docker", "compose", "-p", PROJECT, "-f", "compose.yaml",
               "-f", "scripts/compose.smoke.yaml", *args]
    result = subprocess.run(command, cwd=ROOT, env=ENV, text=True,
                            stdout=subprocess.PIPE if capture else None)
    if check_rc and result.returncode != 0:
        raise SmokeFailure(f"`{' '.join(command)}` exited {result.returncode}")
    return result.stdout or ""


def docker(*args: str) -> str:
    return subprocess.run(["docker", *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def http(method: str, path: str, body: dict | None = None,
         headers: dict[str, str] | None = None) -> tuple[int, dict, dict]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(f"{API}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"}
                                     | (headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return (response.status, {k.lower(): v for k, v in response.headers.items()},
                    json.loads(response.read()))
    except urllib.error.HTTPError as exc:
        return (exc.code, {k.lower(): v for k, v in exc.headers.items()},
                json.loads(exc.read() or b"{}"))


def submit(payload: dict) -> str:
    status, headers, body = http("POST", "/backtests", payload,
                                 {"Idempotency-Key": f'"{uuid.uuid4()}"'})
    check(status == 202, f"POST /backtests -> 202 (job {body.get('id')})")
    check(headers["location"] == f"/backtests/{body['id']}", "Location points at the job")
    return body["id"]


def wait_for_status(job_id: str, wanted: set[str], timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, _, job = http("GET", f"/backtests/{job_id}")
        if status == 200 and job["status"] in wanted:
            return job
        if status == 200 and job["status"] == "failed":
            raise SmokeFailure(f"job {job_id} failed: {job['last_error']}")
        time.sleep(0.5)
    raise SmokeFailure(f"job {job_id} not {wanted} within {timeout:.0f}s")


def direct_run(job: dict) -> dict:
    out = compose("run", "--rm", "--no-deps", "--entrypoint", "python", "api", "-c",
                  DIRECT_RUN, json.dumps({"request": job["request"],
                                          "price_series_id": job["price_series_id"]}),
                  capture=True)
    return json.loads(out.strip().splitlines()[-1])


def check_matches_direct_run(job: dict) -> None:
    direct = direct_run(job)
    check(job["metrics"] == direct["metrics"],
          f"all {len(job['metrics'])} metrics equal a direct qbt.runner run "
          f"(sharpe_ratio {job['metrics']['sharpe_ratio']!r})")
    check(job["result"]["bootstrap_impl"] == direct["bootstrap_impl"] == "cpp",
          "both used the C++ bootstrap")


def worker_containers() -> list[str]:
    return compose("ps", "-q", "worker", capture=True).split()


def logs(container: str) -> str:
    return docker("logs", container)


def health(container: str) -> str:
    return docker("inspect", "--format", "{{.State.Health.Status}}", container).strip()


def run(keep: bool) -> None:
    step("from scratch: down -v, build, up --wait")
    compose("down", "-v", "--remove-orphans")
    compose("build")
    compose("up", "-d", "--wait", "--wait-timeout", "180")
    migrate = compose("ps", "-a", "--format", "{{.State}} {{.ExitCode}}", "migrate",
                      capture=True).strip()
    check(migrate == "exited 0", f"migrate ran once and exited 0 ({migrate})")
    (worker,) = worker_containers()
    check(health(worker) == "healthy", "worker healthcheck passes")
    status, _, body = http("GET", "/health")
    check(status == 200 and body == {"status": "ok"}, "api /health is 200")

    step(f"ingest {TICKER}")
    compose("run", "--rm", "ingest", TICKER, "--start", START, "--end", END)

    step("POST a backtest, poll until it succeeds, compare with a direct run")
    payload = {"strategy": "sma_crossover", "ticker": TICKER, "start": START, "end": END}
    started = time.monotonic()
    job_id = submit(payload)
    job = wait_for_status(job_id, {"succeeded"}, timeout=120)
    check(True, f"succeeded in {time.monotonic() - started:.1f}s, attempt "
                f"{job['result']['attempt']}, backtest {job['result']['duration_ms']} ms")
    check_matches_direct_run(job)

    step(f"two workers, {SLOW_SECONDS}s backtests: stop the one running a job")
    ENV["QBT_TEST_COMPUTE_DELAY_SECONDS"] = str(SLOW_SECONDS)
    compose("up", "-d", "--wait", "--wait-timeout", "180", "--scale", "worker=2", "worker")
    workers = worker_containers()
    check(len(workers) == 2 and all(health(w) == "healthy" for w in workers),
          "2 healthy workers")

    job_id = submit(payload | {"seed": 1})
    wait_for_status(job_id, {"running"}, timeout=60)
    owner = next(w for w in workers if f"claimed job {job_id}" in logs(w))
    other = next(w for w in workers if w != owner)
    time.sleep(2)  # well inside the backtest
    grace = docker("inspect", "--format", "{{.Config.StopTimeout}}", owner).strip()
    stopped_at = time.monotonic()
    docker("stop", owner)  # SIGTERM; SIGKILL after stop_grace_period
    took = time.monotonic() - stopped_at
    exit_code = docker("inspect", "--format", "{{.State.ExitCode}}", owner).strip()
    owner_log = logs(owner)
    check(exit_code == "0", f"stopped worker exited 0 after {took:.1f}s "
                            f"(stop_grace_period {grace}s, not SIGKILLed)")
    check(f"job {job_id}: released" in owner_log, "its log: the job was released")

    job = wait_for_status(job_id, {"succeeded"}, timeout=SLOW_SECONDS + 60)
    released_to_done = time.monotonic() - stopped_at
    other_log = logs(other)
    check(f"claimed job {job_id}" in other_log and f"job {job_id}: succeeded" in other_log,
          f"the other worker claimed and finished it ({released_to_done:.0f}s after the stop)")
    check(job["attempts"] == 1 and job["result"]["attempt"] == 1,
          "the released attempt was given back: attempts 1, result from attempt 1")
    check_matches_direct_run(job)

    if not keep:
        step("down -v")
        compose("down", "-v", "--remove-orphans")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="leave the stack running")
    args = parser.parse_args()
    try:
        run(args.keep)
    except SmokeFailure as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        print(f"(stack left up for inspection: docker compose -p {PROJECT} logs)",
              file=sys.stderr)
        return 1
    print("\nsmoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
