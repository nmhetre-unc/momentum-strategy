"""
End-to-end smoke test of the Compose stack. Standard library only; needs
Docker (Docker Desktop here) and network access to Yahoo Finance.

    python scripts/compose_smoke.py [--keep]
    python scripts/compose_smoke.py --quick --project qbt-ci --start 2023-01-01 --end 2025-01-01

Runs as its own Compose project (qbt-smoke unless --project) on ports
55432/58000, so it never touches the development stack or its data volume:

1. From scratch: `down -v`, build, `up --wait` (db -> migrate -> api,
   worker; waits until the api and worker healthchecks pass).
2. Ingest SPY with the one-shot ingest service.
3. POST a backtest, poll GET until it succeeds, and check its metrics are
   exactly those of qbt.runner.run on the same snapshot, computed in a
   separate one-off container from the same image.
3b. Ingest SPY 2008-2025, train a model with the one-shot train service
   (the C3 default: LightGBM, regime, HMM features), training again finds
   the same version, then POST /predict and check: the response carries
   the out-of-sample accuracy next to its baseline (as GET /models does),
   the prediction was logged, and a direct computation in a one-off
   container -- features rebuilt from the snapshot, scored with the
   registered artifact -- gives the same input hash and probabilities.
4. Two workers, backtests slowed by QBT_TEST_COMPUTE_DELAY_SECONDS (the
   scripts/compose.smoke.yaml overlay). POST another backtest, wait for a
   worker to claim it, `docker stop` that worker container (SIGTERM, as
   `docker compose stop` sends), and check: it exits 0 inside its
   stop_grace_period, its log says the job was released, the other worker
   claims and finishes it, the attempt was given back (still attempt 1),
   and the metrics again equal a direct run.
5. `down -v`, unless --keep.

--quick stops after step 3b (one backtest and one prediction end to
end); CI runs that, with a short backtest range so it is quick. Training
needs the long range regardless (five years of history before the first
out-of-sample month).

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
PROJECT = "qbt-smoke"  # --project overrides
API_PORT, DB_PORT = 58000, 55432
API = f"http://127.0.0.1:{API_PORT}"
TICKER, START, END = "SPY", "2010-01-01", "2025-01-01"  # --start/--end override
TRAIN_START, TRAIN_END = "2008-01-01", "2025-01-01"   # the C2 range
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


# Runs inside a one-off container: rebuild model version argv[1]'s serving
# features from its snapshot, score them with the registered artifact
# (load_model() verifies its SHA-256), and read prediction argv[2] back.
DIRECT_PREDICT = """
import json, sys
from datetime import date
from sqlalchemy import select
from qbt.ml.train import FEATURE_SETS, features_hash, serving_features
from qbt_service.db import make_engine, predictions
from qbt_service.prices import load_frame
from qbt_service.registry import get_model, load_model
from qbt_service.settings import Settings

version, prediction_id = int(sys.argv[1]), int(sys.argv[2])
with make_engine(Settings.from_env().database_url).connect() as conn:
    row = get_model(conn, version)
    model = load_model(conn, version)
    request = row["training_config"]["request"]
    frame = load_frame(conn, row["price_series_id"], date.fromisoformat(request["start"]),
                       date.fromisoformat(request["end"]))
    logged = conn.execute(select(predictions).where(predictions.c.id == prediction_id)).one()
as_of, features = serving_features(frame, list(row["features"]),
                                   FEATURE_SETS[row["feature_set"]],
                                   row["training_config"]["hmm_kwargs"])
proba = model.predict_proba(features.to_frame().T)[0]
print(json.dumps({
    "as_of": as_of.isoformat(), "features_sha256": features_hash(as_of, features).hex(),
    "probabilities": [float(p) for p in proba],
    "logged": {"model_version": logged.model_version, "as_of": logged.as_of.isoformat(),
               "features_sha256": bytes(logged.features_sha256).hex(),
               "prediction": logged.prediction, "probabilities": list(logged.probabilities)}}))
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


def check_predict() -> None:
    step(f"ingest {TICKER} {TRAIN_START} to {TRAIN_END}, train, POST /predict")
    compose("run", "--rm", "ingest", TICKER, "--start", TRAIN_START, "--end", TRAIN_END)
    trained = compose("run", "--rm", "train", TICKER, "--start", TRAIN_START, "--end",
                      TRAIN_END, capture=True)
    print(trained, end="")
    check("registered model version 1" in trained, "train registered model version 1")
    again = compose("run", "--rm", "train", TICKER, "--start", TRAIN_START, "--end", TRAIN_END,
                    capture=True)
    check("unchanged: identical to registered model version 1" in again,
          "training again gives a byte-identical artifact: no new version")

    status, _, listed = http("GET", "/models")
    check(status == 200 and [m["version"] for m in listed["models"]] == [1], "GET /models")
    entry = listed["models"][0]
    status, headers, body = http("POST", "/predict", {})
    check(status == 200, f"POST /predict -> 200 ({body.get('prediction_name')}, "
                         f"as of {body.get('as_of')})")
    check(body["model_version"] == 1 and body["as_of"] == entry["serving"]["as_of"],
          "the latest version, scored on its snapshot's latest complete month")
    check(body["out_of_sample"] == entry["out_of_sample"] and
          body["baseline"] == entry["baseline"],
          f"carries its out-of-sample accuracy {body['out_of_sample']['accuracy']:.3f} next to "
          f"baseline {body['baseline']['name']} {body['baseline']['accuracy']:.3f} "
          f"(beats: {body['baseline']['beats']})")
    probabilities = [p["probability"] for p in body["probabilities"]]
    check(abs(sum(probabilities) - 1) < 1e-12 and
          body["prediction"] == max(body["probabilities"],
                                    key=lambda p: p["probability"])["label"],
          "probabilities sum to 1 and the prediction is the most probable class")
    status, _, problem = http("POST", "/predict", {"version": 99})
    check(status == 422 and problem["type"] == "/problems/unknown-model-version",
          "an unknown version is a 422 problem")

    out = compose("run", "--rm", "--no-deps", "--entrypoint", "python", "api", "-c",
                  DIRECT_PREDICT, "1", str(body["prediction_id"]), capture=True)
    direct = json.loads(out.strip().splitlines()[-1])
    check(direct["as_of"] == body["as_of"] and
          direct["features_sha256"] == body["features_sha256"],
          "features rebuilt from the snapshot hash to the served input")
    check(direct["probabilities"] == probabilities,
          "the registered artifact, verified and loaded directly, gives identical probabilities")
    logged = direct["logged"]
    check(logged == {"model_version": 1, "as_of": body["as_of"],
                     "features_sha256": body["features_sha256"],
                     "prediction": body["prediction"], "probabilities": probabilities},
          f"the prediction was logged (id {body['prediction_id']})")


def worker_containers() -> list[str]:
    return compose("ps", "-q", "worker", capture=True).split()


def logs(container: str) -> str:
    return docker("logs", container)


def health(container: str) -> str:
    return docker("inspect", "--format", "{{.State.Health.Status}}", container).strip()


def run(keep: bool, quick: bool, start: str, end: str) -> None:
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

    step(f"ingest {TICKER} {start} to {end}")
    compose("run", "--rm", "ingest", TICKER, "--start", start, "--end", end)

    step("POST a backtest, poll until it succeeds, compare with a direct run")
    payload = {"strategy": "sma_crossover", "ticker": TICKER, "start": start, "end": end}
    started = time.monotonic()
    job_id = submit(payload)
    job = wait_for_status(job_id, {"succeeded"}, timeout=120)
    check(True, f"succeeded in {time.monotonic() - started:.1f}s, attempt "
                f"{job['result']['attempt']}, backtest {job['result']['duration_ms']} ms")
    check_matches_direct_run(job)
    check_predict()
    if quick:
        down(keep)
        return

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
    down(keep)


def down(keep: bool) -> None:
    if not keep:
        step("down -v")
        compose("down", "-v", "--remove-orphans")


def main() -> int:
    global PROJECT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="leave the stack running")
    parser.add_argument("--quick", action="store_true",
                        help="one backtest end to end; skip the worker-stop test")
    parser.add_argument("--project", default=PROJECT, help=f"Compose project (default {PROJECT})")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    args = parser.parse_args()
    PROJECT = args.project
    try:
        run(args.keep, args.quick, args.start, args.end)
    except SmokeFailure as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        print(f"(stack left up for inspection: docker compose -p {PROJECT} logs)",
              file=sys.stderr)
        return 1
    print("\nsmoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
