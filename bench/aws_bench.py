"""
Measures the service deployed by infra/ on AWS (D2), from this laptop over
the internet. Results go to bench/results/aws/.

    python bench/aws_bench.py rtt | tls | throughput | latency | get | predict | kill | cloudwatch

A reduced version of bench/service_bench.py (step 9), with the same load
generator (bench/loadgen.py: open loop, the fixed four-strategy mix, one
seed per job) and the same power guard (every repetition is kept only if
the laptop stayed awake and on AC power). What differs:

- The client is a laptop on a home connection; the server is one EC2
  instance (api + workers) and RDS in us-east-2. Every latency is reported
  next to the network round trip measured just before it (`rtt`: TCP
  connects to port 8000, so it is the network alone).
- Job timings (submit-to-start, -done, throughput) come from the database's
  own timestamps, read on the instance, so they don't include the network.
  POST/GET/predict latencies are measured at the laptop and do.
- Commands on the instance (scaling workers, `docker kill`, SQL through the
  running api container) go through SSM Run Command (AWS-RunShellScript):
  no SSH.
- Smaller: 1 and 2 workers instead of 1-8 (the instance has 2 vCPUs),
  fewer jobs and shorter runs, 1 repetition (2 for kill), lower request
  rates; see the constants below. t3.small runs with standard CPU
  credits, so sustained load is throttled to its baseline (20% of each
  vCPU) once the credit balance is spent; `cloudwatch` records the balance
  over the run.
"""

import argparse
import asyncio
import base64
import json
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loadgen  # noqa: E402
from environment import KeepAwake, environment, on_ac_power  # noqa: E402
from scrub import scrub  # noqa: E402
from service_bench import guarded, log, pct  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "aws"
REGION = "us-east-2"
COMPOSE = "docker compose -p qbt -f /opt/qbt/compose.yaml"

THROUGHPUT_WORKERS, JOBS_PER_WORKER = (1, 2), 60
LATENCY_WORKERS, LATENCY_LOAD, LATENCY_SECONDS, WARMUP_SECONDS = 2, 0.5, 90, 20
GET_RATE, GET_SECONDS = 20, 30
PREDICT_RATE, PREDICT_SECONDS, PREDICT_WARMUP_SECONDS = 20, 60, 5
KILL_WORKERS, KILL_LOAD, KILL_SECONDS, KILL_AT, KILL_REPS = 2, 0.6, 120, 10, 2
# A backlog queued first, so both workers are busy when one is killed. At 60%
# of the throttled capacity alone (the first attempt, kept in
# kill_first_attempt_no_job_in_flight.json) jobs of ~0.25 s left the workers
# idle almost all the time, and 10 kill attempts found no job in flight.
KILL_BACKLOG = 40
RTT_SAMPLES = 40


def tf_outputs() -> dict[str, str]:
    out = subprocess.run(["terraform", f"-chdir={ROOT / 'infra'}", "output", "-json"],
                         capture_output=True, text=True, check=True).stdout
    return {k: v["value"] for k, v in json.loads(out).items()}


OUT = tf_outputs()
API = OUT["api_url"]
INSTANCE = OUT["instance_id"]


def save(name: str, data: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=2, default=str) + "\n")
    log(f"wrote {RESULTS / f'{name}.json'}")


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


# ---- the instance, through SSM Run Command ----

def aws(*args: str) -> Any:
    out = subprocess.run(["aws", "--region", REGION, "--output", "json", *args],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out) if out.strip() else None


def remote(script: str, timeout: int = 900) -> str:
    """Runs a bash script on the instance as root; returns its stdout. Both
    the output and a failure's stderr pass through scrub(): a database error
    can quote DATABASE_URL, password included."""
    encoded = base64.b64encode(script.encode()).decode()
    params = json.dumps({"commands": [f"echo {encoded} | base64 -d > /tmp/qbt-bench.sh",
                                      "bash /tmp/qbt-bench.sh"],
                         "executionTimeout": [str(timeout)]})
    command_id = aws("ssm", "send-command", "--instance-ids", INSTANCE,
                     "--document-name", "AWS-RunShellScript", "--parameters", params,
                     "--timeout-seconds", "60")["Command"]["CommandId"]
    deadline = time.monotonic() + timeout + 60
    while time.monotonic() < deadline:
        time.sleep(1)
        try:
            inv = aws("ssm", "get-command-invocation", "--command-id", command_id,
                      "--instance-id", INSTANCE)
        except subprocess.CalledProcessError:
            continue  # InvocationDoesNotExist for the first moment
        if inv["Status"] in ("Pending", "InProgress", "Delayed"):
            continue
        if inv["Status"] != "Success":
            raise RuntimeError(scrub(f"remote command {inv['Status']}: "
                                     f"{inv['StandardErrorContent'][-2000:]}\n"
                                     f"{inv['StandardOutputContent'][-2000:]}"))
        return scrub(inv["StandardOutputContent"])
    raise RuntimeError(f"remote command {command_id} timed out")


SQL_PY = """
import json, os, sys, psycopg
# DATABASE_URL is a SQLAlchemy URL (postgresql+psycopg://); libpq wants postgresql://.
url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
with psycopg.connect(url, autocommit=True) as c:
    cur = c.execute(sys.argv[1])
    print(json.dumps(cur.fetchall() if cur.description else [], default=str))
"""


def sql(query: str) -> list[list]:
    """A query through the running api container (its DATABASE_URL, verify-full)."""
    out = remote(f"{COMPOSE} exec -T api python -c {_sh(SQL_PY)} {_sh(query)}")
    return json.loads(out.strip().splitlines()[-1])


def _sh(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def scale(workers: int) -> None:
    """Waits for the containers to run, not for healthy: with the CPU at its
    credit baseline the worker healthcheck (a Python start-up) can exceed its
    5 s timeout while the worker itself is processing jobs."""
    remote(f"{COMPOSE} up -d --scale worker={workers} worker && for i in $(seq 120); do "
           f"[ $({COMPOSE} ps -q --status running worker | wc -l) -eq {workers} ] && exit 0; "
           f"sleep 1; done; exit 1")


def reset() -> None:
    if not on_ac_power():
        sys.exit("not on AC power: stopping (results on battery are not comparable)")
    sql("DELETE FROM jobs")
    sql("VACUUM ANALYZE jobs, results, metrics")


def wait_drained(timeout: float = 1800) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ((pending,),) = sql("SELECT count(*) FROM jobs WHERE status IN ('queued', 'running')")
        if pending == 0:
            return
        time.sleep(5)
    raise RuntimeError("jobs still pending after the timeout")


def job_rows() -> list[list]:
    return sql("""
        SELECT extract(epoch FROM j.created_at)::float8, extract(epoch FROM j.started_at)::float8,
               extract(epoch FROM j.finished_at)::float8, r.duration_ms, j.attempts, j.status::text,
               r.attempt
        FROM jobs j LEFT JOIN results r ON r.job_id = j.id""")


# ---- the network ----

def rtt() -> dict:
    """TCP connect time to the API port: one network round trip, no server work."""
    host, port = urlparse(API).hostname, urlparse(API).port
    samples = []
    for _ in range(RTT_SAMPLES):
        t0 = time.perf_counter()
        with socket.create_connection((host, port), timeout=5):
            samples.append((time.perf_counter() - t0) * 1000)
        time.sleep(0.05)
    return {"method": f"TCP connect to {port}/tcp, {RTT_SAMPLES} samples",
            "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "min_ms": min(samples), "p50_ms": pct(samples, 50), "p90_ms": pct(samples, 90),
            "max_ms": max(samples)}


# ---- TLS to RDS ----

TLS_PY = r"""
import os, socket, psycopg
from urllib.parse import urlsplit, parse_qs
# DATABASE_URL is a SQLAlchemy URL (postgresql+psycopg://); libpq wants postgresql://.
url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
parts = urlsplit(url)
q = parse_qs(parts.query)
print("url sslmode:", q["sslmode"][0], "sslrootcert:", q["sslrootcert"][0])
with psycopg.connect(url) as c:
    i = c.info
    opts = {o.keyword.decode(): (o.val or b"").decode() for o in c.pgconn.info}
    print("connected: ssl_in_use", c.pgconn.ssl_in_use, "| libpq sslmode", opts["sslmode"],
          "| sslrootcert", opts["sslrootcert"], "| host", i.host)
    print("pg_stat_ssl:", c.execute("SELECT ssl, version, cipher, bits FROM pg_stat_ssl "
                                    "WHERE pid = pg_backend_pid()").fetchone())
    print("server:", c.execute("SHOW server_version").fetchone()[0])
host = parts.hostname
ip = socket.gethostbyname(host)
for label, bad in [("host as IP (name mismatch)", url.replace(host, ip)),
                   ("system CA bundle, not RDS's",
                    url.replace(q["sslrootcert"][0], "/etc/ssl/certs/ca-certificates.crt"))]:
    try:
        psycopg.connect(bad, connect_timeout=10).close()
        print("NEGATIVE CHECK FAILED, connected:", label)
    except psycopg.Error as e:
        msg = str(e).replace(parts.password, "***")
        print("refused as expected:", label, "->", msg.strip().splitlines()[-1][:200])
"""


def tls() -> dict:
    endpoint = OUT["db_endpoint"].split(":")[0]
    openssl = remote(
        f"openssl s_client -starttls postgres -connect {endpoint}:5432 "
        f"-CAfile /opt/qbt/rds-global-bundle.pem -verify_hostname {endpoint} "
        f"-verify_return_error -showcerts </dev/null 2>&1 | "
        f"grep -E '^ *[0-9] s:|^ *i:|Verify return code|^Protocol|^ *Protocol|Cipher is' || true")
    client = remote(f"{COMPOSE} exec -T api python -c {_sh(TLS_PY)}")
    data = {"measurement": "tls", "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "openssl_s_client": openssl.strip().splitlines(),
            "api_container_psycopg": client.strip().splitlines()}
    save("tls", data)
    return data


# ---- throughput ----

def throughput_once(workers: int, first_index: int) -> dict:
    reset()
    scale(workers)
    remote(f"{COMPOSE} pause worker")
    posted = asyncio.run(loadgen.submit_burst(API, JOBS_PER_WORKER * workers, first_index))
    remote(f"{COMPOSE} unpause worker")
    wait_drained()
    rows = job_rows()
    assert all(r[5] == "succeeded" for r in rows), "not every job succeeded"
    finished = np.sort([r[2] for r in rows])
    lo, hi = int(0.1 * len(finished)), int(0.9 * len(finished)) - 1
    rate = (hi - lo) / (finished[hi] - finished[lo]) * 60
    return {"workers": workers, "jobs": len(rows), "post_errors": posted.errors,
            "jobs_per_minute": rate,
            "backtest_ms_median": float(np.median([r[3] for r in rows])),
            "backtest_ms_p90": pct([r[3] for r in rows], 90),
            "wall_seconds": float(finished[-1] - min(r[1] for r in rows))}


def throughput() -> dict:
    net, runs = rtt(), []
    for i, workers in enumerate(THROUGHPUT_WORKERS):
        log(f"throughput: {workers} worker(s), {JOBS_PER_WORKER * workers} jobs")
        run = guarded(f"throughput {workers}", throughput_once, workers, 100_000 * (i + 1))
        log(f"  {run['jobs_per_minute']:.0f} jobs/min, backtest median "
            f"{run['backtest_ms_median']:.0f} ms")
        runs.append(run)
    data = {"measurement": "throughput", "rtt": net,
            "config": {"workers": THROUGHPUT_WORKERS, "jobs_per_worker": JOBS_PER_WORKER,
                       "rate_over": "middle 80% of completions (database finished_at)",
                       "reps": 1},
            "runs": runs}
    save("throughput", data)
    return data


def capacity(workers: int) -> float:
    run = next(r for r in load("throughput")["runs"] if r["workers"] == workers)
    return run["jobs_per_minute"] / 60


# ---- queue latency ----

def latency_once(rate: float) -> dict:
    reset()
    scale(LATENCY_WORKERS)
    posted = asyncio.run(loadgen.submit_poisson(API, rate, LATENCY_SECONDS, seed=2000))
    wait_drained()
    rows = job_rows()
    assert all(r[5] == "succeeded" for r in rows)
    start = min(r[0] for r in rows)
    measured = [r for r in rows if r[0] >= start + WARMUP_SECONDS]
    created = np.sort([r[0] for r in rows])
    finished = np.sort([r[2] for r in rows])
    times = np.arange(start + WARMUP_SECONDS, created[-1], 1.0)
    backlog = (np.searchsorted(created, times, side="right")
               - np.searchsorted(finished, times, side="right"))
    slope = float(np.polyfit(times - times[0], backlog, 1)[0]) if len(times) > 2 else 0.0
    post_ms = [s.latency_ms for s in posted.samples]
    return {
        "target_rate_per_s": rate, "submitted": len(posted.samples),
        "post_errors": posted.errors, "measured_jobs": len(measured),
        "backlog_growth_jobs_per_s": slope, "queue_growing": slope > 0.02 * rate,
        "backlog_max": int(backlog.max()) if len(backlog) else None,
        "submit_to_start_ms_p50": pct([(r[1] - r[0]) * 1000 for r in measured], 50),
        "submit_to_start_ms_p99": pct([(r[1] - r[0]) * 1000 for r in measured], 99),
        "submit_to_done_ms_p50": pct([(r[2] - r[0]) * 1000 for r in measured], 50),
        "submit_to_done_ms_p99": pct([(r[2] - r[0]) * 1000 for r in measured], 99),
        "post_ms_p50_at_laptop": pct(post_ms, 50), "post_ms_p99_at_laptop": pct(post_ms, 99),
        "max_send_lag_ms": posted.max_send_lag_ms,
    }


def latency() -> dict:
    rate = LATENCY_LOAD * capacity(LATENCY_WORKERS)
    net = rtt()
    log(f"latency: {LATENCY_WORKERS} workers, Poisson {rate:.2f} jobs/s for {LATENCY_SECONDS}s")
    run = guarded("latency", latency_once, rate)
    log(f"  submit-to-start p50 {run['submit_to_start_ms_p50']:.0f} ms, p99 "
        f"{run['submit_to_start_ms_p99']:.0f} ms; growing {run['queue_growing']}")
    data = {"measurement": "latency", "rtt": net,
            "config": {"workers": LATENCY_WORKERS, "load": LATENCY_LOAD,
                       "seconds": LATENCY_SECONDS, "warmup_seconds": WARMUP_SECONDS,
                       "queue_times_from": "database timestamps (no network in them)",
                       "post_ms_from": "the laptop (includes the network)", "reps": 1},
            "runs": [run]}
    save("latency", data)
    return data


# ---- GET /backtests/{id} ----

def get_once(ids: list[str]) -> dict:
    run = asyncio.run(loadgen.get_fixed_rate(API, ids, GET_RATE, GET_SECONDS, seed=7))
    ok = [s.latency_ms for s in run.samples if s.status == 200]
    return {"requests": len(run.samples), "errors": len(run.samples) - len(ok),
            "p50_ms": pct(ok, 50), "p90_ms": pct(ok, 90), "p99_ms": pct(ok, 99),
            "max_ms": max(ok), "max_send_lag_ms": run.max_send_lag_ms}


def get() -> dict:
    ids = [r[0] for r in sql("SELECT id::text FROM jobs WHERE status = 'succeeded' LIMIT 500")]
    if not ids:
        raise RuntimeError("no succeeded jobs to GET; run latency first")
    net = rtt()
    log(f"get: {GET_RATE} req/s for {GET_SECONDS}s over {len(ids)} jobs, workers idle")
    run = guarded("get", get_once, ids)
    log(f"  p50 {run['p50_ms']:.1f} ms, p99 {run['p99_ms']:.1f} ms (RTT p50 {net['p50_ms']:.1f})")
    data = {"measurement": "get", "rtt": net,
            "config": {"rate_per_s": GET_RATE, "seconds": GET_SECONDS, "workers": "idle",
                       "latency_from": "scheduled send time at the laptop (open loop)"},
            "runs": [run]}
    save("get", data)
    return data


# ---- POST /predict ----

def predict_once() -> dict:
    sql("DELETE FROM predictions")
    remote(f"{COMPOSE} restart api && {COMPOSE} up -d --wait --wait-timeout 120 api")
    with httpx2.Client(base_url=API, timeout=30) as client:
        t0 = time.perf_counter()
        cold = client.post("/predict", json={})
        cold_ms = (time.perf_counter() - t0) * 1000
    warm = asyncio.run(loadgen.predict_fixed_rate(API, PREDICT_RATE, PREDICT_WARMUP_SECONDS))
    run = asyncio.run(loadgen.predict_fixed_rate(API, PREDICT_RATE, PREDICT_SECONDS))
    ok = [s.latency_ms for s in run.samples if s.status == 200]
    oks = (cold.status_code == 200) + sum(s.status == 200 for s in warm.samples) + len(ok)
    ((logged,),) = sql("SELECT count(*) FROM predictions")
    return {"cold_first_request_ms": cold_ms, "cold_status": cold.status_code,
            "requests": len(run.samples), "errors": len(run.samples) - len(ok),
            "p50_ms": pct(ok, 50), "p90_ms": pct(ok, 90), "p99_ms": pct(ok, 99),
            "max_ms": max(ok), "max_send_lag_ms": run.max_send_lag_ms,
            "responses_200": oks, "rows_logged": logged, "every_200_logged_once": oks == logged}


def predict() -> dict:
    net = rtt()
    log(f"predict: {PREDICT_RATE} req/s for {PREDICT_SECONDS}s")
    run = guarded("predict", predict_once)
    log(f"  p50 {run['p50_ms']:.1f} ms, p99 {run['p99_ms']:.1f} ms, cold "
        f"{run['cold_first_request_ms']:.0f} ms (RTT p50 {net['p50_ms']:.1f})")
    model = json.loads(httpx2.get(f"{API}/models/1", timeout=30).text)
    data = {"measurement": "predict", "rtt": net,
            "config": {"rate_per_s": PREDICT_RATE, "seconds": PREDICT_SECONDS,
                       "warmup_seconds": PREDICT_WARMUP_SECONDS, "workers": "idle",
                       "api_processes": 1,
                       "cold_start": "first request after `docker compose restart api`",
                       "latency_from": "scheduled send time at the laptop (open loop)"},
            "model": {k: model.get(k) for k in ("version", "model_type", "target")}
                     | {"out_of_sample": model.get("out_of_sample"),
                        "baseline": model.get("baseline")},
            "runs": [run]}
    save("predict", data)
    return data


# ---- kill a worker ----

FOLLOW_PY = r"""
import json, sys, time, urllib.request
killed_at, job_ids = float(sys.argv[1]), sys.argv[2:]
out = {}
for job in job_ids:
    seen, deadline = {}, time.time() + 300
    while time.time() < deadline:
        with urllib.request.urlopen(f"http://127.0.0.1:8000/backtests/{job}") as r:
            j = json.load(r)
        now = time.time() - killed_at
        if j["status"] == "queued" and "requeued" not in seen:
            seen["requeued"], seen["requeue_reason"] = now, j["last_error"]
        if j["status"] == "running" and j["attempts"] >= 2 and "reclaimed" not in seen:
            seen.setdefault("requeued", now); seen["reclaimed"] = now
        if j["status"] in ("succeeded", "failed"):
            seen.setdefault("requeued", now); seen.setdefault("reclaimed", now)
            seen.update(finished=now, final_status=j["status"], attempts=j["attempts"])
            break
        time.sleep(0.1)
    out[job] = seen
print(json.dumps(out))
"""

KILL_SH = r"""
set -euo pipefail
C="{compose}"
workers=$($C ps -q --status running worker)
# Kill the first worker whose last log line is "claimed job <id>": it is
# running that job now. Reading the log needs no Python start-up, so the gap
# between seeing the claim and the SIGKILL is milliseconds.
victim=""; job=""
for i in $(seq 600); do
  for c in $workers; do
    last=$(docker logs --tail 1 "$c" 2>&1)
    if [[ "$last" == *"claimed job "* ]]; then
      docker kill "$c" >/dev/null; killed_at=$(date +%s.%N)
      victim=$c; job=$(sed -E 's/.*claimed job ([0-9a-f-]+).*/\1/' <<< "$last"); break 2
    fi
  done
  sleep 0.1
done
if [ -z "$victim" ]; then echo '{{"victim": null}}'; exit 0; fi
# In flight only if it is still running (leased to the dead worker).
get_status='import json,sys; print(json.load(sys.stdin)["status"])'
status=$(curl -s "http://127.0.0.1:8000/backtests/$job" | python3 -c "$get_status")
if [ "$status" != "running" ]; then job=""; fi
printf '{{"victim": "%s", "killed_at": %s, "jobs": "%s", "status_after_kill": "%s"}}\n' \
  "${{victim:0:12}}" "$killed_at" "$job" "$status"
if [ -n "$job" ]; then python3 -c '{follow_py}' "$killed_at" $job; fi
"""


def kill_once(rep: int) -> dict:
    reset()
    scale(KILL_WORKERS)
    since = datetime.now(UTC).isoformat()
    rate = KILL_LOAD * capacity(KILL_WORKERS)
    box: dict[str, Any] = {}
    def submit() -> None:
        box["backlog"] = asyncio.run(loadgen.submit_burst(API, KILL_BACKLOG,
                                                          first_index=900_000 + 1000 * rep))
        box["run"] = asyncio.run(loadgen.submit_poisson(API, rate, KILL_SECONDS, seed=500 + rep))

    load_thread = threading.Thread(target=submit)
    load_thread.start()
    time.sleep(KILL_AT)
    script = KILL_SH.format(compose=COMPOSE, follow_py=FOLLOW_PY.replace("'", "'\"'\"'"))
    attempts: list[dict] = []
    followed: dict = {}
    for _ in range(5):
        lines = remote(script).strip().splitlines()
        head = json.loads(lines[0])
        attempts.append(head)
        if head.get("victim") and head.get("jobs"):
            followed = json.loads(lines[-1])
            break
        log(f"  no job in flight at the kill attempt ({head}); retrying in 5 s")
        if head.get("victim"):
            scale(KILL_WORKERS)
        time.sleep(5)
    load_thread.join()
    wait_drained()
    scale(KILL_WORKERS)  # bring the killed worker back
    posted = box["run"]
    victims = sorted(followed)

    submitted = {s.job_id for run in (box["backlog"], posted) for s in run.samples if s.job_id}
    rows = sql("""SELECT j.id::text, j.status::text, j.attempts, r.attempt,
                         (SELECT count(*) FROM metrics m WHERE m.job_id = j.id)
                  FROM jobs j LEFT JOIN results r ON r.job_id = j.id""")
    stored = {r[0] for r in rows}
    logs = remote(f"{COMPOSE} logs --no-log-prefix --since {since} worker 2>&1")
    checks = {
        "posts_accepted": len(submitted),
        "post_errors": posted.errors + box["backlog"].errors,
        "jobs_in_db": len(rows),
        "lost": len(submitted - stored), "unexpected_extra_jobs": len(stored - submitted),
        "not_succeeded": sum(1 for r in rows if r[1] != "succeeded"),
        "without_result": sum(1 for r in rows if r[3] is None),
        "result_attempt_mismatch": sum(1 for r in rows if r[3] is not None and r[3] != r[2]),
        "jobs_with_more_than_one_attempt": sorted(r[0] for r in rows if r[2] > 1),
        "metric_rows_per_job": sorted({r[4] for r in rows}),
        "fencing_rejections_in_logs": logs.count("lease_lost"),
        "results_discarded_in_logs": logs.count("result discarded"),
        "reaper_requeues_in_logs": logs.count("its lease expired"),
    }
    ok = (checks["lost"] == checks["unexpected_extra_jobs"] == checks["not_succeeded"]
          == checks["without_result"] == checks["result_attempt_mismatch"] == 0
          and checks["jobs_with_more_than_one_attempt"] == victims
          and len(checks["metric_rows_per_job"]) == 1 and bool(victims))
    return {"rep": rep + 1, "load_rate_per_s": rate, "kill_attempts": attempts,
            "victims": followed, "checks": checks, "all_checks_pass": ok}


def kill() -> dict:
    net, runs = rtt(), []
    for rep in range(KILL_REPS):
        log(f"kill rep {rep + 1}/{KILL_REPS}")
        run = guarded("kill", kill_once, rep)
        for job, t in run["victims"].items():
            log(f"  job {job[:8]}: requeued {t['requeued']:.1f}s, reclaimed {t['reclaimed']:.1f}s, "
                f"finished {t['finished']:.1f}s after the kill")
        log(f"  all checks pass: {run['all_checks_pass']}")
        runs.append(run)
    data = {"measurement": "kill", "rtt": net,
            "config": {"workers": KILL_WORKERS, "backlog_jobs_first": KILL_BACKLOG,
                       "load": KILL_LOAD, "seconds": KILL_SECONDS,
                       "kill_at_seconds": KILL_AT, "signal": "SIGKILL (docker kill, on the "
                       "instance)", "times_from": "the instance's clock: kill to status "
                       "changes seen polling the API on localhost every 0.1 s",
                       "lease_seconds": 30, "reaper_seconds": 15, "reps": KILL_REPS},
            "all_checks_pass": all(r["all_checks_pass"] for r in runs), "runs": runs}
    save("kill", data)
    return data


# ---- CloudWatch: CPU and credits on the instance ----

def cloudwatch(since: str) -> dict:
    start = datetime.fromisoformat(since)
    end = datetime.now(UTC) + timedelta(minutes=1)
    series = {}
    for metric in ("CPUUtilization", "CPUCreditBalance", "CPUCreditUsage"):
        points = aws("cloudwatch", "get-metric-statistics", "--namespace", "AWS/EC2",
                     "--metric-name", metric, "--dimensions", f"Name=InstanceId,Value={INSTANCE}",
                     "--start-time", start.isoformat(), "--end-time", end.isoformat(),
                     "--period", "300", "--statistics", "Average", "Maximum")["Datapoints"]
        series[metric] = sorted(points, key=lambda p: p["Timestamp"])
    data = {"measurement": "cloudwatch", "instance": INSTANCE, "period_seconds": 300,
            "series": series}
    save("cloudwatch", data)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("step", choices=("rtt", "tls", "throughput", "latency", "get",
                                         "predict", "kill", "cloudwatch", "environment"))
    parser.add_argument("--since", help="cloudwatch: start time (ISO 8601, UTC)")
    args = parser.parse_args()
    if args.step == "environment":
        instance = aws("ec2", "describe-instances", "--instance-ids", INSTANCE)
        inst = instance["Reservations"][0]["Instances"][0]
        db = aws("rds", "describe-db-instances", "--db-instance-identifier", "qbt-db")
        dbi = db["DBInstances"][0]
        save("environment", {
            "client": "this laptop, over the public internet (home connection) to us-east-2",
            "laptop": environment(), "region": REGION,
            "instance": {k: inst.get(k) for k in ("InstanceId", "InstanceType", "ImageId",
                                                  "Placement", "LaunchTime")}
                        | {"cpu_credits": "standard"},
            "database": {k: dbi.get(k) for k in ("DBInstanceClass", "Engine", "EngineVersion",
                                                 "AllocatedStorage", "StorageType",
                                                 "CACertificateIdentifier", "MultiAZ")}})
        return 0
    if args.step not in ("rtt", "tls", "cloudwatch") and not on_ac_power():
        sys.exit("not on AC power")
    with KeepAwake():
        if args.step == "rtt":
            print(json.dumps(rtt(), indent=2))
        elif args.step == "cloudwatch":
            cloudwatch(args.since)
        else:
            globals()[args.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
