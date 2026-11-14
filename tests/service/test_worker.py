"""
The worker against a real Postgres and real processes.

In-process tests drive a Worker (with its real compute subprocess) one
job at a time: success with a result identical to running qbt.runner
directly, each failure kind mapped to retry or no-retry, heartbeats
keeping a slow job's lease, a lost lease abandoning the job, the job
timeout, graceful shutdown releasing the job, and the startup C++ check.

Process tests run `python -m qbt_service.worker` itself: a worker
SIGKILLed mid-job (supervisor and compute child together, as when a
container dies) has its job finished by another worker, on attempt 2,
with a result identical to an uninterrupted run; killing only the
supervisor doesn't leave its compute process behind; SIGTERM exits 0.

QBT_TEST_COMPUTE_DELAY_SECONDS makes jobs slow enough to interrupt.
"""

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select, text, update

import qbt.bootstrap as qb
from qbt.runner import metrics_equal, run
from qbt_service.catalog import sync_catalog
from qbt_service.compute import Handshake, Task, compute
from qbt_service.db import jobs, metrics, results
from qbt_service.models import parse_request, request_hash
from qbt_service.prices import load_frame
from qbt_service.queue import enqueue
from qbt_service.settings import Settings
from qbt_service.worker import StartupError, Worker

from .conftest import base_url, migrate, temporary_database
from .factories import store_synthetic_series, synthetic_frame

pytestmark = pytest.mark.db

START, END = "2018-01-01", "2021-01-01"  # covers the whole synthetic series
FAST = {"lease_seconds": 3.0, "heartbeat_seconds": 1.0, "reaper_seconds": 0.5,
        "poll_seconds": 0.1, "retry_backoff_seconds": 0.1}


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.05, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


class ModuleDB:
    """The module's database: its engine plus the ids every test needs."""

    def __init__(self, engine, **info):
        self.engine, self.info = engine, info

    def begin(self):
        return self.engine.begin()

    def connect(self):
        return self.engine.connect()


@pytest.fixture(scope="module")
def db():
    """One migrated database for this module, with a price series and the catalog."""
    with temporary_database(base_url()) as engine:
        migrate(engine)
        with engine.begin() as conn:
            series_id = store_synthetic_series(conn)
            strategy_ids = sync_catalog(conn)
        yield ModuleDB(engine, series_id=series_id, strategy_ids=strategy_ids,
                       url=engine.url.render_as_string(hide_password=False))


@pytest.fixture(autouse=True)
def empty_queue(db):
    with db.begin() as conn:
        conn.execute(text("DELETE FROM jobs"))


def settings(db, **overrides) -> Settings:
    return Settings(database_url=db.info["url"], worker_id=f"test-{uuid.uuid4().hex[:6]}",
                    **(FAST | overrides))


def submit(db, strategy="sma_crossover", **fields) -> uuid.UUID:
    request = parse_request({"strategy": strategy, "ticker": "SPY", "start": START,
                             "end": END, "n_boot": 500} | fields)
    with db.begin() as conn:
        return enqueue(conn, strategy_id=db.info["strategy_ids"][strategy],
                       price_series_id=db.info["series_id"],
                       request=request.model_dump(mode="json"),
                       request_hash=request_hash(request)).job_id


def job(db, job_id):
    with db.connect() as conn:
        return conn.execute(select(jobs).where(jobs.c.id == job_id)).mappings().one()


def stored_result(db, job_id):
    with db.connect() as conn:
        row = conn.execute(select(results).where(results.c.job_id == job_id)).mappings().one()
        values = dict(conn.execute(select(metrics.c.name, metrics.c.value)
                                   .where(metrics.c.job_id == job_id)).tuples().all())
    return row, values


def reference(db, strategy="sma_crossover", n_boot=500):
    """The same backtest run directly in this process."""
    with db.connect() as conn:
        frame = load_frame(conn, db.info["series_id"])
    request = parse_request({"strategy": strategy, "ticker": "SPY", "start": START,
                             "end": END, "n_boot": n_boot})
    return run(frame, strategy, request.params.model_dump(), cost_bps=request.cost_bps,
               n_boot=request.n_boot, seed=request.seed)


def assert_identical_to_reference(db, job_id, attempt):
    row, values = stored_result(db, job_id)
    expected = reference(db)
    assert row["attempt"] == attempt
    assert metrics_equal(values, expected.metrics)
    assert row["equity"] == expected.result["equity"].tolist()
    assert row["position"] == expected.result["position"].astype(float).tolist()
    assert row["dates"] == [ts.date() for ts in expected.result.index]
    assert row["bootstrap_impl"] == expected.bootstrap_impl
    assert row["numpy_version"] == np.__version__


@pytest.fixture(scope="module")
def worker(db):
    """A started Worker with default (fast) settings, shared by the simple tests."""
    w = Worker(settings(db))
    w.start()
    yield w
    w.stop()


def run_in_thread(worker):
    box = {}
    thread = threading.Thread(target=lambda: box.setdefault("outcome", worker.run_once()))
    thread.start()

    def join(timeout=60):
        thread.join(timeout)
        assert not thread.is_alive()
        return box["outcome"]

    return join


def wait_until_running(db, job_id, seconds_in=0.3):
    """Until the job is claimed and has been running a moment (the compute
    child is inside the backtest, not still receiving the task)."""
    wait_for(lambda: job(db, job_id)["status"] == "running", what="job to start")
    time.sleep(seconds_in)


# ---- the compute function, in this process ----

def test_compute_outcomes():
    frame = synthetic_frame()
    kind, payload = compute(Task(frame, "momentum", {}, 5.0, 50, 0))
    assert kind == "ok"
    assert payload["numpy_version"] == np.__version__
    assert len(payload["dates"]) == len(payload["equity"]) == len(payload["position"]) == 600
    assert compute(Task(frame, "nope", {}, 5.0, 0, 0))[0] == "invalid"
    kind, trace = compute(Task(frame.iloc[:5], "ml_direction", {}, 5.0, 0, 0))
    assert kind == "error" and "Traceback" in trace


# ---- the in-process worker ----

def test_handshake(worker):
    handshake = worker.compute.handshake
    assert handshake.numpy_version == np.__version__
    assert handshake.cpp_available == qb.cpp_available()
    assert set(handshake.thread_settings.values()) == {"1"}
    assert handshake.pid != os.getpid()


def test_no_job(worker):
    assert worker.run_once() is None


def test_success_is_identical_to_a_direct_run(db, worker):
    job_id = submit(db)
    outcome = worker.run_once()
    assert outcome.label == "succeeded"
    row = job(db, job_id)
    assert (row["status"], row["attempts"], row["last_error"]) == ("succeeded", 1, None)
    assert_identical_to_reference(db, job_id, attempt=1)
    assert stored_result(db, job_id)[0]["qbt_version"] == worker.qbt_version


def test_bad_stored_request_fails_without_retry(db, worker):
    with db.begin() as conn:
        job_id = enqueue(conn, strategy_id=db.info["strategy_ids"]["momentum"],
                         price_series_id=db.info["series_id"],
                         request={"strategy": "regime_switch"}, request_hash=b"\0" * 32).job_id
    assert worker.run_once().label == "failed"
    row = job(db, job_id)
    assert (row["status"], row["attempts"]) == ("failed", 1)
    assert row["last_error"].startswith("invalid request:")


def test_range_without_prices_fails_without_retry(db, worker):
    job_id = submit(db, start="2030-01-01", end="2031-01-01")
    assert worker.run_once().label == "failed"
    assert "has no bars" in job(db, job_id)["last_error"]


def test_backtest_exception_fails_without_retry(db, worker):
    # Valid request, but five bars are too few for the model: it raises.
    job_id = submit(db, strategy="ml_direction", start="2018-01-01", end="2018-01-06")
    assert worker.run_once().label == "failed"
    row = job(db, job_id)
    assert row["status"] == "failed" and row["attempts"] == 1
    assert row["last_error"].startswith("backtest raised:\nTraceback")


def test_compute_crash_is_retried_and_the_retry_matches(db):
    w = Worker(settings(db, test_compute_delay_seconds=3.0))
    w.start()
    try:
        job_id = submit(db)
        first_pid = w.compute.pid
        join = run_in_thread(w)
        wait_until_running(db, job_id)
        os.kill(first_pid, signal.SIGKILL)
        outcome = join()
        assert outcome.label == "retrying"
        assert outcome.detail == "compute process killed by signal 9"
        row = job(db, job_id)
        assert (row["status"], row["attempts"]) == ("queued", 1)
        assert w.compute.pid not in (None, first_pid)  # a fresh child

        wait_for(lambda: w.run_once(), what="retry to be claimed", interval=0.1)
        assert job(db, job_id)["status"] == "succeeded"
        assert_identical_to_reference(db, job_id, attempt=2)
    finally:
        w.stop()


def test_heartbeats_keep_a_slow_job(db):
    # Runs 4.5 s against a 3 s lease: only heartbeats keep it.
    w = Worker(settings(db, test_compute_delay_seconds=4.5))
    w.start()
    try:
        job_id = submit(db)
        assert w.run_once().label == "succeeded"
        row = job(db, job_id)
        assert row["attempts"] == 1
        assert (row["heartbeat_at"] - row["started_at"]).total_seconds() >= 3
    finally:
        w.stop()


def test_lost_lease_abandons_the_job(db):
    w = Worker(settings(db, test_compute_delay_seconds=20.0))
    w.start()
    try:
        job_id = submit(db)
        first_pid = w.compute.pid
        started = time.monotonic()
        join = run_in_thread(w)
        wait_until_running(db, job_id)
        with db.begin() as conn:  # someone else now holds the lease
            conn.execute(update(jobs).where(jobs.c.id == job_id)
                         .values(lease_token=func.gen_random_uuid(), lease_owner="other"))
        outcome = join()
        assert outcome.label == "lease_lost"
        assert time.monotonic() - started < 10  # gave up, didn't wait the job out
        assert w.compute.pid != first_pid
        row = job(db, job_id)
        assert (row["status"], row["lease_owner"]) == ("running", "other")
        with db.connect() as conn:
            assert conn.execute(select(func.count()).select_from(results)).scalar_one() == 0
    finally:
        w.stop()


def test_timeout_fails_without_retry(db):
    w = Worker(settings(db, test_compute_delay_seconds=20.0, job_timeout_seconds=1.0))
    w.start()
    try:
        job_id = submit(db)
        outcome = w.run_once()
        assert (outcome.label, outcome.detail) == ("failed", "timed out after 1s")
        assert job(db, job_id)["status"] == "failed"
    finally:
        w.stop()


def test_shutdown_lets_the_job_finish_within_grace(db):
    w = Worker(settings(db, test_compute_delay_seconds=1.5, shutdown_grace_seconds=30.0))
    w.start()
    try:
        job_id = submit(db)
        join = run_in_thread(w)
        wait_until_running(db, job_id)
        w.request_stop()
        assert join().label == "succeeded"
        assert w.stopping
        w.run()  # returns at once: no more claiming
    finally:
        w.stop()


def test_shutdown_releases_the_job_after_grace(db):
    w = Worker(settings(db, test_compute_delay_seconds=20.0, shutdown_grace_seconds=0.5))
    w.start()
    try:
        job_id = submit(db)
        join = run_in_thread(w)
        wait_until_running(db, job_id)
        w.request_stop()
        assert join().label == "released"
        row = job(db, job_id)
        assert (row["status"], row["attempts"], row["lease_token"]) == ("queued", 0, None)
    finally:
        w.stop()


def test_second_stop_request_releases_at_once(db):
    w = Worker(settings(db, test_compute_delay_seconds=20.0, shutdown_grace_seconds=60.0))
    w.start()
    try:
        job_id = submit(db)
        join = run_in_thread(w)
        wait_until_running(db, job_id)
        started = time.monotonic()
        w.request_stop()
        w.request_stop()
        assert join().label == "released"
        assert time.monotonic() - started < 5
    finally:
        w.stop()


def _fake_start(cpp_available):
    def start():
        return Handshake(pid=0, numpy_version=np.__version__, cpp_available=cpp_available,
                         cpp_unavailable_reason="extension not built",
                         thread_settings={})
    return start


def test_startup_requires_the_cpp_bootstrap(db, monkeypatch):
    w = Worker(settings(db))
    monkeypatch.setattr(w.compute, "start", _fake_start(cpp_available=False))
    with pytest.raises(StartupError, match="QBT_ALLOW_NUMPY_FALLBACK=1"):
        w.start()
    w.engine.dispose()


def test_startup_fallback_can_be_allowed(db, monkeypatch, caplog):
    w = Worker(settings(db, allow_numpy_fallback=True))
    monkeypatch.setattr(w.compute, "start", _fake_start(cpp_available=False))
    w.start()
    assert "running without the C++ bootstrap" in caplog.text
    w.engine.dispose()


# ---- real worker processes ----

class WorkerProcess:
    """`python -m qbt_service.worker` in its own process group, logging to a file."""

    def __init__(self, db, log_path: Path, **env):
        self.log_path = log_path
        environment = os.environ | {
            "DATABASE_URL": db.info["url"], "QBT_WORKER_ID": log_path.stem,
            "QBT_LEASE_SECONDS": "3", "QBT_HEARTBEAT_SECONDS": "1", "QBT_REAPER_SECONDS": "0.5",
            "QBT_POLL_SECONDS": "0.1", "QBT_RETRY_BACKOFF_SECONDS": "0.1",
        } | env
        self._log = log_path.open("w")
        self.proc = subprocess.Popen([sys.executable, "-m", "qbt_service.worker"],
                                     env=environment, stdout=self._log,
                                     stderr=subprocess.STDOUT, start_new_session=True)

    def log(self) -> str:
        return self.log_path.read_text()

    def compute_pid(self) -> int:
        line = wait_for(lambda: next((ln for ln in self.log().splitlines()
                                      if "compute process pid=" in ln), None),
                        timeout=60, what=f"{self.log_path.stem} to start")
        return int(line.split("pid=")[1].split()[0])

    def wait(self, timeout=30) -> int:
        try:
            return self.proc.wait(timeout)
        finally:
            self._log.close()

    def kill_group(self):
        """SIGKILL the supervisor and its compute child at once, like a dead container."""
        os.killpg(self.proc.pid, signal.SIGKILL)


def gone(pid: int) -> bool:
    """True once `pid` has exited (a zombie awaiting its reaper counts as gone)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def test_sigkilled_worker_job_is_finished_by_another_worker(db, tmp_path):
    job_id = submit(db)

    a = WorkerProcess(db, tmp_path / "worker-a.log", QBT_TEST_COMPUTE_DELAY_SECONDS="20")
    a_child = a.compute_pid()
    wait_until_running(db, job_id)
    assert job(db, job_id)["lease_owner"] == "worker-a"
    a.kill_group()
    assert a.wait() == -signal.SIGKILL
    wait_for(lambda: gone(a_child), what="worker-a's compute process to exit")
    assert job(db, job_id)["status"] == "running"  # nobody knows yet

    b = WorkerProcess(db, tmp_path / "worker-b.log")
    try:
        wait_for(lambda: job(db, job_id)["status"] == "succeeded", timeout=60,
                 what="worker-b to finish the job")
        row = job(db, job_id)
        assert row["attempts"] == 2
        assert "requeued job" in b.log() and "lease expired" in b.log()
        with db.connect() as conn:
            assert conn.execute(select(func.count()).select_from(results)).scalar_one() == 1
        assert_identical_to_reference(db, job_id, attempt=2)

        # And identical to an uninterrupted run of the same request by a worker.
        clean_id = submit(db)
        wait_for(lambda: job(db, clean_id)["status"] == "succeeded", timeout=60,
                 what="the uninterrupted run")
        (killed, killed_metrics), (clean, clean_metrics) = (
            stored_result(db, job_id), stored_result(db, clean_id))
        assert metrics_equal(killed_metrics, clean_metrics)
        assert (killed["equity"], killed["position"], killed["dates"]) == (
            clean["equity"], clean["position"], clean["dates"])
    finally:
        b.proc.send_signal(signal.SIGTERM)
        assert b.wait() == 0
    assert "stopped" in b.log()


def test_killing_only_the_supervisor_ends_its_compute_process(db, tmp_path):
    w = WorkerProcess(db, tmp_path / "worker-c.log")
    child = w.compute_pid()
    w.proc.send_signal(signal.SIGKILL)
    w.wait()
    wait_for(lambda: gone(child), timeout=10, what="the orphaned compute process to exit")


def test_sigterm_exits_cleanly(db, tmp_path):
    w = WorkerProcess(db, tmp_path / "worker-d.log")
    child = w.compute_pid()
    w.proc.send_signal(signal.SIGTERM)
    assert w.wait() == 0
    wait_for(lambda: gone(child), timeout=10, what="the compute process to exit")
    assert "stop requested" in w.log()


def test_unreachable_database_means_no_claim_not_a_crash(caplog):
    # Port 1 on localhost refuses at once. The worker logs and carries on
    # polling; nothing is claimed, so nothing can be lost.
    w = Worker(Settings(database_url="postgresql+psycopg://qbt:qbt@127.0.0.1:1/qbt"))
    assert w.run_once() is None
    assert "claim failed" in caplog.text
    w.engine.dispose()
