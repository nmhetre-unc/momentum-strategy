"""
The backtest worker.

    python -m qbt_service.worker

Two processes. The supervisor (this module) talks to the database: it
claims jobs, loads their prices, heartbeats their leases, and writes
their outcomes. A long-lived compute subprocess (qbt_service.compute)
runs the backtests. Splitting them means:

- heartbeats keep flowing however long a backtest holds the GIL;
- a job whose lease is lost, that times out, or that must stop at
  shutdown is ended by killing the child, which can't write anything;
- a crash or out-of-memory kill takes out only the child; the supervisor
  records it and starts a new one.

Outcomes:

    succeeded                      complete() (fenced: a lost lease writes nothing)
    bad request / no prices        fail, no retry
    backtest raised                fail, no retry (deterministic: it would raise again)
    backtest exceeded the timeout  fail, no retry (it would run as long again)
    compute process died           fail, retry with backoff (may be transient)
    database error                 fail, retry if possible; otherwise the lease expires
                                   and the reaper requeues the job
    lease lost                     abandon: someone else owns the job now
    shutdown                       release: the job is requeued, its attempt given back

Startup check: the worker refuses to start unless the compute process can
use the C++ bootstrap, because the NumPy fallback, while bit-identical,
is about 9x slower; QBT_ALLOW_NUMPY_FALLBACK=1 overrides that.

On SIGTERM or SIGINT the worker stops claiming, gives the current job up
to QBT_SHUTDOWN_GRACE_SECONDS to finish, then releases it. A second
signal releases it at once.
"""

import contextlib
import importlib.metadata
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import FrameType
from typing import Any

from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError

from qbt_service.compute import THREAD_VARIABLES, Handshake, Task, child_main
from qbt_service.db import make_engine
from qbt_service.models import parse_request
from qbt_service.prices import load_frame
from qbt_service.queue import (
    ClaimedJob,
    JobResult,
    claim,
    complete,
    fail,
    heartbeat,
    reap_expired,
    release,
)
from qbt_service.settings import Settings

log = logging.getLogger("qbt_service.worker")

STARTUP_TIMEOUT_SECONDS = 120.0  # the child imports pandas, sklearn, ...
POLL_INTERVAL_SECONDS = 0.05     # how often a running job checks for news


class StartupError(RuntimeError):
    pass


class ComputeDied(RuntimeError):
    pass


class ComputeProcess:
    """The long-lived compute child, restartable."""

    def __init__(self, delay_seconds: float = 0.0):
        self._context = multiprocessing.get_context("spawn")
        self._delay = delay_seconds
        self._process: BaseProcess | None = None
        self._conn: Connection | None = None
        self.handshake: Handshake | None = None

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    def start(self) -> Handshake:
        parent_end, child_end = self._context.Pipe()
        process = self._context.Process(target=child_main, args=(child_end, self._delay),
                                        name="qbt-compute", daemon=True)
        # A spawned child copies this process's environment as it starts,
        # and its numpy reads these before loading a BLAS: one thread, so
        # one fixed order of floating-point sums. Restored afterwards so
        # the supervisor's own environment is left as it was.
        saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
        os.environ.update(dict.fromkeys(THREAD_VARIABLES, "1"))
        try:
            process.start()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        child_end.close()
        self._process, self._conn = process, parent_end
        if not parent_end.poll(STARTUP_TIMEOUT_SECONDS):
            self.kill()
            raise StartupError("compute process did not start")
        kind, handshake = parent_end.recv()
        assert kind == "ready", kind
        self.handshake = handshake
        log.info("compute process pid=%d ready (numpy %s, C++ bootstrap %s)", handshake.pid,
                 handshake.numpy_version, "yes" if handshake.cpp_available else "no")
        return handshake

    def submit(self, task: Task) -> None:
        assert self._conn is not None
        self._conn.send(task)

    def poll(self, timeout: float) -> tuple[str, Any] | None:
        """The outcome if one arrived within `timeout`; raises ComputeDied."""
        assert self._conn is not None and self._process is not None
        try:
            if self._conn.poll(timeout):
                return self._conn.recv()
        except (EOFError, OSError) as exc:
            raise ComputeDied(self._death_reason()) from exc
        if not self._process.is_alive():
            raise ComputeDied(self._death_reason())
        return None

    def _death_reason(self) -> str:
        assert self._process is not None
        self._process.join(timeout=1)
        code = self._process.exitcode
        if code is not None and code < 0:
            return f"compute process killed by signal {-code}"
        return f"compute process exited with code {code}"

    def kill(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.join(timeout=10)
        if self._conn is not None:
            self._conn.close()
        self._process = self._conn = None

    def restart(self) -> None:
        self.kill()
        self.start()

    def close(self) -> None:
        """Asks the child to exit; kills it if it doesn't."""
        if self._conn is not None and self._process is not None:
            with contextlib.suppress(OSError):  # already gone
                self._conn.send(None)
            self._process.join(timeout=5)
        self.kill()


class Liveness:
    """
    Touches QBT_HEALTH_FILE after each successful claim poll and lease
    heartbeat. Both talk to the database, so a fresh file means the worker
    loop is turning and the database answers; `python -m qbt_service.health
    worker` (the container healthcheck) checks its age.
    """

    def __init__(self, path: str | None):
        self.path = path

    def touch(self) -> None:
        if self.path is None:
            return
        try:
            with open(self.path, "a"):
                pass
            os.utime(self.path)
        except OSError as exc:  # never let the healthcheck file stop the work
            log.warning("could not touch health file %s: %s", self.path, exc)


class Heartbeat(threading.Thread):
    """Renews one job's lease every heartbeat_seconds until stopped or lost."""

    def __init__(self, engine: Engine, job: ClaimedJob, settings: Settings,
                 liveness: Liveness | None = None):
        super().__init__(name=f"heartbeat-{job.id}", daemon=True)
        self._engine, self._job, self._settings = engine, job, settings
        self._liveness = liveness or Liveness(None)
        self._stopped = threading.Event()
        self.lost = threading.Event()

    def run(self) -> None:
        while not self._stopped.wait(self._settings.heartbeat_seconds):
            try:
                with self._engine.begin() as conn:
                    alive = heartbeat(conn, self._job.id, self._job.lease_token,
                                      self._settings.lease_seconds)
            except SQLAlchemyError as exc:
                # Keep trying: if the database stays away, the lease
                # expires and the next successful heartbeat reports it lost.
                log.warning("heartbeat for job %s failed: %s", self._job.id, exc)
                continue
            if not alive:
                self.lost.set()
                return
            self._liveness.touch()

    def stop(self) -> None:
        self._stopped.set()
        self.join(timeout=self._settings.heartbeat_seconds + 5)


@dataclass(frozen=True)
class Outcome:
    label: str     # succeeded, failed, retrying, lease_lost, released, db_error
    detail: str = ""


class Worker:
    def __init__(self, settings: Settings, engine: Engine | None = None):
        self.settings = settings
        self.engine = engine or make_engine(settings.database_url,
                                            settings.statement_timeout_seconds)
        self.compute = ComputeProcess(settings.test_compute_delay_seconds)
        self.liveness = Liveness(settings.health_file)
        self.qbt_version = importlib.metadata.version("qbt")
        self._wake = threading.Event()
        self._stopping = False
        self._release_now = False
        self._grace_deadline = 0.0
        self._next_reap = 0.0

    # ---- lifecycle ----

    def start(self) -> None:
        handshake = self.compute.start()
        if not handshake.cpp_available:
            if not self.settings.allow_numpy_fallback:
                self.compute.close()
                raise StartupError(
                    "the C++ bootstrap is unavailable in the compute process "
                    f"({handshake.cpp_unavailable_reason}); set QBT_ALLOW_NUMPY_FALLBACK=1 "
                    "to run on the bit-identical but ~9x slower NumPy implementation"
                )
            log.warning("running without the C++ bootstrap: %s",
                        handshake.cpp_unavailable_reason)

    def stop(self) -> None:
        self.compute.close()
        self.engine.dispose()

    def request_stop(self) -> None:
        """First call: stop claiming, let the current job use the grace period.
        Second call: release the current job at once."""
        if self._stopping:
            self._release_now = True
            log.info("second stop request: releasing the current job now")
        else:
            self._stopping = True
            self._grace_deadline = time.monotonic() + self.settings.shutdown_grace_seconds
            log.info("stop requested: finishing the current job (grace %.0fs)",
                     self.settings.shutdown_grace_seconds)
        self._wake.set()

    @property
    def stopping(self) -> bool:
        return self._stopping

    def run(self) -> None:
        while not self._stopping:
            if self.run_once() is None:
                self._wake.wait(self.settings.poll_seconds)

    # ---- one job ----

    def run_once(self) -> Outcome | None:
        """Reaps if due, claims one job and processes it; None if there was none."""
        try:
            if time.monotonic() >= self._next_reap:
                self._reap()
            with self.engine.begin() as conn:
                job = claim(conn, self.settings.worker_id, self.settings.lease_seconds)
        except SQLAlchemyError as exc:
            log.warning("claim failed: %s", exc)
            return None
        self.liveness.touch()
        if job is None:
            return None
        log.info("claimed job %s (attempt %d)", job.id, job.attempt)
        outcome = self.process(job)
        log.info("job %s: %s%s", job.id, outcome.label,
                 f" ({outcome.detail})" if outcome.detail else "")
        return outcome

    def _reap(self) -> None:
        self._next_reap = time.monotonic() + self.settings.reaper_seconds
        with self.engine.begin() as conn:
            reaped = reap_expired(conn, backoff_base=self.settings.retry_backoff_seconds,
                                  backoff_max=self.settings.retry_backoff_max_seconds)
        for job_id in reaped.requeued:
            log.warning("requeued job %s: its lease expired", job_id)
        for job_id in reaped.failed:
            log.warning("failed job %s: its lease expired on its last attempt", job_id)

    def process(self, job: ClaimedJob) -> Outcome:
        beat = Heartbeat(self.engine, job, self.settings, self.liveness)
        beat.start()
        try:
            return self._process(job, beat)
        finally:
            beat.stop()

    def _process(self, job: ClaimedJob, beat: Heartbeat) -> Outcome:
        try:
            request = parse_request(job.request)
            with self.engine.connect() as conn:
                frame = load_frame(conn, job.price_series_id, request.start, request.end)
        except (ValidationError, ValueError) as exc:
            return self._fail(job, f"invalid request: {exc}", retry=False)
        except SQLAlchemyError as exc:
            return self._fail(job, f"database error loading prices: {exc}", retry=True)

        if self.compute.pid is None:  # killed at a shutdown that was then not completed
            self.compute.start()
        self.compute.submit(Task(frame=frame, strategy=request.strategy,
                                 params=request.params.model_dump(), cost_bps=request.cost_bps,
                                 n_boot=request.n_boot, seed=request.seed))
        timeout_at = time.monotonic() + self.settings.job_timeout_seconds
        while True:
            try:
                outcome = self.compute.poll(POLL_INTERVAL_SECONDS)
            except ComputeDied as exc:
                self.compute.restart()
                return self._fail(job, str(exc), retry=True)
            if outcome is not None:
                break
            if beat.lost.is_set():
                self.compute.restart()
                return Outcome("lease_lost", "another worker owns the job now")
            if time.monotonic() > timeout_at:
                self.compute.restart()
                return self._fail(
                    job, f"timed out after {self.settings.job_timeout_seconds:.0f}s",
                    retry=False)
            if self._release_now or (self._stopping
                                     and time.monotonic() > self._grace_deadline):
                # Kill, not restart: the worker is shutting down, and a
                # restart (re-importing pandas and scikit-learn, seconds)
                # would hold the job back from other workers and eat into
                # Docker's stop timeout. A later job starts a new child.
                self.compute.kill()
                return self._release(job)

        kind, value = outcome
        if kind == "invalid":
            return self._fail(job, f"invalid input: {value}", retry=False)
        if kind == "error":
            return self._fail(job, f"backtest raised:\n{value}", retry=False)
        return self._complete(job, value)

    # ---- writing outcomes ----

    def _complete(self, job: ClaimedJob, payload: dict[str, Any]) -> Outcome:
        result = JobResult(
            qbt_version=self.qbt_version, numpy_version=payload["numpy_version"],
            bootstrap_impl=payload["bootstrap_impl"], duration_ms=payload["duration_ms"],
            dates=payload["dates"], equity=payload["equity"], position=payload["position"],
            metrics=payload["metrics"],
        )
        try:
            with self.engine.begin() as conn:
                written = complete(conn, job.id, job.lease_token, result)
        except SQLAlchemyError as exc:
            return Outcome("db_error", f"could not store the result: {exc}")
        if not written:
            return Outcome("lease_lost", "result discarded: the lease was lost")
        return Outcome("succeeded", f"{payload['duration_ms']} ms")

    def _fail(self, job: ClaimedJob, error: str, *, retry: bool) -> Outcome:
        try:
            with self.engine.begin() as conn:
                status = fail(conn, job.id, job.lease_token, error, retry=retry,
                              backoff_base=self.settings.retry_backoff_seconds,
                              backoff_max=self.settings.retry_backoff_max_seconds)
        except SQLAlchemyError as exc:
            return Outcome("db_error", f"could not record failure ({error}): {exc}")
        if status is None:
            return Outcome("lease_lost", error)
        return Outcome("failed" if status == "failed" else "retrying", error)

    def _release(self, job: ClaimedJob) -> Outcome:
        try:
            with self.engine.begin() as conn:
                released = release(conn, job.id, job.lease_token)
        except SQLAlchemyError as exc:
            return Outcome("db_error", f"could not release: {exc}")
        return Outcome("released" if released else "lease_lost", "shutting down")


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    log.info("worker %s starting", settings.worker_id)
    worker = Worker(settings)

    def on_signal(signum: int, frame: FrameType | None) -> None:
        worker.request_stop()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        worker.start()
    except StartupError as exc:
        log.error("%s", exc)
        worker.engine.dispose()
        return 2
    try:
        worker.run()
    finally:
        worker.stop()
    log.info("worker %s stopped", settings.worker_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
