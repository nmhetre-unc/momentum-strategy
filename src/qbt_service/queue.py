"""
The job queue, in the `jobs` table.

Life of a job:

    enqueue ──> queued ──claim──> running ──complete──> succeeded
                  ^                  │
                  │                  ├─fail(retry=False)─────────────> failed
                  │                  ├─fail(retry=True), attempts left─> queued (after backoff)
                  ├──────────────────┤ reap: lease expired, attempts left
                  │                  ├─reap: lease expired, no attempts left ─> failed
                  └──────release─────┘ (graceful shutdown; the attempt is given back)

Every operation is one short statement or transaction; nothing holds a
row lock while a backtest runs. Instead a claimed job carries a lease: a
fresh random lease_token plus an expiry that the worker keeps extending
with heartbeat(). Every state change a worker makes is conditioned on its
lease_token ("fencing"), so a worker that stalled past its lease, had the
job requeued and claimed by someone else, and then woke up, can neither
extend the new lease nor write its result.

All times come from the database's now(), never a worker's clock. now()
is the start time of the current transaction, so callers keep these
transactions short (each function here is meant to be its own).

Execution is at least once: a job whose worker dies runs again. That is
safe because a backtest has no side effects outside complete(), and is
deterministic, so whichever attempt commits writes the same numbers.
"""

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import Connection, and_, case, cast, func, literal, select, update
from sqlalchemy.dialects.postgresql import insert

from qbt_service.db import job_status, jobs, metrics, results

# Retry backoff: base * 2**(attempt - 1) seconds, capped.
BACKOFF_BASE_SECONDS = 5.0
BACKOFF_MAX_SECONDS = 300.0
MAX_ERROR_LENGTH = 4000


class IdempotencyConflict(Exception):
    """The Idempotency-Key was already used for a different request."""

    def __init__(self, job_id: uuid.UUID):
        super().__init__(f"idempotency key already used for a different request (job {job_id})")
        self.job_id = job_id


@dataclass(frozen=True)
class Enqueued:
    job_id: uuid.UUID
    created: bool  # False: an earlier request with the same key and body made this job


@dataclass(frozen=True)
class ClaimedJob:
    id: uuid.UUID
    lease_token: uuid.UUID
    attempt: int                 # 1 for the first claim
    strategy_id: int
    price_series_id: int
    request: dict[str, Any]
    lease_expires_at: datetime


@dataclass(frozen=True)
class JobResult:
    qbt_version: str
    numpy_version: str
    bootstrap_impl: str | None
    duration_ms: int
    dates: list[date]
    equity: list[float]
    position: list[float]
    metrics: dict[str, float]


@dataclass(frozen=True)
class Reaped:
    requeued: list[uuid.UUID]
    failed: list[uuid.UUID]


def _backoff(attempts: Any, base: float, cap: float) -> Any:
    """SQL for now() + the backoff after `attempts` attempts."""
    seconds = func.least(cap, base * func.power(2, attempts - 1))
    return func.now() + func.make_interval(0, 0, 0, 0, 0, 0, seconds)


def _status(value: str) -> Any:
    """CAST(value AS job_status). A CASE over plain string parameters
    resolves to text, and Postgres won't assign text to an enum column."""
    return cast(literal(value), job_status)


def _clear_lease() -> dict[str, Any]:
    return {"lease_token": None, "lease_owner": None, "lease_expires_at": None}


def _fenced(job_id: uuid.UUID, lease_token: uuid.UUID) -> Any:
    return and_(jobs.c.id == job_id, jobs.c.lease_token == lease_token,
                jobs.c.status == "running")


def _truncate(error: str) -> str:
    return error if len(error) <= MAX_ERROR_LENGTH else error[:MAX_ERROR_LENGTH - 3] + "..."


def enqueue(conn: Connection, *, strategy_id: int, price_series_id: int,
            request: dict[str, Any], request_hash: bytes, idempotency_key: str | None = None,
            max_attempts: int = 3) -> Enqueued:
    """
    Adds a job. With an idempotency key, a repeat of the same request
    returns the job the first one created; the same key with a different
    request raises IdempotencyConflict. Two concurrent calls with one key
    are safe in READ COMMITTED (the default): the second insert waits on
    the unique index, finds the first job, and the next statement sees it.
    """
    values = {"strategy_id": strategy_id, "price_series_id": price_series_id,
              "request": request, "request_hash": request_hash,
              "idempotency_key": idempotency_key, "max_attempts": max_attempts}
    statement = insert(jobs).values(values)
    if idempotency_key is not None:
        statement = statement.on_conflict_do_nothing(
            index_elements=[jobs.c.idempotency_key],
            index_where=jobs.c.idempotency_key.isnot(None),
        )
    job_id = conn.execute(statement.returning(jobs.c.id)).scalar_one_or_none()
    if job_id is not None:
        return Enqueued(job_id=job_id, created=True)

    existing_id, existing_hash = conn.execute(
        select(jobs.c.id, jobs.c.request_hash).where(jobs.c.idempotency_key == idempotency_key)
    ).one()
    if bytes(existing_hash) != request_hash:
        raise IdempotencyConflict(existing_id)
    return Enqueued(job_id=existing_id, created=False)


def claim(conn: Connection, worker_id: str, lease_seconds: float) -> ClaimedJob | None:
    """
    Takes the oldest queued job that is due, or returns None. SKIP LOCKED
    makes concurrent claimers pass over a row another claimer is taking
    instead of waiting for it, so each job goes to exactly one of them.
    """
    next_job = (
        select(jobs.c.id)
        .where(jobs.c.status == "queued", jobs.c.run_after <= func.now())
        .order_by(jobs.c.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )
    row = conn.execute(
        update(jobs)
        .where(jobs.c.id == next_job)
        .values(status="running", attempts=jobs.c.attempts + 1,
                lease_token=func.gen_random_uuid(), lease_owner=worker_id,
                lease_expires_at=func.now() + timedelta(seconds=lease_seconds),
                heartbeat_at=func.now(),
                started_at=func.coalesce(jobs.c.started_at, func.now()))
        .returning(jobs.c.id, jobs.c.lease_token, jobs.c.attempts, jobs.c.strategy_id,
                   jobs.c.price_series_id, jobs.c.request, jobs.c.lease_expires_at)
    ).one_or_none()
    if row is None:
        return None
    return ClaimedJob(id=row.id, lease_token=row.lease_token, attempt=row.attempts,
                      strategy_id=row.strategy_id, price_series_id=row.price_series_id,
                      request=row.request, lease_expires_at=row.lease_expires_at)


def heartbeat(conn: Connection, job_id: uuid.UUID, lease_token: uuid.UUID,
              lease_seconds: float) -> bool:
    """
    Extends the lease. False means it was lost (requeued by the reaper,
    possibly already claimed by another worker): stop working on the job.
    Touches only lease_expires_at and heartbeat_at, which no index covers,
    so Postgres can update the row in place (a HOT update).
    """
    updated = conn.execute(
        update(jobs).where(_fenced(job_id, lease_token))
        .values(lease_expires_at=func.now() + timedelta(seconds=lease_seconds),
                heartbeat_at=func.now())
    )
    return updated.rowcount == 1


def complete(conn: Connection, job_id: uuid.UUID, lease_token: uuid.UUID,
             result: JobResult) -> bool:
    """
    Marks the job succeeded and stores its result, in the caller's
    transaction. False (nothing written) if the lease was lost. The UPDATE
    locks the job row, so a concurrent requeue or second completion waits
    for this transaction and then finds the token no longer matches.
    """
    attempt = conn.execute(
        update(jobs).where(_fenced(job_id, lease_token))
        .values(status="succeeded", finished_at=func.now(), last_error=None, **_clear_lease())
        .returning(jobs.c.attempts)
    ).scalar_one_or_none()
    if attempt is None:
        return False
    conn.execute(results.insert().values(
        job_id=job_id, attempt=attempt, qbt_version=result.qbt_version,
        numpy_version=result.numpy_version, bootstrap_impl=result.bootstrap_impl,
        duration_ms=result.duration_ms, dates=result.dates, equity=result.equity,
        position=result.position,
    ))
    if result.metrics:
        conn.execute(metrics.insert(), [
            {"job_id": job_id, "name": name, "value": value}
            for name, value in sorted(result.metrics.items())
        ])
    return True


def fail(conn: Connection, job_id: uuid.UUID, lease_token: uuid.UUID, error: str, *,
         retry: bool, backoff_base: float = BACKOFF_BASE_SECONDS,
         backoff_max: float = BACKOFF_MAX_SECONDS) -> str | None:
    """
    Records a failed attempt. retry=False is for failures that would
    repeat (bad input, a strategy error): the job fails now. retry=True is
    for transient ones (a lost connection): the job is requeued after a
    backoff while attempts remain, and fails once they run out. Returns
    the new status, or None if the lease was lost.
    """
    exhausted = jobs.c.attempts >= jobs.c.max_attempts
    give_up = literal(True) if not retry else exhausted
    return conn.execute(
        update(jobs).where(_fenced(job_id, lease_token))
        .values(
            status=case((give_up, _status("failed")), else_=_status("queued")),
            finished_at=case((give_up, func.now()), else_=None),
            run_after=case((give_up, jobs.c.run_after),
                           else_=_backoff(jobs.c.attempts, backoff_base, backoff_max)),
            last_error=_truncate(error),
            **_clear_lease(),
        )
        .returning(jobs.c.status)
    ).scalar_one_or_none()


def release(conn: Connection, job_id: uuid.UUID, lease_token: uuid.UUID) -> bool:
    """
    Hands a claimed job back unfinished (a worker shutting down cleanly).
    The attempt is given back, and the job can be claimed again at once.
    """
    updated = conn.execute(
        update(jobs).where(_fenced(job_id, lease_token))
        .values(status="queued", attempts=jobs.c.attempts - 1, run_after=func.now(),
                **_clear_lease())
    )
    return updated.rowcount == 1


def reap_expired(conn: Connection, *, backoff_base: float = BACKOFF_BASE_SECONDS,
                 backoff_max: float = BACKOFF_MAX_SECONDS) -> Reaped:
    """
    Requeues running jobs whose lease has expired (their worker died or
    stalled), after a backoff, or fails them if they have used all their
    attempts; a job that kills every worker that runs it stops there.
    Safe to run from every worker at once: a second reaper waits on the
    rows the first is updating, re-checks the WHERE clause, and skips them.
    """
    exhausted = jobs.c.attempts >= jobs.c.max_attempts
    rows = conn.execute(
        update(jobs)
        .where(jobs.c.status == "running", jobs.c.lease_expires_at < func.now())
        .values(
            status=case((exhausted, _status("failed")), else_=_status("queued")),
            finished_at=case((exhausted, func.now()), else_=None),
            run_after=case((exhausted, jobs.c.run_after),
                           else_=_backoff(jobs.c.attempts, backoff_base, backoff_max)),
            last_error=func.concat("lease expired (worker ", jobs.c.lease_owner,
                                   ", attempt ", jobs.c.attempts, ")"),
            **_clear_lease(),
        )
        .returning(jobs.c.id, jobs.c.status)
    ).all()
    return Reaped(requeued=[r.id for r in rows if r.status == "queued"],
                  failed=[r.id for r in rows if r.status == "failed"])
