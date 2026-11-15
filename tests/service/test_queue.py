"""
The job queue against a real Postgres.

Single-connection tests walk each transition (enqueue, claim, heartbeat,
complete, fail, release, reap) and the fencing rules inside one
rolled-back transaction. Concurrency tests use their own committed
database and real threads: N workers claiming M jobs get each job exactly
once, concurrent enqueues with one Idempotency-Key make one job,
concurrent reapers requeue each job once, and a stale worker's completion
racing the reaper never lets both win.

Lease expiry is simulated by moving lease_expires_at into the past, never
by sleeping.
"""

import math
import threading
import time
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import func, select, text, update

from qbt_service.db import jobs, metrics, results
from qbt_service.queue import (
    IdempotencyConflict,
    JobResult,
    claim,
    complete,
    enqueue,
    fail,
    heartbeat,
    reap_expired,
    release,
)

from .factories import add_series, add_strategy, sha

pytestmark = pytest.mark.db

LEASE = 30.0


def result(**overrides) -> JobResult:
    values = {
        "qbt_version": "0.1.0", "numpy_version": "2.5.4", "bootstrap_impl": "cpp",
        "duration_ms": 42, "dates": [date(2020, 1, 2), date(2020, 1, 3)],
        "equity": [1.0, 1.01], "position": [0.0, 1.0],
        "metrics": {"sharpe_ratio": 0.61, "sortino_ratio": math.nan},
    } | overrides
    return JobResult(**values)


def targets(conn) -> dict:
    return {"strategy_id": add_strategy(conn, name=f"s-{uuid.uuid4()}"),
            "price_series_id": add_series(conn)}


def new_job(conn, key=None, body="request", max_attempts=3, refs=None):
    refs = refs or targets(conn)
    return enqueue(conn, **refs, request={"body": body}, request_hash=sha(body),
                   idempotency_key=key, max_attempts=max_attempts).job_id


def job(conn, job_id):
    return conn.execute(select(jobs).where(jobs.c.id == job_id)).mappings().one()


def expire(conn, job_id):
    conn.execute(update(jobs).where(jobs.c.id == job_id)
                 .values(lease_expires_at=func.now() - timedelta(seconds=1)))


def make_due(conn, job_id):
    conn.execute(update(jobs).where(jobs.c.id == job_id).values(run_after=func.now()))


def seconds_until_due(conn, job_id) -> float:
    return conn.execute(
        select(func.extract("epoch", jobs.c.run_after - func.now())).where(jobs.c.id == job_id)
    ).scalar_one()


def count(conn, table, job_id) -> int:
    return conn.execute(
        select(func.count()).select_from(table).where(table.c.job_id == job_id)
    ).scalar_one()


@pytest.fixture
def queue_conn(conn):
    """`conn`, with the queue emptied first: the session database is shared,
    so claim() could otherwise pick up a job another test committed."""
    conn.execute(text("DELETE FROM jobs"))
    return conn


# ---- enqueue and idempotency ----

def test_enqueue_without_a_key_always_creates(queue_conn):
    refs = targets(queue_conn)
    a = enqueue(queue_conn, **refs, request={}, request_hash=sha("x"))
    b = enqueue(queue_conn, **refs, request={}, request_hash=sha("x"))
    assert a.created and b.created and a.job_id != b.job_id
    row = job(queue_conn, a.job_id)
    assert (row["status"], row["attempts"], row["max_attempts"]) == ("queued", 0, 3)


def test_same_key_and_body_returns_the_same_job(queue_conn):
    first = enqueue(queue_conn, **targets(queue_conn), request={"a": 1}, request_hash=sha("a"),
                    idempotency_key="k1")
    again = enqueue(queue_conn, **targets(queue_conn), request={"a": 1}, request_hash=sha("a"),
                    idempotency_key="k1")
    assert first.created and not again.created
    assert again.job_id == first.job_id


def test_same_key_with_a_different_body_conflicts(queue_conn):
    first = new_job(queue_conn, key="k2", body="one")
    with pytest.raises(IdempotencyConflict) as exc:
        new_job(queue_conn, key="k2", body="two")
    assert exc.value.job_id == first


def test_key_still_maps_to_its_job_after_it_finishes(queue_conn):
    job_id = new_job(queue_conn, key="k3")
    claimed = claim(queue_conn, "w", LEASE)
    complete(queue_conn, claimed.id, claimed.lease_token, result())
    assert new_job(queue_conn, key="k3") == job_id


# ---- claim ----

def test_claim_takes_the_oldest_due_job_and_leases_it(queue_conn):
    refs = targets(queue_conn)
    first = new_job(queue_conn, refs=refs)
    second = new_job(queue_conn, refs=refs)
    # Same transaction, same now(): make creation order explicit.
    queue_conn.execute(update(jobs).where(jobs.c.id == first)
                       .values(created_at=func.now() - timedelta(minutes=1)))

    claimed = claim(queue_conn, "worker-a", LEASE)
    assert claimed.id == first
    assert claimed.attempt == 1
    assert claimed.request == {"body": "request"}
    assert claimed.strategy_id == refs["strategy_id"]
    assert claimed.price_series_id == refs["price_series_id"]
    row = job(queue_conn, first)
    assert row["status"] == "running" and row["lease_owner"] == "worker-a"
    assert row["lease_token"] == claimed.lease_token
    assert row["started_at"] is not None and row["heartbeat_at"] is not None
    assert row["lease_expires_at"] - row["heartbeat_at"] == timedelta(seconds=LEASE)

    assert claim(queue_conn, "worker-b", LEASE).id == second
    assert claim(queue_conn, "worker-c", LEASE) is None


def test_claim_waits_for_run_after(queue_conn):
    job_id = new_job(queue_conn)
    queue_conn.execute(update(jobs).where(jobs.c.id == job_id)
                       .values(run_after=func.now() + timedelta(seconds=60)))
    assert claim(queue_conn, "w", LEASE) is None
    make_due(queue_conn, job_id)
    assert claim(queue_conn, "w", LEASE).id == job_id


def test_every_claim_gets_a_new_token(queue_conn):
    new_job(queue_conn)
    first = claim(queue_conn, "w", LEASE)
    release(queue_conn, first.id, first.lease_token)
    second = claim(queue_conn, "w", LEASE)
    assert second.id == first.id and second.lease_token != first.lease_token


# ---- heartbeat ----

def test_heartbeat_extends_only_the_holders_lease(queue_conn):
    new_job(queue_conn)
    claimed = claim(queue_conn, "w", LEASE)
    queue_conn.execute(update(jobs).where(jobs.c.id == claimed.id)
                       .values(lease_expires_at=func.now()))
    assert heartbeat(queue_conn, claimed.id, claimed.lease_token, 90.0)
    row = job(queue_conn, claimed.id)
    assert row["lease_expires_at"] - row["heartbeat_at"] == timedelta(seconds=90)
    assert not heartbeat(queue_conn, claimed.id, uuid.uuid4(), 90.0)


# ---- complete ----

def test_complete_stores_the_result(queue_conn):
    new_job(queue_conn)
    claimed = claim(queue_conn, "w", LEASE)
    assert complete(queue_conn, claimed.id, claimed.lease_token, result())

    row = job(queue_conn, claimed.id)
    assert row["status"] == "succeeded" and row["finished_at"] is not None
    assert row["lease_token"] is None and row["lease_owner"] is None
    stored = queue_conn.execute(select(results).where(results.c.job_id == claimed.id)
                                ).mappings().one()
    assert (stored["attempt"], stored["bootstrap_impl"], stored["duration_ms"]) == (1, "cpp", 42)
    assert stored["equity"] == [1.0, 1.01] and stored["dates"] == [date(2020, 1, 2),
                                                                     date(2020, 1, 3)]
    values = dict(queue_conn.execute(
        select(metrics.c.name, metrics.c.value).where(metrics.c.job_id == claimed.id)
    ).all())
    assert values["sharpe_ratio"] == 0.61 and math.isnan(values["sortino_ratio"])


def test_complete_twice_writes_once(queue_conn):
    new_job(queue_conn)
    claimed = claim(queue_conn, "w", LEASE)
    assert complete(queue_conn, claimed.id, claimed.lease_token, result())
    assert not complete(queue_conn, claimed.id, claimed.lease_token, result())
    assert count(queue_conn, results, claimed.id) == 1


def test_complete_without_metrics(queue_conn):
    new_job(queue_conn)
    claimed = claim(queue_conn, "w", LEASE)
    assert complete(queue_conn, claimed.id, claimed.lease_token,
                    result(metrics={}, bootstrap_impl=None))
    assert count(queue_conn, metrics, claimed.id) == 0


# ---- fencing ----

def test_stale_worker_cannot_touch_a_reclaimed_job(queue_conn):
    new_job(queue_conn)
    stale = claim(queue_conn, "worker-a", LEASE)
    expire(queue_conn, stale.id)
    assert reap_expired(queue_conn, backoff_base=0).requeued == [stale.id]
    make_due(queue_conn, stale.id)
    fresh = claim(queue_conn, "worker-b", LEASE)
    assert fresh.id == stale.id and fresh.attempt == 2

    # worker-a wakes up: every write it tries is rejected.
    assert not heartbeat(queue_conn, stale.id, stale.lease_token, LEASE)
    assert not complete(queue_conn, stale.id, stale.lease_token, result())
    assert fail(queue_conn, stale.id, stale.lease_token, "boom", retry=True) is None
    assert not release(queue_conn, stale.id, stale.lease_token)
    assert count(queue_conn, results, stale.id) == 0
    assert job(queue_conn, stale.id)["lease_owner"] == "worker-b"

    assert complete(queue_conn, fresh.id, fresh.lease_token, result())
    stored = queue_conn.execute(select(results.c.attempt).where(results.c.job_id == fresh.id))
    assert stored.scalar_one() == 2


def test_stale_worker_cannot_touch_a_requeued_job(queue_conn):
    new_job(queue_conn)
    stale = claim(queue_conn, "worker-a", LEASE)
    expire(queue_conn, stale.id)
    reap_expired(queue_conn)
    # Requeued but not yet claimed again: still fenced off.
    assert not heartbeat(queue_conn, stale.id, stale.lease_token, LEASE)
    assert not complete(queue_conn, stale.id, stale.lease_token, result())
    assert job(queue_conn, stale.id)["status"] == "queued"


# ---- fail ----

def test_permanent_failure_fails_at_once(queue_conn):
    new_job(queue_conn)
    claimed = claim(queue_conn, "w", LEASE)
    assert fail(queue_conn, claimed.id, claimed.lease_token, "unknown param", retry=False) == (
        "failed")
    row = job(queue_conn, claimed.id)
    assert (row["status"], row["attempts"], row["last_error"]) == ("failed", 1, "unknown param")
    assert row["finished_at"] is not None and row["lease_token"] is None


def test_transient_failure_retries_with_backoff_then_fails(queue_conn):
    job_id = new_job(queue_conn, max_attempts=3)
    for attempt, delay in [(1, 5.0), (2, 10.0)]:
        claimed = claim(queue_conn, "w", LEASE)
        assert claimed.attempt == attempt
        assert fail(queue_conn, job_id, claimed.lease_token, "db went away", retry=True) == (
            "queued")
        assert seconds_until_due(queue_conn, job_id) == pytest.approx(delay)
        assert job(queue_conn, job_id)["finished_at"] is None
        make_due(queue_conn, job_id)
    claimed = claim(queue_conn, "w", LEASE)
    assert fail(queue_conn, job_id, claimed.lease_token, "db went away", retry=True) == "failed"
    assert job(queue_conn, job_id)["attempts"] == 3
    assert claim(queue_conn, "w", LEASE) is None


def test_backoff_is_capped(queue_conn):
    job_id = new_job(queue_conn, max_attempts=20)
    queue_conn.execute(update(jobs).where(jobs.c.id == job_id).values(attempts=15))
    claimed = claim(queue_conn, "w", LEASE)
    fail(queue_conn, job_id, claimed.lease_token, "x", retry=True, backoff_max=300)
    assert seconds_until_due(queue_conn, job_id) == pytest.approx(300.0)


def test_long_errors_are_truncated(queue_conn):
    new_job(queue_conn)
    claimed = claim(queue_conn, "w", LEASE)
    fail(queue_conn, claimed.id, claimed.lease_token, "x" * 10_000, retry=False)
    error = job(queue_conn, claimed.id)["last_error"]
    assert len(error) == 4000 and error.endswith("...")


# ---- release ----

def test_release_gives_the_attempt_back(queue_conn):
    new_job(queue_conn)
    claimed = claim(queue_conn, "w", LEASE)
    assert release(queue_conn, claimed.id, claimed.lease_token)
    row = job(queue_conn, claimed.id)
    assert (row["status"], row["attempts"], row["lease_token"]) == ("queued", 0, None)
    assert claim(queue_conn, "w2", LEASE).attempt == 1


# ---- reaper ----

def test_reaper_requeues_then_fails_at_max_attempts(queue_conn):
    job_id = new_job(queue_conn, max_attempts=2)

    claim(queue_conn, "worker-a", LEASE)
    expire(queue_conn, job_id)
    assert reap_expired(queue_conn).requeued == [job_id]
    row = job(queue_conn, job_id)
    assert (row["status"], row["attempts"], row["lease_token"]) == ("queued", 1, None)
    assert row["last_error"] == "lease expired (worker worker-a, attempt 1)"
    assert seconds_until_due(queue_conn, job_id) == pytest.approx(5.0)
    assert claim(queue_conn, "w", LEASE) is None  # backing off

    make_due(queue_conn, job_id)
    claim(queue_conn, "worker-b", LEASE)
    expire(queue_conn, job_id)
    assert reap_expired(queue_conn).failed == [job_id]
    row = job(queue_conn, job_id)
    assert (row["status"], row["attempts"]) == ("failed", 2)
    assert row["finished_at"] is not None
    assert row["last_error"] == "lease expired (worker worker-b, attempt 2)"
    assert claim(queue_conn, "w", LEASE) is None


def test_reaper_leaves_live_and_idle_jobs_alone(queue_conn):
    queued = new_job(queue_conn)
    live = new_job(queue_conn)
    queue_conn.execute(update(jobs).where(jobs.c.id == queued)
                       .values(created_at=func.now() + timedelta(seconds=1)))
    claimed = claim(queue_conn, "w", LEASE)
    assert claimed.id == live
    reaped = reap_expired(queue_conn)
    assert reaped.requeued == [] and reaped.failed == []
    assert job(queue_conn, live)["status"] == "running"
    assert job(queue_conn, queued)["status"] == "queued"


# ---- concurrency, on a committed database ----

def run_threads(n, target):
    """Runs target(i) on n threads started together; re-raises the first error."""
    barrier = threading.Barrier(n)
    errors = []

    def wrapped(i):
        try:
            barrier.wait()
            target(i)
        except Exception as exc:  # reported after join
            errors.append(exc)

    threads = [threading.Thread(target=wrapped, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


@pytest.fixture
def engine(committed_engine):
    return committed_engine


def enqueue_many(engine, m, max_attempts=3):
    with engine.begin() as c:
        refs = targets(c)
        return [new_job(c, body=f"job-{i}", refs=refs, max_attempts=max_attempts)
                for i in range(m)]


def test_n_workers_claim_m_jobs_exactly_once(engine):
    n_workers, m_jobs = 8, 60
    job_ids = enqueue_many(engine, m_jobs)
    claims: list[tuple[uuid.UUID, int]] = []
    lock = threading.Lock()

    def worker(i):
        while True:
            with engine.begin() as c:
                claimed = claim(c, f"worker-{i}", LEASE)
                if claimed is None:
                    return
                # Hold the row lock a moment, so other claimers must skip it.
                c.execute(text("SELECT pg_sleep(0.002)"))
            with lock:
                claims.append((claimed.id, i))

    run_threads(n_workers, worker)

    claimed_ids = [job_id for job_id, _ in claims]
    assert len(claimed_ids) == m_jobs
    assert set(claimed_ids) == set(job_ids)
    with engine.connect() as c:
        states = c.execute(select(jobs.c.status, jobs.c.attempts, func.count())
                           .group_by(jobs.c.status, jobs.c.attempts)).all()
    assert states == [("running", 1, m_jobs)]


def test_concurrent_enqueues_with_one_key_make_one_job(engine):
    with engine.begin() as c:
        refs = targets(c)
    outcomes = []
    lock = threading.Lock()

    def submit(_):
        with engine.begin() as c:
            enqueued = enqueue(c, **refs, request={"a": 1}, request_hash=sha("same"),
                               idempotency_key="shared-key")
        with lock:
            outcomes.append(enqueued)

    run_threads(8, submit)

    assert sum(o.created for o in outcomes) == 1
    assert len({o.job_id for o in outcomes}) == 1
    with engine.connect() as c:
        assert c.execute(select(func.count()).select_from(jobs)).scalar_one() == 1


def test_concurrent_reapers_requeue_each_job_once(engine):
    job_ids = enqueue_many(engine, 20)
    with engine.begin() as c:
        while claim(c, "dead-worker", LEASE):
            pass
        c.execute(update(jobs).values(lease_expires_at=func.now() - timedelta(seconds=1)))
    reaped: list[list[uuid.UUID]] = []
    lock = threading.Lock()

    def reaper(_):
        with engine.begin() as c:
            ids = reap_expired(c).requeued
        with lock:
            reaped.append(ids)

    run_threads(4, reaper)

    all_ids = [job_id for ids in reaped for job_id in ids]
    assert sorted(all_ids) == sorted(job_ids)  # every job once, none twice
    with engine.connect() as c:
        assert c.execute(select(func.max(jobs.c.attempts))).scalar_one() == 1


def race_stale_completion_against_reaper(engine) -> bool:
    """One race between a stale worker completing and the reaper requeueing
    the same expired job. Checks exactly one won; returns whether the
    completion did."""
    with engine.begin() as c:
        c.execute(text("DELETE FROM jobs"))
        job_id = new_job(c)
    with engine.begin() as c:
        stale = claim(c, "worker-a", LEASE)
        expire(c, job_id)
    outcome = {}

    def racer(i):
        with engine.begin() as c:
            if i == 0:
                outcome["completed"] = complete(c, stale.id, stale.lease_token, result())
            else:
                outcome["requeued"] = reap_expired(c, backoff_base=0).requeued == [stale.id]

    run_threads(2, racer)

    assert outcome["completed"] != outcome["requeued"]
    with engine.connect() as c:
        state = (job(c, job_id)["status"], count(c, results, job_id))
    assert state == (("succeeded", 1) if outcome["completed"] else ("queued", 0))
    return outcome["completed"]


def test_stale_completion_racing_the_reaper_never_both_win(engine):
    # Asserts only the invariant, every round. Which side wins is left to
    # the scheduler (measured: the completion about 7 rounds in 8); each
    # ordering is forced explicitly in the two tests below.
    for _ in range(40):
        race_stale_completion_against_reaper(engine)


# ---- forced orderings: one transaction holds a row lock, another is made
# to queue behind it, then the first commits ----

def wait_until_blocked(engine, timeout: float = 10.0) -> None:
    """Returns once some backend is waiting for a lock (polls pg_locks)."""
    deadline = time.monotonic() + timeout
    with engine.connect() as c:
        while time.monotonic() < deadline:
            waiting = c.execute(text("SELECT count(*) FROM pg_locks WHERE NOT granted"))
            if waiting.scalar_one() > 0:
                return
            c.rollback()
            time.sleep(0.01)
    raise AssertionError("no backend started waiting for a lock")


def in_thread(fn):
    """Starts fn on a thread; returns a join() that gives its result or raises its error."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except Exception as exc:  # re-raised by join
            box["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()

    def join():
        thread.join(timeout=10)
        assert not thread.is_alive(), "thread did not finish"
        if "error" in box:
            raise box["error"]
        return box["value"]

    return join


def stale_claim(engine):
    """A job claimed by worker-a whose lease has already expired."""
    with engine.begin() as c:
        job_id = new_job(c)
    with engine.begin() as c:
        stale = claim(c, "worker-a", LEASE)
        expire(c, job_id)
    return stale


def test_completion_first_then_reaper_finds_nothing(engine):
    stale = stale_claim(engine)
    with engine.connect() as first:
        with first.begin():
            assert complete(first, stale.id, stale.lease_token, result())
            # The reaper's UPDATE now waits on this transaction's row lock...
            reaper = in_thread(lambda: _reap(engine))
            wait_until_blocked(engine)
        # ...and on commit, re-checks its WHERE clause against the new row.
        assert reaper() == ([], [])
    with engine.connect() as c:
        assert job(c, stale.id)["status"] == "succeeded"
        assert count(c, results, stale.id) == 1


def test_reaper_first_then_completion_is_rejected(engine):
    stale = stale_claim(engine)
    with engine.connect() as first:
        with first.begin():
            assert reap_expired(first, backoff_base=0).requeued == [stale.id]
            completion = in_thread(lambda: _complete(engine, stale))
            wait_until_blocked(engine)
        assert completion() is False
    with engine.connect() as c:
        assert job(c, stale.id)["status"] == "queued"
        assert count(c, results, stale.id) == 0


def test_claim_skips_a_row_another_claim_holds(engine):
    first_job, second_job = enqueue_many(engine, 2)
    with engine.connect() as holder, holder.begin():
        held = claim(holder, "worker-a", LEASE)
        assert held.id == first_job
        # lock_timeout: if the second claim waited on the held row
        # instead of skipping it, this would fail rather than hang.
        with engine.begin() as other:
            other.execute(text("SET LOCAL lock_timeout = '2s'"))
            assert claim(other, "worker-b", LEASE).id == second_job
            assert claim(other, "worker-c", LEASE) is None


def _reap(engine):
    with engine.begin() as c:
        reaped = reap_expired(c, backoff_base=0)
    return reaped.requeued, reaped.failed


def _complete(engine, claimed):
    with engine.begin() as c:
        return complete(c, claimed.id, claimed.lease_token, result())


def _hot_updates(conn) -> int:
    # Statistics are flushed asynchronously; force this backend's out first.
    conn.execute(text("SELECT pg_stat_force_next_flush()"))
    conn.commit()
    value = conn.execute(text(
        "SELECT n_tup_hot_upd FROM pg_stat_user_tables WHERE relname = 'jobs'"
    )).scalar_one()
    conn.commit()  # stats are cached per transaction
    return value


def test_heartbeats_are_hot_updates(engine):
    # The point of keeping heartbeat columns out of every index: each
    # heartbeat should be an in-page (HOT) update, not a new index entry.
    # 10 row versions fit in the page's free space (fillfactor 90), so this
    # holds without depending on Postgres pruning dead versions, which an
    # unrelated open snapshot could delay.
    (job_id,) = enqueue_many(engine, 1)
    with engine.begin() as c:
        claimed = claim(c, "w", LEASE)
    with engine.connect() as c:
        before = _hot_updates(c)
        for _ in range(10):
            assert heartbeat(c, job_id, claimed.lease_token, LEASE)
            c.commit()
        assert _hot_updates(c) - before == 10
