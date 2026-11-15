"""
The queue's NOTIFY: sent by every transition into queued (enqueue, retry,
release, reaper requeue) and only when its transaction commits; not by
transitions that leave nothing to claim. And next_due_seconds().
"""

import time
import uuid

import psycopg
import pytest
from sqlalchemy import make_url, text, update

from qbt_service.db import jobs
from qbt_service.queue import (
    QUEUE_CHANNEL,
    claim,
    complete,
    enqueue,
    fail,
    next_due_seconds,
    reap_expired,
    release,
)

from .factories import add_series, add_strategy, sha
from .test_queue import result

pytestmark = pytest.mark.db


@pytest.fixture
def engine(committed_engine):
    with committed_engine.begin() as conn:
        refs = {"strategy_id": add_strategy(conn), "price_series_id": add_series(conn)}
    committed_engine.refs = refs
    return committed_engine


@pytest.fixture
def listener(engine):
    url = make_url(engine.url).set(drivername="postgresql")
    conn = psycopg.connect(url.render_as_string(hide_password=False), autocommit=True)
    conn.execute(f"LISTEN {QUEUE_CHANNEL}")
    yield conn
    conn.close()


def received(listener, wait: float = 0.3) -> int:
    return sum(1 for _ in listener.notifies(timeout=wait))


def add(conn, refs, key=None):
    return enqueue(conn, **refs, request={"x": str(uuid.uuid4())}, request_hash=sha("r"),
                   idempotency_key=key).job_id


def test_enqueue_notifies_when_it_commits(engine, listener):
    with engine.begin() as conn:
        add(conn, engine.refs)
        assert received(listener) == 0  # not yet: the transaction is open
    assert received(listener) == 1


def test_rolled_back_enqueue_does_not_notify(engine, listener):
    with engine.connect() as conn:
        transaction = conn.begin()
        add(conn, engine.refs)
        transaction.rollback()
    assert received(listener) == 0


def test_idempotent_replay_does_not_notify(engine, listener):
    with engine.begin() as conn:
        add(conn, engine.refs, key="k")
    assert received(listener) == 1
    with engine.begin() as conn:
        enqueue(conn, **engine.refs, request={}, request_hash=sha("r"), idempotency_key="k")
    assert received(listener) == 0


def claimed(engine, lease=30.0):
    with engine.begin() as conn:
        add(conn, engine.refs)
    with engine.begin() as conn:
        job = claim(conn, "w", lease)
    return job


@pytest.mark.parametrize(("action", "notifies"), [
    (lambda c, j: fail(c, j.id, j.lease_token, "transient", retry=True), True),
    (lambda c, j: fail(c, j.id, j.lease_token, "permanent", retry=False), False),
    (lambda c, j: release(c, j.id, j.lease_token), True),
    (lambda c, j: complete(c, j.id, j.lease_token, result()), False),
    (lambda c, j: c.execute(update(jobs).where(jobs.c.id == j.id)
                            .values(lease_expires_at=text("now() - interval '1 s'")))
     and reap_expired(c), True),
], ids=["retry", "permanent-failure", "release", "complete", "reaper-requeue"])
def test_transitions_into_queued_notify(engine, listener, action, notifies):
    job = claimed(engine)
    received(listener)  # the enqueue's notification
    with engine.begin() as conn:
        action(conn, job)
    assert received(listener) == (1 if notifies else 0)


def test_reaper_failing_a_job_does_not_notify(engine, listener):
    with engine.begin() as conn:
        job_id = enqueue(conn, **engine.refs, request={}, request_hash=sha("r"),
                         max_attempts=1).job_id
    with engine.begin() as conn:
        claim(conn, "w", 30)
        conn.execute(update(jobs).where(jobs.c.id == job_id)
                     .values(lease_expires_at=text("now() - interval '1 s'")))
    received(listener)
    with engine.begin() as conn:
        assert reap_expired(conn).failed == [job_id]
    assert received(listener) == 0


def test_next_due_seconds(engine):
    with engine.connect() as conn:
        assert next_due_seconds(conn) is None
    with engine.begin() as conn:
        job_id = add(conn, engine.refs)
    with engine.connect() as conn:
        assert next_due_seconds(conn) <= 0
    with engine.begin() as conn:
        conn.execute(update(jobs).where(jobs.c.id == job_id)
                     .values(run_after=text("now() + interval '20 s'")))
    with engine.connect() as conn:
        assert 19 < next_due_seconds(conn) <= 20


def test_one_transaction_notifies_once(engine, listener):
    with engine.begin() as conn:
        for _ in range(5):
            add(conn, engine.refs)
    time.sleep(0.1)
    assert received(listener) == 1  # Postgres folds identical notifications
