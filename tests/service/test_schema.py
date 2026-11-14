"""
The schema's own guarantees, tested against a real Postgres: the CHECK
constraints that keep a job's status, lease and timestamps consistent,
idempotency-key uniqueness, immutable price snapshots, one result per
job, cascades, exact float round trips, and the index and storage
choices that keep heartbeats cheap (alembic check can't see any of these).
"""

import hashlib
import math
import uuid
from datetime import UTC, date, datetime

import numpy as np
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DataError, IntegrityError

from qbt.bootstrap import IMPLEMENTATIONS
from qbt_service.db import jobs, metrics, price_bars, price_series, results, strategies

pytestmark = pytest.mark.db

NOW = datetime(2026, 1, 2, tzinfo=UTC)


def sha(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def add_series(conn, **overrides) -> int:
    values = {
        "ticker": "SPY", "fetch_start": date(2020, 1, 1), "fetch_end": date(2021, 1, 1),
        "first_date": date(2020, 1, 1), "last_date": date(2020, 12, 31),
        "n_bars": 252, "content_sha256": sha(str(uuid.uuid4())),
    } | overrides
    return conn.execute(
        price_series.insert().values(values).returning(price_series.c.id)
    ).scalar_one()


def add_strategy(conn, name: str = "momentum", **overrides) -> int:
    values = {
        "name": name, "schema_hash": sha(str(uuid.uuid4())), "param_schema": {"type": "object"},
    } | overrides
    return conn.execute(strategies.insert().values(values).returning(strategies.c.id)).scalar_one()


def add_job(conn, **overrides) -> uuid.UUID:
    values = {
        "strategy_id": add_strategy(conn, name=f"s-{uuid.uuid4()}"),
        "price_series_id": add_series(conn),
        "request": {"strategy": "momentum", "params": {}},
        "request_hash": sha("request"),
    } | overrides
    return conn.execute(jobs.insert().values(values).returning(jobs.c.id)).scalar_one()


def add_result(conn, job_id, n: int = 3, **overrides) -> None:
    values = {
        "job_id": job_id, "attempt": 1, "qbt_version": "0.1.0", "numpy_version": "2.5.4",
        "bootstrap_impl": "cpp", "duration_ms": 12,
        "dates": [date(2020, 1, d + 1) for d in range(n)],
        "equity": [1.0] * n, "position": [0.0] * n,
    } | overrides
    conn.execute(results.insert().values(values))


def rejected(conn, statement_fn, match: str, exc: type[Exception] = IntegrityError):
    """
    Asserts the statement fails. For an IntegrityError, `match` is a
    constraint name, matched with Postgres's quotes so a misnamed
    constraint can't pass; otherwise it's a regex. Runs inside a savepoint
    so the test can continue.
    """
    pattern = f'"{match}"' if exc is IntegrityError else match
    with pytest.raises(exc, match=pattern), conn.begin_nested():
        statement_fn()


RUNNING = {"status": "running", "lease_token": uuid.uuid4(), "lease_expires_at": NOW}


# ---- jobs: defaults and state consistency ----

def test_job_defaults(conn):
    job_id = add_job(conn)
    row = conn.execute(select(jobs).where(jobs.c.id == job_id)).mappings().one()
    assert isinstance(row["id"], uuid.UUID)
    assert row["status"] == "queued"
    assert (row["attempts"], row["max_attempts"]) == (0, 3)
    assert row["created_at"] is not None and row["run_after"] is not None
    assert row["lease_token"] is None and row["finished_at"] is None


def test_running_job_needs_a_lease(conn):
    rejected(conn, lambda: add_job(conn, status="running"), "ck_jobs_running_has_lease")
    rejected(conn, lambda: add_job(conn, status="running", lease_token=uuid.uuid4()),
             "ck_jobs_running_has_lease")
    add_job(conn, attempts=1, **RUNNING)


def test_only_running_jobs_hold_a_lease(conn):
    for status in ("queued", "succeeded", "failed"):
        extra = {"finished_at": NOW} if status != "queued" else {}
        rejected(conn, lambda s=status, e=extra: add_job(
            conn, status=s, lease_token=uuid.uuid4(), lease_expires_at=NOW, **e,
        ), "ck_jobs_running_has_lease")


def test_finished_at_iff_terminal(conn):
    for status in ("succeeded", "failed"):
        rejected(conn, lambda s=status: add_job(conn, status=s), "ck_jobs_finished_iff_terminal")
        add_job(conn, status=status, finished_at=NOW)
    rejected(conn, lambda: add_job(conn, finished_at=NOW), "ck_jobs_finished_iff_terminal")


def test_attempts_bounded_by_max_attempts(conn):
    rejected(conn, lambda: add_job(conn, attempts=4), "ck_jobs_attempts_in_range")
    rejected(conn, lambda: add_job(conn, attempts=-1), "ck_jobs_attempts_in_range")
    rejected(conn, lambda: add_job(conn, max_attempts=0), "ck_jobs_max_attempts_positive")
    add_job(conn, attempts=3)


def test_unknown_status_is_rejected(conn):
    rejected(conn, lambda: add_job(conn, status="cancelled"), "enum job_status",
             exc=DataError)


def test_hash_lengths_are_sha256(conn):
    rejected(conn, lambda: add_job(conn, request_hash=b"short"), "ck_jobs_request_hash_length")
    rejected(conn, lambda: add_series(conn, content_sha256=b"short"),
             "ck_price_series_sha256_length")


# ---- idempotency keys ----

def test_idempotency_key_is_unique_when_present(conn):
    add_job(conn, idempotency_key="abc")
    rejected(conn, lambda: add_job(conn, idempotency_key="abc"), "jobs_idempotency_key")
    add_job(conn)
    add_job(conn)  # any number of jobs without a key


def test_idempotency_key_length(conn):
    rejected(conn, lambda: add_job(conn, idempotency_key=""), "ck_jobs_idempotency_key_length")
    rejected(conn, lambda: add_job(conn, idempotency_key="k" * 256),
             "ck_jobs_idempotency_key_length")
    add_job(conn, idempotency_key="k" * 255)


# ---- price snapshots ----

def test_price_series_checks(conn):
    rejected(conn, lambda: add_series(conn, n_bars=0), "ck_price_series_n_bars_positive")
    rejected(conn, lambda: add_series(conn, first_date=date(2020, 12, 31),
                                      last_date=date(2020, 6, 1)),
             "ck_price_series_date_order")
    # fetch_end is exclusive: a bar on it is outside the range. (The
    # fetch_range_order check is implied by this one plus date_order, so no
    # row can violate it alone; it stays as documentation.)
    rejected(conn, lambda: add_series(conn, last_date=date(2021, 1, 1)),
             "ck_price_series_bars_within_fetch_range")
    rejected(conn, lambda: add_series(conn, first_date=date(2019, 12, 31)),
             "ck_price_series_bars_within_fetch_range")


def test_identical_snapshot_is_not_stored_twice(conn):
    add_series(conn, content_sha256=sha("same"))
    rejected(conn, lambda: add_series(conn, content_sha256=sha("same")),
             "uq_price_series_ticker_source_content_sha256")
    add_series(conn, ticker="QQQ", content_sha256=sha("same"))


def test_series_in_use_cannot_be_deleted(conn):
    def delete(series_id):
        conn.execute(price_series.delete().where(price_series.c.id == series_id))

    with_bars = add_series(conn)
    conn.execute(price_bars.insert().values(series_id=with_bars, date=date(2020, 1, 2), close=1.0))
    rejected(conn, lambda: delete(with_bars), "fk_price_bars_series_id_price_series")

    pinned = add_series(conn)
    add_job(conn, price_series_id=pinned)
    rejected(conn, lambda: delete(pinned), "fk_jobs_price_series_id_price_series")


def test_prices_round_trip_bit_exactly(conn):
    series_id = add_series(conn)
    values = [0.1 + 0.2, np.nextafter(1.0, 2.0), 1 / 3, 123456.789012345678, 5e-324]
    for day, value in enumerate(values, start=1):
        conn.execute(price_bars.insert().values(
            series_id=series_id, date=date(2020, 1, day), close=float(value)))
    stored = conn.execute(
        select(price_bars.c.close).where(price_bars.c.series_id == series_id)
        .order_by(price_bars.c.date)
    ).scalars().all()
    assert [v.hex() for v in stored] == [float(v).hex() for v in values]


# ---- strategy catalog ----

def test_one_current_schema_per_strategy(conn):
    add_strategy(conn, name="sma_crossover")
    rejected(conn, lambda: add_strategy(conn, name="sma_crossover"), "strategies_one_current")
    add_strategy(conn, name="sma_crossover", is_current=False)
    add_strategy(conn, name="sma_crossover", is_current=False)


# ---- results and metrics ----

def test_one_result_per_job(conn):
    job_id = add_job(conn)
    add_result(conn, job_id)
    rejected(conn, lambda: add_result(conn, job_id), "pk_results")


def test_result_series_lengths_must_match(conn):
    job_id = add_job(conn)
    rejected(conn, lambda: add_result(conn, job_id, equity=[1.0, 1.1]),
             "ck_results_series_lengths_match")
    rejected(conn, lambda: add_result(conn, job_id, position=[0.0]),
             "ck_results_series_lengths_match")


def test_bootstrap_impl_values(conn):
    for impl in (*IMPLEMENTATIONS, None):
        add_result(conn, add_job(conn), bootstrap_impl=impl)
    rejected(conn, lambda: add_result(conn, add_job(conn), bootstrap_impl="auto"),
             "ck_results_bootstrap_impl_known")


def test_deleting_a_job_removes_its_result_and_metrics(conn):
    job_id = add_job(conn)
    add_result(conn, job_id)
    conn.execute(metrics.insert().values(job_id=job_id, name="sharpe_ratio", value=0.6))
    conn.execute(jobs.delete().where(jobs.c.id == job_id))
    assert conn.execute(select(func.count()).select_from(results)
                        .where(results.c.job_id == job_id)).scalar_one() == 0
    assert conn.execute(select(func.count()).select_from(metrics)
                        .where(metrics.c.job_id == job_id)).scalar_one() == 0


def test_metrics_store_nan_and_null(conn):
    job_id = add_job(conn)
    add_result(conn, job_id)
    conn.execute(metrics.insert(), [
        {"job_id": job_id, "name": "sortino_ratio", "value": float("nan")},
        {"job_id": job_id, "name": "sharpe_ratio", "value": None},
    ])
    stored = dict(conn.execute(
        select(metrics.c.name, metrics.c.value).where(metrics.c.job_id == job_id)
    ).tuples().all())
    assert math.isnan(stored["sortino_ratio"])
    assert stored["sharpe_ratio"] is None


def test_metric_needs_a_result(conn):
    rejected(conn, lambda: conn.execute(
        metrics.insert().values(job_id=add_job(conn), name="sharpe_ratio", value=1.0)
    ), "fk_metrics_job_id_results")


# ---- queue storage design ----

def _index_defs(conn, table: str) -> dict[str, str]:
    return dict(conn.execute(
        text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = :t"), {"t": table}
    ).tuples().all())


def test_jobs_indexes_are_exactly_the_planned_ones(conn):
    defs = _index_defs(conn, "jobs")
    assert set(defs) == {"pk_jobs", "jobs_idempotency_key", "jobs_claimable", "jobs_running"}
    assert "WHERE (status = 'queued'" in defs["jobs_claimable"]
    assert "WHERE (status = 'running'" in defs["jobs_running"]
    assert "WHERE (idempotency_key IS NOT NULL)" in defs["jobs_idempotency_key"]


def test_no_index_covers_heartbeat_columns(conn):
    # A heartbeat updates only these two columns. If any index mentioned
    # them, Postgres could not make the update in-page (HOT).
    for definition in _index_defs(conn, "jobs").values():
        assert "lease_expires_at" not in definition
        assert "heartbeat_at" not in definition


def test_jobs_storage_options(conn):
    options = conn.execute(
        text("SELECT reloptions FROM pg_class WHERE relname = 'jobs'")
    ).scalar_one()
    assert set(options) == {"fillfactor=90", "autovacuum_vacuum_scale_factor=0.02"}
