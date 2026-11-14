"""
POST /backtests and GET /backtests/{id} against a real Postgres.

Idempotency: the same key and body replay the job, canonically equal
bodies count as the same, a different body gets 422, quoted and bare
keys are the same key, malformed keys get 400, and concurrent POSTs with
one key through a real uvicorn server create exactly one job. Then the
error responses (validation, no prices, 404, database down mid-request
and refused), every GET state, the OpenAPI document, and one job end to
end through a real worker.
"""

import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import date

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, make_url, select, text

from qbt.runner import metrics_equal, run
from qbt_service.api import MAX_KEY_LENGTH, create_app, parse_idempotency_key
from qbt_service.catalog import sync_catalog
from qbt_service.db import jobs, price_bars, price_series
from qbt_service.models import parse_request, request_hash
from qbt_service.prices import load_frame
from qbt_service.queue import JobResult, claim, complete, fail
from qbt_service.settings import Settings
from qbt_service.worker import Worker

from .conftest import base_url, migrate, temporary_database
from .factories import store_synthetic_series

pytestmark = pytest.mark.db

START, END = "2018-01-01", "2020-01-01"  # inside the synthetic series (2018-01 .. 2020-04)
PROBLEM_JSON = "application/problem+json"


def body(**overrides):
    return {"strategy": "sma_crossover", "ticker": "SPY", "start": START, "end": END,
            "n_boot": 200} | overrides


class ModuleDB:
    def __init__(self, engine, **info):
        self.engine, self.info = engine, info


@pytest.fixture(scope="module")
def db():
    with temporary_database(base_url()) as engine:
        migrate(engine)
        with engine.begin() as conn:
            series_id = store_synthetic_series(conn)
            strategy_ids = sync_catalog(conn)
        yield ModuleDB(engine, series_id=series_id, strategy_ids=strategy_ids,
                       url=engine.url.render_as_string(hide_password=False),
                       name=engine.url.database)


@pytest.fixture(scope="module")
def client(db):
    with TestClient(create_app(Settings(database_url=db.info["url"], max_attempts=4))) as c:
        yield c


@pytest.fixture(autouse=True)
def empty_queue(db):
    with db.engine.begin() as conn:
        conn.execute(text("DELETE FROM jobs"))


def job_count(db) -> int:
    with db.engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(jobs)).scalar_one()


def stored_job(db, job_id):
    with db.engine.connect() as conn:
        return conn.execute(select(jobs).where(jobs.c.id == job_id)).mappings().one()


def post(client, payload, key=None):
    headers = {} if key is None else {"Idempotency-Key": key}
    return client.post("/backtests", json=payload, headers=headers)


def assert_problem(response, status, slug):
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM_JSON
    problem = response.json()
    assert problem["type"] == f"/problems/{slug}"
    assert problem["status"] == status
    assert problem["title"] and problem["detail"]
    return problem


# ---- POST ----

def test_post_queues_a_job(client, db):
    response = post(client, body())
    assert response.status_code == 202
    accepted = response.json()
    job_id = uuid.UUID(accepted["id"])
    assert accepted == {"id": str(job_id), "status": "queued", "href": f"/backtests/{job_id}"}
    assert response.headers["location"] == f"/backtests/{job_id}"
    assert "idempotent-replayed" not in response.headers

    row = stored_job(db, job_id)
    request = parse_request(body())
    assert row["request"] == request.model_dump(mode="json")  # defaults filled in
    assert row["request"]["params"] == {"short_window": 50, "long_window": 200}
    assert bytes(row["request_hash"]) == request_hash(request)
    assert row["strategy_id"] == db.info["strategy_ids"]["sma_crossover"]
    assert row["price_series_id"] == db.info["series_id"]
    assert row["max_attempts"] == 4  # from the app's settings
    assert row["idempotency_key"] is None
    assert row["status"] == "queued"


def test_without_a_key_every_post_is_a_new_job(client, db):
    first, second = post(client, body()), post(client, body())
    assert first.json()["id"] != second.json()["id"]
    assert job_count(db) == 2


def test_post_uses_the_newest_covering_snapshot(client, db):
    with db.engine.begin() as conn:
        newer = store_synthetic_series(conn, seed=1)  # same span, other prices
    try:
        job_id = post(client, body()).json()["id"]
        assert stored_job(db, job_id)["price_series_id"] == newer
    finally:
        with db.engine.begin() as conn:
            conn.execute(text("DELETE FROM jobs"))
            conn.execute(price_bars.delete().where(price_bars.c.series_id == newer))
            conn.execute(price_series.delete().where(price_series.c.id == newer))


# ---- idempotency ----

def test_same_key_and_body_replays_the_job(client, db):
    first = post(client, body(), key='"abc-1"')
    second = post(client, body(), key='"abc-1"')
    assert first.status_code == second.status_code == 202
    assert first.json()["id"] == second.json()["id"]
    assert second.headers["idempotent-replayed"] == "true"
    assert second.headers["location"] == first.headers["location"]
    assert "idempotent-replayed" not in first.headers
    assert job_count(db) == 1
    assert stored_job(db, first.json()["id"])["idempotency_key"] == "abc-1"


def test_replay_reports_the_jobs_current_status(client, db):
    job_id = post(client, body(), key='"abc-2"').json()["id"]
    with db.engine.begin() as conn:
        conn.execute(jobs.update().where(jobs.c.id == job_id)
                     .values(status="failed", finished_at=func.now(), last_error="x"))
    replay = post(client, body(), key='"abc-2"')
    assert replay.status_code == 202
    assert replay.json()["status"] == "failed"


@pytest.mark.parametrize("equivalent", [
    # keys reordered, defaults spelled out, an int where a float is allowed,
    # a lower-case ticker: all validate to the same request
    {"n_boot": 200, "end": END, "start": START, "ticker": "spy", "strategy": "sma_crossover"},
    body(params={"short_window": 50, "long_window": 200}, cost_bps=5, seed=0),
    body(ticker=" SPY "),
], ids=["reordered", "defaults-explicit", "ticker-case"])
def test_canonically_equal_bodies_are_the_same_request(client, db, equivalent):
    first = post(client, body(), key='"canon"')
    second = post(client, equivalent, key='"canon"')
    assert second.status_code == 202, second.text
    assert second.json()["id"] == first.json()["id"]
    assert second.headers["idempotent-replayed"] == "true"
    assert job_count(db) == 1


@pytest.mark.parametrize("different", [
    body(seed=1),
    body(params={"short_window": 20, "long_window": 200}),
    body(strategy="momentum"),
    body(cost_bps=5.000001),
], ids=["seed", "params", "strategy", "cost"])
def test_same_key_with_a_different_body_is_422(client, db, different):
    first = post(client, body(), key='"reuse"')
    second = post(client, different, key='"reuse"')
    problem = assert_problem(second, 422, "idempotency-key-reused")
    assert "errors" not in problem
    assert first.json()["id"] not in second.text  # another client's job id isn't revealed
    assert job_count(db) == 1
    # The original is untouched and still replays.
    assert post(client, body(), key='"reuse"').json()["id"] == first.json()["id"]


def test_quoted_and_bare_keys_are_the_same_key(client, db):
    first = post(client, body(), key='"same-key"')
    second = post(client, body(), key="same-key")
    assert second.json()["id"] == first.json()["id"]
    assert second.headers["idempotent-replayed"] == "true"


@pytest.mark.parametrize(("header", "key"), [
    ('"plain"', "plain"),
    ("plain", "plain"),
    ('  "padded"\t', "padded"),
    (r'"with \"quotes\" and \\"', 'with "quotes" and \\'),
    ('"has spaces inside"', "has spaces inside"),
    ("bare with spaces", "bare with spaces"),
    ('"' + "k" * MAX_KEY_LENGTH + '"', "k" * MAX_KEY_LENGTH),
    (None, None),
])
def test_parse_idempotency_key(header, key):
    assert parse_idempotency_key(header) == key


@pytest.mark.parametrize("header", [
    "", '""', '"unterminated', '"bad \\n escape"', '"inner " quote"', '"tab\there"',
    "k" * (MAX_KEY_LENGTH + 1), '"' + "k" * (MAX_KEY_LENGTH + 1) + '"', "café",
], ids=["empty", "empty-quoted", "unterminated", "bad-escape", "inner-quote", "tab",
        "too-long", "too-long-quoted", "non-ascii"])
def test_malformed_keys(header):
    with pytest.raises(ValueError):
        parse_idempotency_key(header)


@pytest.mark.parametrize("header", ['""', '"unterminated', "k" * (MAX_KEY_LENGTH + 1)])
def test_malformed_key_is_400_and_creates_nothing(client, db, header):
    assert_problem(post(client, body(), key=header), 400, "invalid-idempotency-key")
    assert job_count(db) == 0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(db):
    """The app under a real uvicorn, with several worker threads per request burst."""
    port = free_port()
    env = os.environ | {"DATABASE_URL": db.info["url"]}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "--factory", "qbt_service.api:create_app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while True:
        try:
            if httpx2.get(f"{url}/strategies", timeout=1).status_code == 200:
                break
        except httpx2.TransportError:
            pass
        assert proc.poll() is None, "uvicorn exited"
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.1)
    yield url
    proc.terminate()
    proc.wait(10)


def concurrent_posts(url, payloads, key):
    """POSTs every payload at once (released together by a barrier)."""
    barrier = threading.Barrier(len(payloads))
    responses = [None] * len(payloads)

    def send(i):
        with httpx2.Client(base_url=url, timeout=30) as http:
            barrier.wait()
            responses[i] = http.post("/backtests", json=payloads[i],
                                     headers={"Idempotency-Key": key})

    threads = [threading.Thread(target=send, args=(i,)) for i in range(len(payloads))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    return responses


def test_concurrent_posts_with_one_key_create_one_job(server, db):
    for round_ in range(5):
        key = f'"concurrent-{round_}"'
        responses = concurrent_posts(server, [body()] * 16, key)
        assert all(r.status_code == 202 for r in responses), [r.text for r in responses]
        assert len({r.json()["id"] for r in responses}) == 1
        fresh = [r for r in responses if "idempotent-replayed" not in r.headers]
        assert len(fresh) == 1  # exactly one request created the job
    assert job_count(db) == 5


def test_concurrent_posts_with_one_key_and_different_bodies(server, db):
    for round_ in range(5):
        key = f'"mixed-{round_}"'
        payloads = [body(seed=i % 2) for i in range(16)]
        responses = concurrent_posts(server, payloads, key)
        with db.engine.connect() as conn:
            winner = conn.execute(select(jobs.c.id, jobs.c.request)
                                  .where(jobs.c.idempotency_key == key[1:-1])).one()
        for payload, response in zip(payloads, responses, strict=True):
            if payload["seed"] == winner.request["seed"]:
                assert response.status_code == 202
                assert response.json()["id"] == str(winner.id)
            else:
                assert response.status_code == 422
                assert response.json()["type"] == "/problems/idempotency-key-reused"
    assert job_count(db) == 5


# ---- POST errors ----

@pytest.mark.parametrize(("payload", "loc"), [
    (body(params={"short_window": 80, "long_window": 60}), ["body", "sma_crossover", "params"]),
    (body(params={"short_window": 5.0}),
     ["body", "sma_crossover", "params", "short_window"]),
    (body(n_boot="100"), ["body", "sma_crossover", "n_boot"]),
    (body(strategy="nope"), ["body"]),
    (body(colour="red"), ["body", "sma_crossover", "colour"]),
    (body(start=END, end=START), ["body", "sma_crossover"]),
    (body(ticker="S P Y"), ["body", "sma_crossover", "ticker"]),
])
def test_invalid_body_is_a_validation_problem(client, db, payload, loc):
    problem = assert_problem(post(client, payload), 422, "validation-error")
    assert [e["loc"] for e in problem["errors"]] == [loc]
    assert all(set(e) == {"loc", "msg", "type"} for e in problem["errors"])
    assert job_count(db) == 0


def test_malformed_json_is_a_validation_problem(client):
    response = client.post("/backtests", content=b"{not json",
                           headers={"Content-Type": "application/json"})
    problem = assert_problem(response, 422, "validation-error")
    assert problem["errors"][0]["type"] == "json_invalid"


@pytest.mark.parametrize("payload", [
    body(ticker="QQQ"),                          # no series for the ticker
    body(start="2017-01-01"),                    # starts before the snapshot
    body(end="2021-01-01"),                      # ends after it
], ids=["ticker", "early-start", "late-end"])
def test_no_covering_snapshot_is_422(client, db, payload):
    problem = assert_problem(post(client, payload, key='"np"'), 422, "no-price-data")
    assert "python -m qbt_service.ingest" in problem["detail"]
    assert job_count(db) == 0


def test_unknown_route_is_a_problem(client):
    assert_problem(client.get("/nope"), 404, "http-error")
    assert_problem(client.delete("/backtests"), 405, "http-error")


# ---- GET ----

def submit(client) -> str:
    return post(client, body()).json()["id"]


def test_get_a_queued_job(client, db):
    job_id = submit(client)
    response = client.get(f"/backtests/{job_id}")
    assert response.status_code == 200
    got = response.json()
    row = stored_job(db, job_id)
    assert got["id"] == job_id
    assert got["status"] == "queued"
    assert (got["attempts"], got["max_attempts"]) == (0, 4)
    assert got["request"] == row["request"] == parse_request(body()).model_dump(mode="json")
    assert got["price_series_id"] == db.info["series_id"]
    assert got["created_at"] and got["run_after"]
    for absent in ("started_at", "heartbeat_at", "finished_at", "last_error", "metrics",
                   "result"):
        assert got[absent] is None


def test_get_a_running_job(client, db):
    job_id = submit(client)
    with db.engine.begin() as conn:
        claimed = claim(conn, "w1", 30)
    assert str(claimed.id) == job_id
    got = client.get(f"/backtests/{job_id}").json()
    assert got["status"] == "running"
    assert got["attempts"] == 1
    assert got["started_at"] and got["heartbeat_at"]
    assert got["metrics"] is None and got["finished_at"] is None
    assert "lease_token" not in got and str(claimed.lease_token) not in str(got)


def test_get_a_job_waiting_to_retry(client, db):
    job_id = submit(client)
    with db.engine.begin() as conn:
        claimed = claim(conn, "w1", 30)
        fail(conn, claimed.id, claimed.lease_token, "connection lost", retry=True,
             backoff_base=60)
    got = client.get(f"/backtests/{job_id}").json()
    assert got["status"] == "queued"
    assert got["attempts"] == 1
    assert got["last_error"] == "connection lost"
    assert got["run_after"] > got["created_at"]
    assert got["finished_at"] is None


def test_get_a_failed_job(client, db):
    job_id = submit(client)
    with db.engine.begin() as conn:
        claimed = claim(conn, "w1", 30)
        fail(conn, claimed.id, claimed.lease_token, "backtest raised: boom", retry=False)
    got = client.get(f"/backtests/{job_id}").json()
    assert got["status"] == "failed"
    assert got["last_error"] == "backtest raised: boom"
    assert got["finished_at"] is not None
    assert got["metrics"] is None and got["result"] is None


def test_get_a_succeeded_job_with_nan_metrics(client, db):
    job_id = submit(client)
    with db.engine.begin() as conn:
        claimed = claim(conn, "w1", 30)
        complete(conn, claimed.id, claimed.lease_token, JobResult(
            qbt_version="0.1.0", numpy_version="2.0.0", bootstrap_impl=None, duration_ms=12,
            dates=[date(2020, 1, 2), date(2020, 1, 3)], equity=[1.0, 1.0],
            position=[0.0, 0.0],
            metrics={"total_return": 0.0, "sharpe_ratio": float("nan"), "num_trades": 0.0}))
    response = client.get(f"/backtests/{job_id}")
    assert response.status_code == 200
    got = response.json()
    assert got["status"] == "succeeded"
    assert got["metrics"] == {"num_trades": 0.0, "sharpe_ratio": None, "total_return": 0.0}
    assert list(got["metrics"]) == sorted(got["metrics"])
    result = got["result"]
    assert result["attempt"] == 1 and result["n_days"] == 2 and result["duration_ms"] == 12
    assert result["bootstrap_impl"] is None
    assert got["last_error"] is None and got["finished_at"]


def test_unknown_job_is_404(client):
    unknown = uuid.uuid4()
    problem = assert_problem(client.get(f"/backtests/{unknown}"), 404, "job-not-found")
    assert str(unknown) in problem["detail"]


def test_malformed_job_id_is_422(client):
    problem = assert_problem(client.get("/backtests/abc"), 422, "validation-error")
    assert problem["errors"][0]["loc"] == ["path", "job_id"]


# ---- the database going away ----

def admin_engine(db):
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool
    return create_engine(make_url(db.info["url"]).set(database="postgres"),
                         isolation_level="AUTOCOMMIT", poolclass=NullPool)


def terminate_app_connections(admin, db):
    with admin.connect() as conn:
        conn.execute(text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                          "WHERE datname = :name AND pid <> pg_backend_pid()"),
                     {"name": db.info["name"]})


def test_refused_connections_are_503_and_recover(client, db):
    job_id = submit(client)
    admin = admin_engine(db)
    name = db.info["name"]
    try:
        with admin.connect() as conn:
            conn.execute(text(f'ALTER DATABASE "{name}" ALLOW_CONNECTIONS false'))
        terminate_app_connections(admin, db)  # pooled ones fail pre-ping and reconnect

        for response in (post(client, body(), key='"down"'), client.get(f"/backtests/{job_id}")):
            problem = assert_problem(response, 503, "database-unavailable")
            assert response.headers["retry-after"] == "5"
            assert "same Idempotency-Key" in problem["detail"]
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'ALTER DATABASE "{name}" ALLOW_CONNECTIONS true'))
        admin.dispose()

    assert client.get(f"/backtests/{job_id}").status_code == 200
    retried = post(client, body(), key='"down"')
    assert retried.status_code == 202 and "idempotent-replayed" not in retried.headers
    assert job_count(db) == 2


def test_connection_lost_mid_insert_is_503_and_safe_to_retry(client, db):
    """The INSERT is blocked on a table lock, its backend killed: 503, no job.
    The retry with the same key creates the job, once."""
    admin = admin_engine(db)
    box = {}
    with db.engine.connect() as holder:
        holder.execute(text("LOCK TABLE jobs IN EXCLUSIVE MODE"))
        thread = threading.Thread(
            target=lambda: box.setdefault("r", post(client, body(), key='"cut"')))
        thread.start()
        deadline = time.monotonic() + 30
        while True:
            with admin.connect() as conn:
                pid = conn.execute(text(
                    "SELECT pid FROM pg_stat_activity WHERE datname = :name "
                    "AND wait_event_type = 'Lock' AND query LIKE 'INSERT INTO jobs%'"),
                    {"name": db.info["name"]}).scalar_one_or_none()
                if pid is not None:
                    conn.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
                    break
            assert time.monotonic() < deadline, "the INSERT never blocked"
            time.sleep(0.02)
        thread.join(30)
        holder.rollback()
    admin.dispose()

    assert_problem(box["r"], 503, "database-unavailable")
    assert job_count(db) == 0
    retried = post(client, body(), key='"cut"')
    assert retried.status_code == 202 and "idempotent-replayed" not in retried.headers
    assert post(client, body(), key='"cut"').headers["idempotent-replayed"] == "true"
    assert job_count(db) == 1


# ---- OpenAPI ----

def test_openapi_documents_the_backtest_endpoints(client):
    schema = client.get("/openapi.json").json()
    post_op = schema["paths"]["/backtests"]["post"]
    get_op = schema["paths"]["/backtests/{job_id}"]["get"]

    assert "Idempotency-Key" in post_op["description"]
    assert "409" in post_op["description"]  # why it is never returned
    assert set(post_op["responses"]) == {"202", "400", "422", "503"}
    assert set(get_op["responses"]) == {"200", "404", "422", "503"}
    assert set(post_op["responses"]["202"]["headers"]) == {"Location", "Idempotent-Replayed"}
    assert "Retry-After" in post_op["responses"]["503"]["headers"]
    (header,) = post_op["parameters"]
    assert header["name"] == "Idempotency-Key" and header["in"] == "header"
    assert not header["required"] and header["description"]

    for op in (post_op, get_op):
        for code, response in op["responses"].items():
            assert response["description"]
            if code.startswith(("4", "5")):
                assert list(response["content"]) == [PROBLEM_JSON]
                for example in response["content"][PROBLEM_JSON]["examples"].values():
                    assert example["value"]["status"] == int(code)

    examples = post_op["requestBody"]["content"]["application/json"]["examples"]
    assert {"sma_crossover", "momentum", "mean_reversion", "ml_direction"} <= set(examples)
    for example in examples.values():
        parse_request(example["value"])  # every request example is valid

    from qbt_service.api import BacktestJob
    states = get_op["responses"]["200"]["content"]["application/json"]["examples"]
    assert set(states) == {"queued", "running", "succeeded", "failed"}
    for example in states.values():
        BacktestJob.model_validate(example["value"])


# ---- end to end ----

def test_post_run_get(client, db):
    response = post(client, body(strategy="momentum"), key='"e2e"')
    job_id = response.json()["id"]

    worker = Worker(Settings(database_url=db.info["url"], worker_id="e2e", poll_seconds=0.1))
    worker.start()
    try:
        assert worker.run_once().label == "succeeded"
    finally:
        worker.stop()

    got = client.get(response.headers["location"]).json()
    assert got["id"] == job_id
    assert got["status"] == "succeeded"
    assert got["attempts"] == 1 and got["result"]["attempt"] == 1

    request = parse_request(body(strategy="momentum"))
    with db.engine.connect() as conn:
        frame = load_frame(conn, db.info["series_id"], request.start, request.end)
    expected = run(frame, "momentum", request.params.model_dump(), cost_bps=request.cost_bps,
                   n_boot=request.n_boot, seed=request.seed)
    assert got["result"]["n_days"] == len(expected.result)
    assert metrics_equal({k: float("nan") if v is None else v
                          for k, v in got["metrics"].items()}, expected.metrics)
