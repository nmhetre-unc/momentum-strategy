"""
GET /backtests/{id}/series against a real Postgres, with jobs run by a real
worker: the stored series come back unchanged, the drawdown's minimum is the
stored max_drawdown metric, a learning strategy's out_of_sample_from is
qbt.timesplit's split date on the job's own snapshot (and its positions are
flat before it), and the errors are problem details: 404 for an unknown job,
409 for one that hasn't succeeded, 422 for a malformed id.
"""

import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from qbt.analytics import drawdown_series
from qbt.ml import ml_direction_signal
from qbt.timesplit import out_of_sample_start, split_date
from qbt_service.api import BacktestSeries, create_app
from qbt_service.catalog import sync_catalog
from qbt_service.db import results
from qbt_service.models import parse_request
from qbt_service.prices import load_frame
from qbt_service.queue import claim, fail
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


@pytest.fixture(scope="module")
def db():
    with temporary_database(base_url()) as engine:
        migrate(engine)
        with engine.begin() as conn:
            series_id = store_synthetic_series(conn)
            sync_catalog(conn)
        yield engine, series_id, engine.url.render_as_string(hide_password=False)


@pytest.fixture(scope="module")
def client(db):
    _, _, url = db
    with TestClient(create_app(Settings(database_url=url))) as c:
        yield c


@pytest.fixture(autouse=True)
def empty_queue(db):
    engine, _, _ = db
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM jobs"))


def submit(client, payload) -> str:
    response = client.post("/backtests", json=payload)
    assert response.status_code == 202, response.text
    return response.json()["id"]


def run_job(client, db, payload) -> str:
    """Submits `payload` and runs it to success on a real worker."""
    job_id = submit(client, payload)
    _, _, url = db
    worker = Worker(Settings(database_url=url, worker_id="series", poll_seconds=0.1))
    worker.start()
    try:
        assert worker.run_once().label == "succeeded"
    finally:
        worker.stop()
    return job_id


def assert_problem(response, status, slug):
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM_JSON
    problem = response.json()
    assert problem["type"] == f"/problems/{slug}"
    assert problem["status"] == status
    assert problem["title"] and problem["detail"]
    return problem


def stored_result(db, job_id):
    engine, _, _ = db
    with engine.connect() as conn:
        return conn.execute(
            select(results.c.dates, results.c.equity, results.c.position)
            .where(results.c.job_id == uuid.UUID(job_id))
        ).mappings().one()


# ---- a succeeded job ----

def test_series_is_the_stored_result(client, db):
    job_id = run_job(client, db, body())
    response = client.get(f"/backtests/{job_id}/series")
    assert response.status_code == 200, response.text
    series = BacktestSeries.model_validate(response.json())

    stored = stored_result(db, job_id)
    assert series.dates == list(stored["dates"])
    assert series.equity == list(stored["equity"])
    assert series.position == list(stored["position"])
    assert len(series.drawdown) == len(series.dates)

    job = client.get(f"/backtests/{job_id}").json()
    assert len(series.dates) == job["result"]["n_days"]
    assert series.strategy == "sma_crossover"
    assert series.strategy_version == job["strategy_version"]
    # sma_crossover learns nothing: every day is out of sample.
    assert series.out_of_sample_from is None


@pytest.mark.parametrize("strategy", ["sma_crossover", "momentum", "mean_reversion",
                                      "ml_direction"])
def test_drawdown_minimum_is_the_stored_max_drawdown(client, db, strategy):
    job_id = run_job(client, db, body(strategy=strategy))
    series = client.get(f"/backtests/{job_id}/series").json()
    metrics = client.get(f"/backtests/{job_id}").json()["metrics"]
    # Exactly the stored metric: the same formula on the same curve.
    assert min(series["drawdown"]) == metrics["max_drawdown"]
    assert series["drawdown"] == drawdown_series(pd.Series(series["equity"])).tolist()
    assert max(series["drawdown"]) == 0.0


# ---- learning strategies ----

@pytest.mark.parametrize("train_frac", [None, 0.6])
def test_out_of_sample_from_is_the_split_date_of_the_jobs_snapshot(client, db, train_frac):
    params = {} if train_frac is None else {"params": {"train_frac": train_frac}}
    payload = body(strategy="ml_direction", **params)
    job_id = run_job(client, db, payload)
    series = client.get(f"/backtests/{job_id}/series").json()

    engine, series_id, _ = db
    request = parse_request(payload)
    with engine.connect() as conn:
        frame = load_frame(conn, series_id, request.start, request.end)
    frac = request.params.model_dump()["train_frac"]
    expected = split_date(frame.index, frac).date()
    assert series["out_of_sample_from"] == expected.isoformat()
    # The same day the strategy's own signal records on that frame.
    assert out_of_sample_start(ml_direction_signal(frame, train_frac=frac)).date() == expected

    # Flat through the training period (positions shift one day, so the
    # first day that can hold one is the day after the split date), and
    # the metrics cover it: the series starts at the job's start, not there.
    first_oos = series["dates"].index(expected.isoformat())
    assert all(p == 0.0 for p in series["position"][:first_oos + 1])
    assert first_oos > 0 and series["dates"][0] < expected.isoformat()


# ---- errors ----

def test_unknown_job_is_404(client):
    unknown = uuid.uuid4()
    problem = assert_problem(client.get(f"/backtests/{unknown}/series"), 404, "job-not-found")
    assert str(unknown) in problem["detail"]


def test_queued_job_is_409(client):
    job_id = submit(client, body())
    problem = assert_problem(client.get(f"/backtests/{job_id}/series"), 409, "job-not-succeeded")
    assert "queued" in problem["detail"] and f"GET /backtests/{job_id}" in problem["detail"]


def test_running_job_is_409(client, db):
    job_id = submit(client, body())
    engine, _, _ = db
    with engine.begin() as conn:
        assert claim(conn, "w1", 30) is not None
    problem = assert_problem(client.get(f"/backtests/{job_id}/series"), 409, "job-not-succeeded")
    assert "running" in problem["detail"]


def test_failed_job_is_409(client, db):
    job_id = submit(client, body())
    engine, _, _ = db
    with engine.begin() as conn:
        claimed = claim(conn, "w1", 30)
        fail(conn, claimed.id, claimed.lease_token, "boom", retry=False)
    problem = assert_problem(client.get(f"/backtests/{job_id}/series"), 409, "job-not-succeeded")
    assert "failed" in problem["detail"] and "last_error" in problem["detail"]


def test_malformed_job_id_is_422(client):
    problem = assert_problem(client.get("/backtests/abc/series"), 422, "validation-error")
    assert problem["errors"][0]["loc"] == ["path", "job_id"]


def test_openapi_documents_the_endpoint(client):
    schema = client.get("/openapi.json").json()
    op = schema["paths"]["/backtests/{job_id}/series"]["get"]
    assert set(op["responses"]) == {"200", "404", "409", "422", "503"}
    for status in ("404", "409", "422", "503"):
        assert list(op["responses"][status]["content"]) == [PROBLEM_JSON]
    props = schema["components"]["schemas"]["BacktestSeries"]["properties"]
    assert set(props) == {"job_id", "strategy", "strategy_version", "dates", "equity",
                          "drawdown", "position", "out_of_sample_from"}
    assert "FULL period" in props["out_of_sample_from"]["description"]
