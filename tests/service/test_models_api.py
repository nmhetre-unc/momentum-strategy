"""
The model registry and its API (C3): python -m qbt_service.train,
GET /models, GET /models/{version} and POST /predict, on a synthetic
price snapshot, with the training evaluation shrunk to fit it.

- Versioning: each new model gets the next version, concurrent
  registrations included; training the identical model again finds the
  existing version instead of adding one.
- Reproducibility: the same snapshot and seed give a byte-identical
  artifact (not merely identical predictions): LightGBM is deterministic
  here and its text model format carries no timestamps.
- Integrity: an artifact or serving input that doesn't match its stored
  SHA-256 is refused (500, nothing scored, nothing logged).
"""

import dataclasses
import hashlib
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

pytest.importorskip("lightgbm")

from qbt.ml.models import from_artifact  # noqa: E402
from qbt.ml.train import TrainingConfig, train_for_serving  # noqa: E402
from qbt_service import train as train_cli  # noqa: E402
from qbt_service.api import create_app  # noqa: E402
from qbt_service.db import model_registry, predictions, price_series  # noqa: E402
from qbt_service.prices import load_frame  # noqa: E402
from qbt_service.registry import (  # noqa: E402
    IntegrityError,
    latest_version,
    list_models,
    load_model,
    register,
    serving_input,
)
from qbt_service.settings import Settings  # noqa: E402

from .conftest import base_url, migrate, temporary_database  # noqa: E402
from .factories import store_synthetic_series  # noqa: E402

pytestmark = pytest.mark.db

PROBLEM_JSON = "application/problem+json"
SMALL = {"min_train": 24, "test_size": 12, "n_boot": 200}
SERIES = {"start": "2012-01-02", "periods": 1500, "seed": 3}


def _cli_args(window, *extra: str) -> list[str]:
    first, end = window
    return ["SPY", "--start", first.isoformat(), "--end", end.isoformat(),
            "--features", "base", *extra]


def _window(conn, series_id):
    row = conn.execute(select(price_series.c.fetch_start, price_series.c.fetch_end)
                       .where(price_series.c.id == series_id)).one()
    return row.fetch_start, row.fetch_end


class ModuleDB:
    def __init__(self, engine, **info):
        self.engine, self.info = engine, info


@pytest.fixture(scope="module")
def db():
    with temporary_database(base_url()) as engine, pytest.MonkeyPatch.context() as mp:
        migrate(engine)
        url = engine.url.render_as_string(hide_password=False)
        mp.setenv("DATABASE_URL", url)
        with engine.begin() as conn:
            series_id = store_synthetic_series(conn, **SERIES)
            window = _window(conn, series_id)
        # Version 1: regime (the default target); version 2: direction.
        assert train_cli.main(_cli_args(window), SMALL) == 0
        assert train_cli.main(_cli_args(window, "--target", "direction"), SMALL) == 0
        yield ModuleDB(engine, series_id=series_id, window=window, url=url)


@pytest.fixture(scope="module")
def client(db):
    with TestClient(create_app(Settings(database_url=db.info["url"]))) as c:
        yield c


def _rows(db):
    with db.engine.connect() as conn:
        return {r["version"]: r for r in list_models(conn)}


def _prediction_count(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(predictions)).scalar_one()


# ---- versioning ----

def test_each_trained_model_is_a_new_version(db):
    rows = _rows(db)
    assert sorted(rows) == [1, 2]
    assert rows[1]["target"] == "regime" and rows[2]["target"] == "direction"
    assert all(r["price_series_id"] == db.info["series_id"] for r in rows.values())
    with db.engine.connect() as conn:
        assert latest_version(conn) == 2


def test_training_the_same_model_again_finds_its_version(db, capsys, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", db.info["url"])
    assert train_cli.main(_cli_args(db.info["window"]), SMALL) == 0
    assert "unchanged: identical to registered model version 1" in capsys.readouterr().out
    assert sorted(_rows(db)) == [1, 2]


def test_train_without_a_snapshot_fails(db, capsys, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", db.info["url"])
    assert train_cli.main(["QQQ", "--start", "2012-01-02", "--end", "2015-01-01"]) == 1
    assert "No price snapshot for QQQ" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        train_cli.main(["SPY", "--start", "2015-01-01", "--end", "2012-01-01"])


def test_concurrent_registrations_get_consecutive_versions(committed_engine, db):
    with db.engine.connect() as conn:
        frame = load_frame(conn, db.info["series_id"])
    base = train_for_serving(frame, TrainingConfig(feature_set="base", **SMALL))
    with committed_engine.begin() as conn:
        series_id = store_synthetic_series(conn, **SERIES)
    variants = []
    for i in range(6):
        artifact = base.artifact + b" " * (i + 1)   # distinct content, distinct hash
        variants.append(dataclasses.replace(
            base, artifact=artifact, artifact_sha256=hashlib.sha256(artifact).digest()))
    versions: list[int] = []
    barrier = threading.Barrier(len(variants))

    def worker(trained):
        barrier.wait()
        with committed_engine.begin() as conn:
            versions.append(register(conn, series_id, trained).version)

    threads = [threading.Thread(target=worker, args=(v,)) for v in variants]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(versions) == list(range(1, len(variants) + 1))


# ---- reproducibility ----

def test_same_snapshot_and_seed_give_a_byte_identical_artifact(db):
    with db.engine.connect() as conn:
        frame = load_frame(conn, db.info["series_id"], *db.info["window"])
        stored = conn.execute(select(model_registry.c.artifact, model_registry.c.artifact_sha256)
                              .where(model_registry.c.version == 1)).one()
    again = train_for_serving(frame, TrainingConfig(feature_set="base", **SMALL))
    assert again.artifact == bytes(stored.artifact)
    assert again.artifact_sha256 == bytes(stored.artifact_sha256)


# ---- integrity ----

def _copy_version(conn, source: int, target: int, **changes: str) -> None:
    """Copies a registry row to a new version, with SQL expressions replacing some columns."""
    columns = [c.name for c in model_registry.c if c.name != "version"]
    select_list = ", ".join(changes.get(c, c) for c in columns)
    conn.execute(text(
        f"INSERT INTO model_registry (version, {', '.join(columns)}) "
        f"SELECT :target, {select_list} FROM model_registry WHERE version = :source"
    ), {"target": target, "source": source})


@pytest.fixture
def tampered(committed_engine, db):
    """
    A database of its own: version 1 intact, version 2 with one artifact
    byte flipped, version 3 with one serving feature nudged; each copy on
    its own snapshot, since (snapshot, artifact hash) is unique.
    """
    with db.engine.connect() as conn:
        frame = load_frame(conn, db.info["series_id"], *db.info["window"])
    trained = train_for_serving(frame, TrainingConfig(feature_set="base", **SMALL))
    with committed_engine.begin() as conn:
        series_id = store_synthetic_series(conn, **SERIES)
        other = [store_synthetic_series(conn, **(SERIES | {"seed": seed})) for seed in (4, 5)]
        register(conn, series_id, trained)
        _copy_version(conn, 1, 2, price_series_id=str(other[0]),
                      artifact="set_byte(artifact, length(artifact) - 20, "
                               "get_byte(artifact, length(artifact) - 20) # 1)")
        _copy_version(conn, 1, 3, price_series_id=str(other[1]),
                      serving_features="jsonb_set(serving_features, '{ret_1m}', "
                                       "to_jsonb((serving_features->>'ret_1m')::float8 + 1e-9))")
    return committed_engine


def test_load_model_verifies_the_artifact_sha256(tampered):
    with tampered.connect() as conn:
        load_model(conn, 1)
        with pytest.raises(IntegrityError, match="artifact doesn't match its SHA-256"):
            load_model(conn, 2)
        rows = {r["version"]: r for r in list_models(conn)}
        serving_input(rows[1])
        with pytest.raises(IntegrityError, match="serving features don't match"):
            serving_input(rows[3])
        with pytest.raises(LookupError):
            load_model(conn, 9)


def test_predict_refuses_a_corrupted_model(tampered):
    with TestClient(create_app(Settings.from_env())) as client:
        assert client.post("/predict", json={"version": 1}).status_code == 200
        for version, detail in ((2, "artifact"), (3, "serving features")):
            response = client.post("/predict", json={"version": version})
            assert response.status_code == 500
            assert response.headers["content-type"] == PROBLEM_JSON
            problem = response.json()
            assert problem["type"] == "/problems/model-integrity-error"
            assert detail in problem["detail"]
    assert _prediction_count(tampered) == 1


# ---- the endpoints ----

def test_list_models_shows_accuracy_next_to_the_baseline(client, db):
    rows = _rows(db)
    body = client.get("/models").json()
    assert [m["version"] for m in body["models"]] == [1, 2]
    for entry in body["models"]:
        row = rows[entry["version"]]
        assert entry["out_of_sample"]["accuracy"] == row["oos_accuracy"]
        assert entry["out_of_sample"]["n_months"] == row["oos_months"]
        assert entry["baseline"]["name"] == row["baseline_name"]
        assert entry["baseline"]["accuracy"] == row["baseline_accuracy"]
        assert entry["baseline"]["beats"] == row["beats_baseline"]
        assert ("did not beat" in entry["summary"]) == (not row["beats_baseline"])
        assert entry["artifact"]["sha256"] == bytes(row["artifact_sha256"]).hex()
        assert "artifact" not in entry["artifact"]          # never the bytes themselves
    assert body["models"][0]["baseline"]["name"] == "rules_regime_lookup"
    assert body["models"][1]["baseline"]["name"] == "majority_class"


def test_get_model(client):
    detail = client.get("/models/1").json()
    assert detail["version"] == 1 and detail["metrics"]["bar"]["name"] == "rules_regime_lookup"
    assert detail["serving_features"] and detail["training"]["config"]["min_train"] == 24
    assert [c["name"] for c in detail["classes"]] == [
        "Calm Uptrend", "Quiet Range", "Volatile Rally", "Crisis / Selloff"]
    missing = client.get("/models/9")
    assert missing.status_code == 404 and missing.headers["content-type"] == PROBLEM_JSON
    assert missing.json()["type"] == "/problems/model-not-found"
    assert client.get("/models/one").json()["type"] == "/problems/validation-error"


@pytest.mark.parametrize("payload, version", [(None, 2), ({}, 2), ({"version": 1}, 1)])
def test_predict_scores_and_logs(client, db, payload, version):
    before = _prediction_count(db.engine)
    response = (client.post("/predict") if payload is None
                else client.post("/predict", json=payload))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_version"] == version
    row = _rows(db)[version]
    assert body["as_of"] == row["serving_as_of"].isoformat()
    assert body["features_sha256"] == bytes(row["serving_features_sha256"]).hex()
    assert body["out_of_sample"]["accuracy"] == row["oos_accuracy"]
    assert body["baseline"]["accuracy"] == row["baseline_accuracy"]

    with db.engine.connect() as conn:
        model = load_model(conn, version)
        _, features = serving_input(row)
        logged = conn.execute(select(predictions)
                              .where(predictions.c.id == body["prediction_id"])).one()
    expected = model.predict_proba(features)[0]
    assert [p["probability"] for p in body["probabilities"]] == expected.tolist()
    assert body["prediction"] == model.classes[int(np.argmax(expected))]
    assert _prediction_count(db.engine) == before + 1
    assert (logged.model_version, logged.as_of.isoformat()) == (version, body["as_of"])
    assert list(logged.probabilities) == expected.tolist()
    assert logged.prediction == body["prediction"]


def test_predict_matches_the_artifact_as_trained(client, db):
    """The served model is the registered artifact, loaded: same probabilities as a fresh fit."""
    with db.engine.connect() as conn:
        frame = load_frame(conn, db.info["series_id"], *db.info["window"])
        row = _rows(db)[1]
    fresh = train_for_serving(frame, TrainingConfig(feature_set="base", **SMALL))
    _, features = serving_input(row)
    served = client.post("/predict", json={"version": 1}).json()
    assert [p["probability"] for p in served["probabilities"]] == \
        from_artifact(fresh.artifact).predict_proba(features)[0].tolist()


@pytest.mark.parametrize("payload, slug", [
    ({"version": 9}, "unknown-model-version"),
    ({"version": 0}, "validation-error"),
    ({"version": "1"}, "validation-error"),
    ({"version": 1.0}, "validation-error"),
    ({"model": 1}, "validation-error"),
])
def test_predict_errors_are_problems(client, db, payload, slug):
    before = _prediction_count(db.engine)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM_JSON
    assert response.json()["type"] == f"/problems/{slug}"
    assert _prediction_count(db.engine) == before


def test_predict_without_any_model(committed_engine):
    with TestClient(create_app(Settings.from_env())) as client:
        response = client.post("/predict")
        assert response.status_code == 422
        assert response.json()["type"] == "/problems/no-model-registered"
        assert client.get("/models").json() == {"models": []}


def test_openapi_documents_the_model_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/models", "/models/{version}", "/predict"} <= set(paths)
    problem = paths["/predict"]["post"]["responses"]["500"]["content"]
    assert "application/problem+json" in problem

