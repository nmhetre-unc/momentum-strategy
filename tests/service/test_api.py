"""GET /strategies, and the app's startup catalog sync."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from qbt_service.api import create_app
from qbt_service.catalog import CATALOG
from qbt_service.db import strategies
from qbt_service.models import PARAMS_MODELS, parse_request
from qbt_service.settings import Settings

pytestmark = pytest.mark.db


@pytest.fixture
def client(committed_engine):
    app = create_app(Settings.from_env())  # DATABASE_URL -> committed_engine's database
    with TestClient(app) as test_client:
        yield test_client


def test_startup_syncs_the_catalog(client, committed_engine):
    with committed_engine.connect() as conn:
        stored = conn.execute(
            select(strategies.c.name, strategies.c.id).where(strategies.c.is_current)
        ).all()
    assert dict(stored) == client.app.state.strategy_ids
    assert set(client.app.state.strategy_ids) == set(CATALOG)


def test_list_strategies(client):
    response = client.get("/strategies")
    assert response.status_code == 200
    body = response.json()
    assert [s["name"] for s in body["strategies"]] == sorted(PARAMS_MODELS)
    for item in body["strategies"]:
        entry = CATALOG[item["name"]]
        assert item["params_schema"] == PARAMS_MODELS[item["name"]].model_json_schema()
        assert item["defaults"] == PARAMS_MODELS[item["name"]]().model_dump()
        assert item["schema_hash"] == entry.schema_hash.hex()
        assert item["family"] and item["description"]


def test_listed_defaults_make_a_valid_request(client):
    for item in client.get("/strategies").json()["strategies"]:
        parse_request({"strategy": item["name"], "params": item["defaults"], "ticker": "SPY",
                       "start": "2010-01-01", "end": "2020-01-01"})


def test_openapi_documents_the_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "get" in paths["/strategies"]


def test_startup_fails_without_a_database():
    # Port 1 on localhost: refused at once, no wait for a timeout.
    app = create_app(Settings(database_url="postgresql+psycopg://qbt:qbt@127.0.0.1:1/qbt"))
    with pytest.raises(OperationalError), TestClient(app):
        pass
