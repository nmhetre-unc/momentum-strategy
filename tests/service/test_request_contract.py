"""
The request contract shared with the dashboard: contracts/backtest_requests.json
lists POST /backtests bodies and whether they are valid. Every case is sent to
the real endpoint; the database holds no prices, so a valid body gets as far as
422 no-price-data and an invalid one stops at 422 validation-error.

The dashboard's client-side validation is checked against the same file by
web/src/features/run/form.contract.test.ts. That test reads each strategy's
params schema from web/openapi.json; the last test here checks those are the
schemas GET /strategies serves.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qbt_service.api import _strategy_list, create_app
from qbt_service.catalog import sync_catalog
from qbt_service.settings import Settings

from .conftest import base_url, migrate, temporary_database

pytestmark = pytest.mark.db

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "contracts" / "backtest_requests.json").read_text())["cases"]


@pytest.fixture(scope="module")
def client():
    with temporary_database(base_url()) as engine:
        migrate(engine)
        with engine.begin() as conn:
            sync_catalog(conn)
        url = engine.url.render_as_string(hide_password=False)
        with TestClient(create_app(Settings(database_url=url))) as c:
            yield c


def test_cases_cover_both_outcomes_and_every_strategy():
    assert {c["valid"] for c in CASES} == {True, False}
    assert {c["body"]["strategy"] for c in CASES} == {
        s.name for s in _strategy_list().strategies}
    assert len({c["name"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_api_agrees(client, case):
    response = client.post("/backtests", json=case["body"])
    assert response.status_code == 422, response.text
    expected = "no-price-data" if case["valid"] else "validation-error"
    assert response.json()["type"] == f"/problems/{expected}", response.text


def test_web_schemas_are_the_served_schemas():
    components = json.loads((ROOT / "web" / "openapi.json").read_text())["components"]["schemas"]
    for strategy in _strategy_list().strategies:
        name = "".join(part.title() for part in strategy.name.split("_")) + "Params"
        assert components[name] == strategy.params_schema, strategy.name
