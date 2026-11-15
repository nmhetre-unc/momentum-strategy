"""
Strategy catalog sync: rows match the code, a re-sync changes nothing, a
schema change adds a new current row and keeps the old one for old jobs,
a strategy that is no longer served stops being current, and concurrent
syncs (several API replicas starting at once) don't collide.
"""

import threading

import pytest
from pydantic import Field
from sqlalchemy import select

from qbt_service.catalog import CATALOG, build_catalog, sync_catalog
from qbt_service.db import strategies
from qbt_service.models import PARAMS_MODELS, StrategyParams

pytestmark = pytest.mark.db


def rows(conn):
    return conn.execute(
        select(strategies.c.id, strategies.c.name, strategies.c.schema_hash,
               strategies.c.param_schema, strategies.c.is_current)
        .order_by(strategies.c.id)
    ).mappings().all()


def current(conn):
    return {r["name"]: r for r in rows(conn) if r["is_current"]}


def test_sync_writes_the_catalog(conn):
    ids = sync_catalog(conn)
    assert set(ids) == set(PARAMS_MODELS)
    stored = current(conn)
    assert set(stored) == set(CATALOG)
    for name, entry in CATALOG.items():
        assert stored[name]["id"] == ids[name]
        assert stored[name]["schema_hash"] == entry.schema_hash
        assert stored[name]["param_schema"] == entry.param_schema


def test_resync_is_a_no_op(conn):
    first = sync_catalog(conn)
    before = rows(conn)
    assert sync_catalog(conn) == first
    assert rows(conn) == before


def _with_extra_param(name: str) -> dict[str, type[StrategyParams]]:
    """The served models, but `name` has gained a parameter (a schema change)."""
    from pydantic import create_model

    changed = create_model(f"{name}_v2", __base__=PARAMS_MODELS[name],
                           extra_knob=(int, Field(default=1, ge=0, le=2)))
    return {**PARAMS_MODELS, name: changed}


def test_schema_change_adds_a_new_current_row(conn):
    old_ids = sync_catalog(conn)
    new_ids = sync_catalog(conn, build_catalog(_with_extra_param("momentum")))

    assert new_ids["momentum"] != old_ids["momentum"]
    assert {k: v for k, v in new_ids.items() if k != "momentum"} == {
        k: v for k, v in old_ids.items() if k != "momentum"}
    momentum_rows = [r for r in rows(conn) if r["name"] == "momentum"]
    assert [(r["id"], r["is_current"]) for r in momentum_rows] == [
        (old_ids["momentum"], False), (new_ids["momentum"], True)]
    assert "extra_knob" in current(conn)["momentum"]["param_schema"]["properties"]

    # Rolling back to the old code makes the old row current again; no third row.
    assert sync_catalog(conn) == old_ids
    assert len([r for r in rows(conn) if r["name"] == "momentum"]) == 2


def test_unserved_strategy_stops_being_current(conn):
    ids = sync_catalog(conn)
    without = {k: v for k, v in PARAMS_MODELS.items() if k != "ml_direction"}
    sync_catalog(conn, build_catalog(without))
    assert "ml_direction" not in current(conn)
    # The row stays: jobs that used it still reference it.
    assert any(r["id"] == ids["ml_direction"] for r in rows(conn))


def test_concurrent_syncs_agree(committed_engine):
    results, failures = [], []
    barrier = threading.Barrier(4)

    def sync():
        try:
            barrier.wait()
            with committed_engine.begin() as c:
                results.append(sync_catalog(c))
        except Exception as exc:  # reported below
            failures.append(exc)

    threads = [threading.Thread(target=sync) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert failures == []
    assert all(r == results[0] for r in results)
    with committed_engine.connect() as c:
        assert len(rows(c)) == len(CATALOG)
