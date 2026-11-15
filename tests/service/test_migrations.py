"""
Alembic migrations: a full upgrade/downgrade round trip, a single head,
and agreement between the migrations and qbt_service.db (what
`alembic check` verifies in CI), including a check that a deliberate
difference is actually detected, so the agreement test can't pass vacuously.
"""

import io

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Column, MetaData, Text, inspect, text
from sqlalchemy import exc as sa_exc

from qbt_service.db import metadata

from .conftest import alembic_config, migrate

pytestmark = pytest.mark.db

TABLES = {"price_series", "price_bars", "strategies", "jobs", "results", "metrics",
          "model_registry", "predictions"}


def _diff(engine, target: MetaData) -> list:
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        return compare_metadata(context, target)


def test_single_head():
    assert ScriptDirectory.from_config(alembic_config()).get_heads() == ["0004"]


def test_upgrade_downgrade_upgrade(empty_engine):
    migrate(empty_engine, "upgrade", "head")
    assert set(inspect(empty_engine).get_table_names()) >= TABLES

    migrate(empty_engine, "downgrade", "base")
    assert set(inspect(empty_engine).get_table_names()) == {"alembic_version"}
    with empty_engine.connect() as conn:
        assert conn.execute(
            text("SELECT count(*) FROM pg_type WHERE typname = 'job_status'")
        ).scalar_one() == 0

    migrate(empty_engine, "upgrade", "head")
    assert set(inspect(empty_engine).get_table_names()) >= TABLES


def test_0003_backfills_version_1_and_refuses_a_lossy_downgrade(empty_engine):
    migrate(empty_engine, "upgrade", "0002")
    with empty_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO strategies (name, schema_hash, param_schema) "
            "VALUES ('momentum', decode(repeat('ab', 32), 'hex'), '{}')"))
    migrate(empty_engine, "upgrade", "0003")
    with empty_engine.begin() as conn:
        assert conn.execute(text("SELECT version FROM strategies")).scalar_one() == 1
        # Version 2 with the same schema: allowed now, and not representable in 0002.
        conn.execute(text(
            "INSERT INTO strategies (name, version, schema_hash, param_schema, is_current) "
            "VALUES ('momentum', 2, decode(repeat('ab', 32), 'hex'), '{}', false)"))
    with pytest.raises(RuntimeError, match="several versions with the same parameter schema"):
        migrate(empty_engine, "downgrade", "0002")
    with empty_engine.begin() as conn:
        conn.execute(text("DELETE FROM strategies WHERE version = 2"))
    migrate(empty_engine, "downgrade", "0002")
    with empty_engine.connect() as conn:
        assert "version" not in {c["name"] for c in inspect(conn).get_columns("strategies")}


def _registry_row(**overrides) -> dict:
    sha = "decode(repeat('ab', 32), 'hex')"
    values = {
        "version": "1", "model_type": "'lightgbm'", "target": "'regime'",
        "feature_set": "'base'", "features": "ARRAY['ret_1m']", "classes": "ARRAY[0.0, 1.0]",
        "hyperparameters": "'{}'", "training_config": "'{}'", "seed": "0",
        "price_series_id": "1", "train_start": "'2015-01-30'", "train_end": "'2020-11-30'",
        "train_label_end": "'2020-12-31'", "n_train": "60", "oos_accuracy": "0.8",
        "oos_months": "24", "baseline_name": "'rules_regime_lookup'",
        "baseline_accuracy": "0.82", "beats_baseline": "false", "metrics": "'{}'",
        "serving_as_of": "'2020-12-31'", "serving_features": "'{}'",
        "serving_features_sha256": sha, "artifact_format": "'qbt-ml-model/1'",
        "artifact": "'\\x00'::bytea", "artifact_sha256": sha, "library_versions": "'{}'",
    } | overrides
    return values


def _insert_registry_row(conn, **overrides) -> None:
    values = _registry_row(**overrides)
    conn.execute(text(f"INSERT INTO model_registry ({', '.join(values)}) "
                      f"VALUES ({', '.join(values.values())})"))


def test_0004_round_trip_leaves_earlier_tables_alone(empty_engine):
    migrate(empty_engine, "upgrade", "0003")
    with empty_engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO price_series (ticker, fetch_start, fetch_end, first_date, last_date, "
            "n_bars, content_sha256) VALUES ('SPY', '2008-01-01', '2021-01-01', '2008-01-02', "
            "'2020-12-31', 3000, decode(repeat('cd', 32), 'hex'))"))
    before = set(inspect(empty_engine).get_table_names())

    migrate(empty_engine, "upgrade", "0004")
    assert set(inspect(empty_engine).get_table_names()) == before | {"model_registry",
                                                                     "predictions"}
    with empty_engine.begin() as conn:
        _insert_registry_row(conn)
        conn.execute(text(
            "INSERT INTO predictions (model_version, as_of, features_sha256, prediction, "
            "probabilities) VALUES (1, '2020-12-31', decode(repeat('ab', 32), 'hex'), 0, "
            "ARRAY[0.6, 0.4])"))
    # The constraints that keep a registry row honest.
    for bad in ({"version": "2", "oos_accuracy": "1.5"},
                {"version": "2", "baseline_name": "NULL"},
                {"version": "2", "beats_baseline": "NULL"},
                {"version": "2", "target": "'volatility'"},
                {"version": "2", "artifact_sha256": "'\\x00'::bytea"},
                {"version": "2", "serving_as_of": "'2020-11-30'"},
                {"version": "2", "price_series_id": "99"},
                {"version": "2"}):          # same snapshot and artifact hash as version 1
        with pytest.raises(sa_exc.IntegrityError), empty_engine.begin() as conn:
            _insert_registry_row(conn, **bad)
    with pytest.raises(sa_exc.IntegrityError), empty_engine.begin() as conn:
        conn.execute(text("DELETE FROM model_registry"))   # predictions point at it

    migrate(empty_engine, "downgrade", "0003")
    assert set(inspect(empty_engine).get_table_names()) == before
    with empty_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM price_series")).scalar_one() == 1
    migrate(empty_engine, "upgrade", "0004")
    assert set(inspect(empty_engine).get_table_names()) >= TABLES


def test_migrations_match_metadata(migrated_engine):
    assert _diff(migrated_engine, metadata) == []


def _schema_objects(engine) -> dict[str, set]:
    """Every constraint and index on the service's tables, with its full definition."""
    names = tuple(metadata.tables)
    with engine.connect() as conn:
        constraints = conn.execute(text(
            "SELECT conrelid::regclass::text, conname, pg_get_constraintdef(oid) "
            "FROM pg_constraint WHERE conrelid::regclass::text = ANY(:t)"
        ), {"t": list(names)}).all()
        indexes = conn.execute(text(
            "SELECT tablename, indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename = ANY(:t)"
        ), {"t": list(names)}).all()
    return {"constraints": set(constraints), "indexes": set(indexes)}


def test_migrated_schema_equals_metadata_schema(migrated_engine, empty_engine):
    # alembic check can't see CHECK constraints, their names, or partial-index
    # predicates. Building the same schema straight from qbt_service.db and
    # comparing Postgres's own definitions covers all of them.
    metadata.create_all(empty_engine)
    assert _schema_objects(migrated_engine) == _schema_objects(empty_engine)


def test_drift_would_be_detected(migrated_engine):
    drifted = MetaData()
    for table in metadata.sorted_tables:
        table.to_metadata(drifted)
    drifted.tables["jobs"].append_column(Column("unmigrated", Text))

    diff = _diff(migrated_engine, drifted)
    assert [(op[0], op[3].name) for op in diff] == [("add_column", "unmigrated")]


# ---- env.py's own connection path (what `alembic` on the command line runs) ----

def _bare_config(**kwargs) -> Config:
    # No ini file, so env.py leaves logging alone (fileConfig would replace
    # pytest's handlers); the script location is the one alembic.ini names.
    cfg = Config(**kwargs)
    cfg.set_main_option("script_location", alembic_config().get_main_option("script_location"))
    return cfg


def test_env_connects_with_database_url(empty_engine, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", empty_engine.url.render_as_string(hide_password=False))
    command.upgrade(_bare_config(), "head")
    assert set(inspect(empty_engine).get_table_names()) >= TABLES


def test_env_offline_sql(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user@localhost/unused")
    buffer = io.StringIO()
    command.upgrade(_bare_config(output_buffer=buffer), "head", sql=True)
    sql = buffer.getvalue()
    assert "CREATE TYPE job_status AS ENUM" in sql
    assert "ALTER TABLE jobs SET (fillfactor = 90" in sql


def test_env_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        command.upgrade(_bare_config(), "head")
