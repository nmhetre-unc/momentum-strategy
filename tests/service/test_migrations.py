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

from qbt_service.db import metadata

from .conftest import alembic_config, migrate

pytestmark = pytest.mark.db

TABLES = {"price_series", "price_bars", "strategies", "jobs", "results", "metrics"}


def _diff(engine, target: MetaData) -> list:
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        return compare_metadata(context, target)


def test_single_head():
    assert ScriptDirectory.from_config(alembic_config()).get_heads() == ["0002"]


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


def test_migrations_match_metadata(migrated_engine):
    assert _diff(migrated_engine, metadata) == []


def _schema_objects(engine) -> dict[str, set]:
    """Every constraint and index on the service's tables, with its full definition."""
    names = tuple(metadata.tables)
    with engine.connect() as conn:
        constraints = conn.execute(text(
            "SELECT conrelid::regclass::text, conname, pg_get_constraintdef(oid) "
            "FROM pg_constraint WHERE conrelid::regclass::text = ANY(:t)"
        ), {"t": list(names)}).tuples().all()
        indexes = conn.execute(text(
            "SELECT tablename, indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename = ANY(:t)"
        ), {"t": list(names)}).tuples().all()
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
