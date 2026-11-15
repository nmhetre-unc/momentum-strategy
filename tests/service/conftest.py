"""
Database fixtures for the service tests.

Each test session gets its own throwaway database, created next to the
one DATABASE_URL points at and dropped afterwards, so tests never touch
data you care about. Locally: `docker compose up -d db` and
`export DATABASE_URL=postgresql+psycopg://qbt:qbt@localhost:5432/qbt`.

Without DATABASE_URL these tests skip, except where QBT_REQUIRE_DB=1
(CI), where they fail instead -- the same rule as QBT_REQUIRE_CPP.
"""

import contextlib
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, make_url, text
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parents[2]


def base_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        if os.environ.get("QBT_REQUIRE_DB") == "1":
            pytest.fail("QBT_REQUIRE_DB=1 but DATABASE_URL is not set")
        pytest.skip("DATABASE_URL is not set (start one with `docker compose up -d db`)")
    return url


@contextlib.contextmanager
def temporary_database(url: str) -> Iterator[Engine]:
    """An engine on a new, empty database, dropped on exit."""
    base = make_url(url)
    name = f"qbt_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(base, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    engine = create_engine(base.set(database=name), poolclass=NullPool)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def alembic_config() -> Config:
    """The repo's real alembic.ini, so the tests exercise what the CLI runs."""
    return Config(str(ROOT / "alembic.ini"))


def migrate(engine: Engine, direction: str = "upgrade", revision: str = "head") -> None:
    cfg = alembic_config()
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        getattr(command, direction)(cfg, revision)


@pytest.fixture(scope="session")
def migrated_engine() -> Iterator[Engine]:
    with temporary_database(base_url()) as engine:
        migrate(engine)
        yield engine


@pytest.fixture
def empty_engine() -> Iterator[Engine]:
    with temporary_database(base_url()) as engine:
        yield engine


@pytest.fixture
def committed_engine(empty_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Engine:
    """
    A migrated database of this test's own, for code that commits (the
    ingest command, the API). DATABASE_URL points at it for the test.
    """
    migrate(empty_engine)
    monkeypatch.setenv("DATABASE_URL", empty_engine.url.render_as_string(hide_password=False))
    return empty_engine


@pytest.fixture
def conn(migrated_engine: Engine) -> Iterator[Connection]:
    """A connection inside a transaction that is rolled back after the test."""
    with migrated_engine.connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            transaction.rollback()
