"""
Alembic environment. Connects to DATABASE_URL, or, when the caller has
put an open connection in config.attributes["connection"] (the tests do),
runs on that connection inside the caller's transaction.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, create_engine, pool

from qbt_service.db import metadata

config = context.config


def _url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


def _configure(**kwargs) -> None:
    # compare_type: let `alembic check` catch column type drift too.
    context.configure(target_metadata=metadata, compare_type=True, **kwargs)


def _run_on(connection: Connection) -> None:
    _configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


def run_offline() -> None:
    """`alembic upgrade head --sql`: print the SQL instead of running it."""
    _configure(url=_url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run_on(connection)
        return
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as conn:
        _run_on(conn)


# Only the CLI configures logging: reconfiguring it inside a test run
# would replace pytest's own handlers.
if config.config_file_name is not None and "connection" not in config.attributes:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

if context.is_offline_mode():
    run_offline()
else:
    run_online()
