"""Minimal rows for service tests that need a strategy or a price series to point at."""

import hashlib
import uuid
from datetime import date

from qbt_service.db import price_series, strategies


def sha(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def add_series(conn, **overrides) -> int:
    values = {
        "ticker": "SPY", "fetch_start": date(2020, 1, 1), "fetch_end": date(2021, 1, 1),
        "first_date": date(2020, 1, 1), "last_date": date(2020, 12, 31),
        "n_bars": 252, "content_sha256": sha(str(uuid.uuid4())),
    } | overrides
    return conn.execute(
        price_series.insert().values(values).returning(price_series.c.id)
    ).scalar_one()


def add_strategy(conn, name: str = "momentum", **overrides) -> int:
    values = {
        "name": name, "schema_hash": sha(str(uuid.uuid4())), "param_schema": {"type": "object"},
    } | overrides
    return conn.execute(strategies.insert().values(values).returning(strategies.c.id)).scalar_one()
