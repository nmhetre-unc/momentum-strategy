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


def synthetic_frame(start: str = "2018-01-01", periods: int = 600, seed: int = 0):
    """A price frame shaped like fetch_ohlcv()'s cached frame."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=periods)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, periods))
    return pd.DataFrame({
        "Close": close,
        "High": close * (1 + rng.uniform(0, 0.01, periods)),
        "Low": close * (1 - rng.uniform(0, 0.01, periods)),
        "Open": close * (1 + rng.normal(0, 0.004, periods)),
        "Volume": rng.integers(10**6, 10**8, periods),
    }, index=pd.DatetimeIndex(days.to_numpy(), name="Date").as_unit("us"))


def store_synthetic_series(conn, ticker: str = "SPY", **frame_kwargs) -> int:
    """Stores synthetic_frame() as a snapshot fetched for exactly its own span."""
    from datetime import timedelta

    from qbt_service.prices import store_snapshot

    frame = synthetic_frame(**frame_kwargs)
    first, last = frame.index[0].date(), frame.index[-1].date()
    return store_snapshot(conn, ticker, first, last + timedelta(days=1), frame).id
