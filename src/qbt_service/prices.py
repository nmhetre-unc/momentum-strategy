"""
Price snapshots in PostgreSQL: store one downloaded frame as an immutable
series, find the newest series that covers a backtest's date range, and
load it back as exactly the frame qbt.data.fetch_ohlcv() returns.

"Exactly" is the point. A job's result must not depend on whether its
prices came from the database or the CSV cache, and a rerun must not
depend on when it ran. So prices are stored as float64 (no rounding),
snapshots are never updated, and a job pins the series it was submitted
against. Re-downloading after a dividend changes every adjusted close,
which gives a new content hash and therefore a new series.

Dates follow yfinance: a fetch of [start, end) excludes `end`.
"""

import hashlib
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Connection, select
from sqlalchemy.dialects.postgresql import insert

from qbt.runner import validate_prices
from qbt_service.db import price_bars, price_series

SOURCE = "yfinance"
# Column order of fetch_ohlcv()'s frame (yfinance's own order).
COLUMNS = ("Close", "High", "Low", "Open", "Volume")
FLOAT_COLUMNS = ("Close", "High", "Low", "Open")
HASH_VERSION = b"qbt-price-snapshot-v1\0"


@dataclass(frozen=True)
class Snapshot:
    id: int
    created: bool          # False: an identical snapshot was already stored
    n_bars: int
    first_date: date
    last_date: date


def normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("ticker must not be empty")
    return cleaned


def check_frame(df: pd.DataFrame) -> None:
    """Raises ValueError unless `df` can be stored without losing anything."""
    if set(df.columns) != set(COLUMNS):
        raise ValueError(f"expected columns {list(COLUMNS)}, got {list(df.columns)}")
    validate_prices(df)  # Close present and positive, dates unique and increasing
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None or not (index == index.normalize()).all():
        raise ValueError("daily bars need tz-naive midnight timestamps (one bar per date)")
    for column in FLOAT_COLUMNS:
        if df[column].dtype != np.float64:
            raise ValueError(f"{column} must be float64, got {df[column].dtype}")
    if not pd.api.types.is_integer_dtype(df["Volume"]):
        raise ValueError(f"Volume must be an integer column, got {df['Volume'].dtype}")


def content_hash(ticker: str, source: str, fetch_start: date, fetch_end: date,
                 df: pd.DataFrame) -> bytes:
    """
    SHA-256 over the fetch range and every bar's exact bits. Any change to
    any price, even one ULP, gives a different hash. NaNs are canonicalized
    first, since NaN has many bit patterns that all mean "missing".
    """
    h = hashlib.sha256(HASH_VERSION)
    for part in (ticker, source, fetch_start.isoformat(), fetch_end.isoformat()):
        h.update(part.encode() + b"\0")
    days = pd.DatetimeIndex(df.index).to_numpy().astype("datetime64[D]").astype("<i8")
    h.update(days.tobytes())
    for column in FLOAT_COLUMNS:
        values = df[column].to_numpy(dtype="<f8", copy=True)
        values[np.isnan(values)] = np.nan
        h.update(values.tobytes())
    h.update(df["Volume"].to_numpy(dtype="<i8").tobytes())
    return h.digest()


def store_snapshot(conn: Connection, ticker: str, fetch_start: date, fetch_end: date,
                   df: pd.DataFrame, source: str = SOURCE) -> Snapshot:
    """
    Stores `df` (as returned by qbt.data.download_ohlcv for the same range)
    as a new series, or returns the existing series if an identical one is
    already stored. Runs in the caller's transaction. Two concurrent calls
    with the same content are safe: the second waits on the unique index,
    then finds the first one's series.
    """
    ticker = normalize_ticker(ticker)
    if fetch_start >= fetch_end:
        raise ValueError(f"fetch_start {fetch_start} must be before fetch_end {fetch_end}")
    check_frame(df)
    index = pd.DatetimeIndex(df.index)
    first, last = index[0].date(), index[-1].date()
    if first < fetch_start or last >= fetch_end:
        raise ValueError(
            f"bars run {first}..{last}, outside the fetch range [{fetch_start}, {fetch_end})"
        )
    digest = content_hash(ticker, source, fetch_start, fetch_end, df)

    series_id = conn.execute(
        insert(price_series)
        .values(ticker=ticker, source=source, fetch_start=fetch_start, fetch_end=fetch_end,
                first_date=first, last_date=last, n_bars=len(df), content_sha256=digest)
        .on_conflict_do_nothing(index_elements=["ticker", "source", "content_sha256"])
        .returning(price_series.c.id)
    ).scalar_one_or_none()
    created = series_id is not None
    if series_id is None:
        series_id = conn.execute(
            select(price_series.c.id).where(
                price_series.c.ticker == ticker, price_series.c.source == source,
                price_series.c.content_sha256 == digest,
            )
        ).scalar_one()
    else:
        conn.execute(price_bars.insert(), [
            {"series_id": series_id, "date": day.date(), "open": float(o), "high": float(h),
             "low": float(lo), "close": float(c), "volume": int(v)}
            for day, c, h, lo, o, v in zip(
                index, df["Close"], df["High"], df["Low"], df["Open"], df["Volume"], strict=True,
            )
        ])
    return Snapshot(id=series_id, created=created, n_bars=len(df), first_date=first,
                    last_date=last)


def find_series(conn: Connection, ticker: str, start: date, end: date,
                source: str = SOURCE) -> int | None:
    """The newest series whose fetch range covers [start, end), or None."""
    return conn.execute(
        select(price_series.c.id)
        .where(price_series.c.ticker == normalize_ticker(ticker),
               price_series.c.source == source,
               price_series.c.fetch_start <= start,
               price_series.c.fetch_end >= end)
        .order_by(price_series.c.fetched_at.desc(), price_series.c.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def load_frame(conn: Connection, series_id: int, start: date | None = None,
               end: date | None = None) -> pd.DataFrame:
    """
    The bars of one series in [start, end), shaped like fetch_ohlcv()'s
    cached frame: a DatetimeIndex named "Date" (microsecond unit, as
    read_csv gives), columns Close/High/Low/Open as float64, Volume as int64.
    """
    query = (
        select(price_bars.c.date, price_bars.c.close, price_bars.c.high, price_bars.c.low,
               price_bars.c.open, price_bars.c.volume)
        .where(price_bars.c.series_id == series_id)
        .order_by(price_bars.c.date)
    )
    if start is not None:
        query = query.where(price_bars.c.date >= start)
    if end is not None:
        query = query.where(price_bars.c.date < end)
    rows = conn.execute(query).all()
    if not rows:
        raise ValueError(f"series {series_id} has no bars in [{start}, {end})")

    dates, close, high, low, open_, volume = zip(*rows, strict=True)
    index = pd.DatetimeIndex(np.array(dates, dtype="datetime64[D]"), name="Date").as_unit("us")
    return pd.DataFrame({
        "Close": np.array(close, dtype=np.float64),
        "High": np.array(high, dtype=np.float64),
        "Low": np.array(low, dtype=np.float64),
        "Open": np.array(open_, dtype=np.float64),
        "Volume": np.array(volume, dtype=np.int64),
    }, index=index)
