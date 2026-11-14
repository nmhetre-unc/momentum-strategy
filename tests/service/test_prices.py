"""
Price snapshots: a frame stored and loaded back is exactly the frame
qbt.data.fetch_ohlcv() returns, from a fresh download and from the CSV
cache, and backtests on either give bit-identical metrics. Also covers
snapshot dedup, coverage lookup, slicing, input checks, and the ingest
command end to end. yfinance is replaced with a fake that returns a
frame shaped like its real output, so no test touches the network.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

import qbt.data
from qbt.data import download_ohlcv, fetch_ohlcv
from qbt.runner import metrics_equal, run
from qbt_service import ingest
from qbt_service.db import price_bars, price_series
from qbt_service.prices import (
    check_frame,
    content_hash,
    find_series,
    load_frame,
    store_snapshot,
)

pytestmark = pytest.mark.db

START, END = date(2019, 1, 1), date(2021, 1, 1)


def yahoo_frame(ticker: str, start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """What yf.download(..., auto_adjust=True) returns: (Price, Ticker)
    MultiIndex columns, a datetime64[s] "Date" index, full-precision floats."""
    days = pd.bdate_range(start, end, inclusive="left")
    rng = np.random.default_rng(seed)
    close = 250 * np.cumprod(1 + rng.normal(0.0004, 0.012, len(days)))
    data = {
        "Close": close,
        "High": close * (1 + rng.uniform(0, 0.01, len(days))),
        "Low": close * (1 - rng.uniform(0, 0.01, len(days))),
        "Open": close * (1 + rng.normal(0, 0.004, len(days))),
        "Volume": rng.integers(10**7, 3 * 10**8, len(days)),
    }
    # freq=None, like yfinance's index (bdate_range would carry a BusinessDay freq).
    index = pd.DatetimeIndex(days.to_numpy(), name="Date", freq=None).as_unit("s")
    df = pd.DataFrame(data, index=index)
    df.columns = pd.MultiIndex.from_product([df.columns, [ticker]], names=["Price", "Ticker"])
    return df


@pytest.fixture
def fake_yahoo(monkeypatch, tmp_path):
    """Patches yfinance (and points the CSV cache at a temp dir); returns the
    list of download calls, so a test can see whether the cache was used."""
    calls = []

    def download(ticker, start, end, **kwargs):
        calls.append((ticker, start, end))
        return yahoo_frame(ticker, start, end, seed=len(ticker))

    monkeypatch.setattr(qbt.data.yf, "download", download)
    monkeypatch.setattr(qbt.data, "CACHE_DIR", str(tmp_path / "cache"))
    return calls


def cached_frame(ticker="SPY", start=START, end=END) -> pd.DataFrame:
    """fetch_ohlcv()'s frame as read back from its CSV cache (the second call)."""
    fetch_ohlcv(ticker, start.isoformat(), end.isoformat())
    return fetch_ohlcv(ticker, start.isoformat(), end.isoformat())


def store(conn, ticker="SPY", start=START, end=END, df=None):
    df = download_ohlcv(ticker, start.isoformat(), end.isoformat()) if df is None else df
    return store_snapshot(conn, ticker, start, end, df)


# ---- the frame is exactly fetch_ohlcv's ----

def test_loaded_frame_equals_fetch_ohlcv(conn, fake_yahoo):
    fresh = fetch_ohlcv("SPY", START.isoformat(), END.isoformat())   # downloads, writes CSV
    cached = fetch_ohlcv("SPY", START.isoformat(), END.isoformat())  # reads the CSV
    assert len(fake_yahoo) == 1

    loaded = load_frame(conn, store(conn).id)

    # The cached frame is what the CLI and dashboard normally see: identical
    # in every respect, index unit and dtypes included.
    pd.testing.assert_frame_equal(loaded, cached, check_exact=True)
    # A fresh download differs from its own cache read only in metadata
    # (index unit "s" vs "us", a "Price" columns name); the values are identical.
    assert fresh.index.unit == "s" and fresh.columns.name == "Price"
    normalized = fresh.set_axis(fresh.index.as_unit("us"), axis=0).rename_axis(columns=None)
    pd.testing.assert_frame_equal(loaded, normalized, check_exact=True)


@pytest.mark.parametrize("strategy", ["sma_crossover", "momentum", "mean_reversion",
                                      "ml_direction"])
def test_backtests_on_loaded_prices_are_bit_identical(conn, fake_yahoo, strategy):
    fresh = fetch_ohlcv("SPY", START.isoformat(), END.isoformat())
    cached = fetch_ohlcv("SPY", START.isoformat(), END.isoformat())
    loaded = load_frame(conn, store(conn).id)

    expected = run(loaded, strategy, {}, n_boot=200, seed=4).metrics
    for frame in (fresh, cached):
        assert metrics_equal(run(frame, strategy, {}, n_boot=200, seed=4).metrics, expected)


def test_slice_equals_the_same_slice_of_fetch_ohlcv(conn, fake_yahoo):
    cached = cached_frame()
    series_id = store(conn).id
    # [start, end): a bar on `end` is excluded, as in yfinance.
    loaded = load_frame(conn, series_id, date(2019, 6, 3), date(2020, 6, 1))
    expected = cached.loc["2019-06-03":"2020-05-31"]
    pd.testing.assert_frame_equal(loaded, expected, check_exact=True)
    assert loaded.index[0] == pd.Timestamp("2019-06-03")
    assert loaded.index[-1] == pd.Timestamp("2020-05-29")


def test_nan_prices_round_trip(conn, fake_yahoo):
    df = download_ohlcv("SPY", START.isoformat(), END.isoformat())
    df.iloc[5, df.columns.get_loc("Open")] = np.nan  # yfinance does emit these
    loaded = load_frame(conn, store(conn, df=df).id)
    assert np.isnan(loaded["Open"].iloc[5])
    pd.testing.assert_frame_equal(loaded, df.set_axis(df.index.as_unit("us"), axis=0)
                                  .rename_axis(columns=None), check_exact=True)


def test_empty_range_is_an_error(conn, fake_yahoo):
    series_id = store(conn).id
    with pytest.raises(ValueError, match="no bars"):
        load_frame(conn, series_id, date(2030, 1, 1), date(2031, 1, 1))


# ---- snapshots are immutable and deduplicated ----

def _count(conn, table, **where):
    query = select(func.count()).select_from(table)
    for column, value in where.items():
        query = query.where(table.c[column] == value)
    return conn.execute(query).scalar_one()


def test_identical_snapshot_is_reused(conn, fake_yahoo):
    first = store(conn)
    second = store(conn)
    assert first.created and not second.created
    assert second.id == first.id
    assert _count(conn, price_bars, series_id=first.id) == first.n_bars


def test_changed_prices_make_a_new_series(conn, fake_yahoo):
    df = download_ohlcv("SPY", START.isoformat(), END.isoformat())
    old = store(conn, df=df)
    revised = df.copy()
    revised.iloc[0, revised.columns.get_loc("Close")] = np.nextafter(df["Close"].iloc[0], 0)
    new = store(conn, df=revised)
    assert new.created and new.id != old.id
    # Old jobs keep their prices; new jobs get the newest snapshot.
    assert load_frame(conn, old.id)["Close"].iloc[0] == df["Close"].iloc[0]
    assert find_series(conn, "SPY", START, END) == new.id


def test_same_bars_for_a_different_range_are_a_different_snapshot(conn, fake_yahoo):
    df = download_ohlcv("SPY", START.isoformat(), END.isoformat())
    a = store_snapshot(conn, "SPY", START, END, df)
    b = store_snapshot(conn, "SPY", date(2018, 12, 31), END, df)  # 2018-12-31 has no bar here
    assert a.id != b.id


def test_content_hash_sees_one_ulp_and_ignores_nan_payload(fake_yahoo):
    df = download_ohlcv("SPY", START.isoformat(), END.isoformat())
    base = content_hash("SPY", "yfinance", START, END, df)
    assert content_hash("SPY", "yfinance", START, END, df.copy()) == base

    nudged = df.copy()
    nudged.iloc[10, nudged.columns.get_loc("Low")] = np.nextafter(df["Low"].iloc[10], np.inf)
    assert content_hash("SPY", "yfinance", START, END, nudged) != base

    nan_a, nan_b = df.copy(), df.copy()
    nan_a.iloc[3, 0] = np.nan
    nan_b.iloc[3, 0] = np.frombuffer(np.uint64(0x7FF8000000000BAD).tobytes(), dtype="<f8")[0]
    assert np.isnan(nan_b.iloc[3, 0])
    assert content_hash("SPY", "yfinance", START, END, nan_a) == content_hash(
        "SPY", "yfinance", START, END, nan_b)


# ---- coverage lookup ----

def test_find_series_coverage(conn, fake_yahoo):
    series_id = store(conn).id
    assert find_series(conn, "spy", START, END) == series_id      # tickers are normalized
    assert find_series(conn, "SPY", date(2019, 6, 1), date(2020, 6, 1)) == series_id
    assert find_series(conn, "SPY", date(2018, 12, 31), END) is None
    assert find_series(conn, "SPY", START, date(2021, 1, 2)) is None
    assert find_series(conn, "QQQ", START, END) is None


def test_find_series_prefers_the_newest_covering_snapshot(conn, fake_yahoo):
    wide = store(conn, start=date(2018, 1, 1), end=date(2022, 1, 1)).id
    narrow = store(conn).id
    assert find_series(conn, "SPY", START, END) == narrow
    assert find_series(conn, "SPY", date(2018, 6, 1), END) == wide


# ---- input checks ----

def _good_frame():
    return download_ohlcv("SPY", START.isoformat(), END.isoformat())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda df: df.drop(columns="Volume"), "expected columns"),
        (lambda df: df.assign(Extra=1.0), "expected columns"),
        (lambda df: df.tz_localize("UTC"), "tz-naive"),
        (lambda df: df.set_axis(df.index + pd.Timedelta(hours=16), axis=0), "midnight"),
        (lambda df: df.astype({"Volume": float}), "Volume must be an integer"),
        (lambda df: df.astype({"High": np.float32}), "High must be float64"),
        (lambda df: df.assign(Close=df["Close"].where(df.index != df.index[2])), "missing"),
    ],
)
def test_frames_that_would_lose_information_are_rejected(fake_yahoo, mutate, message):
    with pytest.raises(ValueError, match=message):
        check_frame(mutate(_good_frame()))


def test_store_rejects_bad_ranges(conn, fake_yahoo):
    df = _good_frame()
    with pytest.raises(ValueError, match="must be before"):
        store_snapshot(conn, "SPY", END, START, df)
    with pytest.raises(ValueError, match="outside the fetch range"):
        store_snapshot(conn, "SPY", date(2019, 6, 1), END, df)
    with pytest.raises(ValueError, match="empty"):
        store_snapshot(conn, "  ", START, END, df)


# ---- the ingest command, end to end ----

def test_ingest_command(committed_engine, fake_yahoo, capsys):
    argv = ["spy", "QQQ", "--start", "2019-01-01", "--end", "2021-01-01"]
    assert ingest.main(argv) == 0
    out = capsys.readouterr().out
    assert "SPY: stored new series" in out and "QQQ: stored new series" in out

    assert ingest.main(argv) == 0
    assert capsys.readouterr().out.count("unchanged, already stored") == 2

    with committed_engine.connect() as conn:
        assert _count(conn, price_series) == 2
        series_id = find_series(conn, "SPY", START, END)
        pd.testing.assert_frame_equal(load_frame(conn, series_id), cached_frame(),
                                      check_exact=True)


def test_ingest_reports_failures(committed_engine, monkeypatch, capsys):
    def empty(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(qbt.data.yf, "download", empty)
    assert ingest.main(["NOPE", "--start", "2019-01-01", "--end", "2021-01-01"]) == 1
    assert "NOPE: FAILED: No data returned" in capsys.readouterr().err


def test_ingest_rejects_reversed_dates(capsys):
    with pytest.raises(SystemExit):
        ingest.main(["SPY", "--start", "2021-01-01", "--end", "2019-01-01"])
    assert "--start must be before --end" in capsys.readouterr().err
