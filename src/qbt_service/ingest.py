"""
Downloads daily prices from Yahoo Finance and stores them as immutable
snapshots. Workers never download; a backtest can only run over prices
ingested here first.

    python -m qbt_service.ingest SPY QQQ --start 2008-01-01 --end 2025-01-01

`--end` is exclusive, as in yfinance and the CLI. Ingesting again when
nothing changed reuses the existing snapshot; after a dividend or split
(adjusted closes are rewritten) it stores a new one, and new jobs use it.
"""

import argparse
import sys
from datetime import date

from qbt.data import download_ohlcv
from qbt_service.db import make_engine
from qbt_service.prices import normalize_ticker, store_snapshot
from qbt_service.settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qbt_service.ingest", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, required=True,
                        help="YYYY-MM-DD, exclusive")
    args = parser.parse_args(argv)
    if args.start >= args.end:
        parser.error("--start must be before --end")

    engine = make_engine(Settings.from_env().database_url)
    failures = 0
    try:
        for raw in args.tickers:
            ticker = normalize_ticker(raw)
            try:
                df = download_ohlcv(ticker, args.start.isoformat(), args.end.isoformat())
                with engine.begin() as conn:
                    snap = store_snapshot(conn, ticker, args.start, args.end, df)
            except ValueError as exc:
                failures += 1
                print(f"{ticker}: FAILED: {exc}", file=sys.stderr)
                continue
            verb = "stored new series" if snap.created else "unchanged, already stored as series"
            print(f"{ticker}: {verb} {snap.id} "
                  f"({snap.n_bars} bars, {snap.first_date} to {snap.last_date})")
    finally:
        engine.dispose()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
