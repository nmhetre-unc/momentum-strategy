"""
Fetches historical OHLCV price data for a ticker, with local CSV caching
so repeated runs don't keep hitting the Yahoo Finance API.
"""

import os
import pandas as pd
import yfinance as yf

CACHE_DIR = "data_cache"


def fetch_ohlcv(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame of daily OHLCV data for `ticker` between `start`
    and `end` (both 'YYYY-MM-DD' strings). Caches to disk so re-running
    the same query doesn't re-download.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.csv")

    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance sometimes returns a MultiIndex column header for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")

    df.to_csv(cache_path)
    return df
