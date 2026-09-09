"""
Assembles a multi-ticker price panel (dates x tickers) from qbt.data's
per-ticker OHLCV cache, for cross-sectional / universe-level research.
"""

import hashlib
import os

import pandas as pd

from qbt.data import CACHE_DIR, fetch_ohlcv

# Current S&P 100 (OEX) membership, hardcoded as of this writing -- NOT a
# point-in-time roster. This list reflects which companies belong to the
# index TODAY; it says nothing about who belonged on any past date. Every
# ticker here survived (through growth, not being acquired, not going
# bankrupt) to still be a large, liquid US company. Backtesting a strategy
# against this universe over history therefore carries survivorship bias:
# the panel is silent about every company that was once index-sized and
# isn't anymore. Verify against an authoritative source (S&P Dow Jones
# Indices) before relying on this for anything beyond illustration, and
# expect it to drift out of date -- index membership changes several
# times a year.
SP100_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS", "CVX",
    "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "F", "FDX", "GD", "GE",
    "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU",
    "ISRG", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM",
    "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "T", "TGT", "TMO", "TMUS",
    "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
    "XOM",
]

# A gap this short is plausibly a data hiccup (a missed vendor update, a
# holiday absent from the business-day calendar); forward-filling it is a
# defensible approximation. A longer gap is a real absence -- a halt, a
# delisting, a ticker that hadn't IPO'd yet -- and filling it would
# fabricate a flat (zero) return the strategy layer would treat as real.
MAX_FFILL_DAYS = 5


def _panel_cache_path(tickers: list, start: str, end: str) -> str:
    """
    One cache file per (ticker set, date range). The ticker list itself
    is too long for a filename, so it's hashed; sorting first means the
    same set of tickers in a different order still hits the same cache.
    """
    signature = hashlib.sha1(",".join(sorted(tickers)).encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"universe_{signature}_{start}_{end}.csv")


def fetch_universe(
    tickers: list = SP100_TICKERS,
    start: str = "2008-01-01",
    end: str = "2025-01-01",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Daily adjusted closes for `tickers`, aligned to one common business-day
    index spanning `start` to `end` and cached as a single assembled panel
    so a repeat call doesn't re-download or re-assemble anything.

    A ticker with a shorter real history (a later IPO, a later addition to
    whatever list produced `tickers`) is kept as a column of NaN before its
    first real observation rather than dropped -- dropping it would throw
    away exactly the information that makes the panel survivorship-biased
    in the first place, and silently shrink the universe in a way that's
    easy to miss. It is the ranking/selection logic downstream, not this
    function, that must check for enough lookback before using a ticker on
    a given day; `first_valid_index()` per column gives it that date.

    Gaps up to MAX_FFILL_DAYS days are forward-filled; longer gaps
    (including the leading NaN before a ticker's first observation, which
    is not a "gap" and is never filled) are left as NaN.
    """
    cache_path = _panel_cache_path(tickers, start, end)
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    # A single ticker's fetch failing (a vendor-side symbol glitch, a
    # genuine delisting) must not abort the other ~100. Keep it as an
    # all-NaN column rather than dropping it, so the panel's ticker set
    # stays exactly `tickers` regardless of which ones the vendor served
    # today; a column with no valid observations excludes itself from
    # ranking under the same "insufficient lookback" rule as any other.
    closes = {}
    failed = []
    for ticker in tickers:
        try:
            closes[ticker] = fetch_ohlcv(ticker, start, end, use_cache=use_cache)["Close"]
        except ValueError as exc:
            failed.append(ticker)
            print(f"universe: skipping {ticker}, fetch failed: {exc}")

    if failed:
        print(f"universe: {len(failed)} of {len(tickers)} tickers failed to fetch: {failed}")

    panel = pd.DataFrame(closes).reindex(columns=tickers)

    # Reindex to the full requested business-day span, not just the union
    # of days any ticker happened to trade, so a ticker's pre-IPO period
    # is explicit NaN in a row that exists, rather than the row itself
    # being absent from the panel.
    full_index = pd.bdate_range(start, end)
    panel = panel.reindex(full_index)
    panel = panel.ffill(limit=MAX_FFILL_DAYS)

    os.makedirs(CACHE_DIR, exist_ok=True)
    panel.to_csv(cache_path)
    return panel
