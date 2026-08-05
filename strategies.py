"""
Each strategy function takes a price DataFrame (with a 'Close' column)
and returns a pandas Series of positions: 1 = long, 0 = flat.

Keeping this interface consistent is what makes it easy to add new
strategies and plug any of them into the same backtest engine.
"""

import numpy as np
import pandas as pd


def sma_crossover(df: pd.DataFrame, short_window: int = 50, long_window: int = 200) -> pd.Series:
    """Long when the short-term SMA is above the long-term SMA, flat otherwise."""
    short_sma = df["Close"].rolling(short_window).mean()
    long_sma = df["Close"].rolling(long_window).mean()
    signal = (short_sma > long_sma).astype(int)
    return signal


def momentum(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.0) -> pd.Series:
    """Long when the trailing `lookback`-day return exceeds `threshold`."""
    trailing_return = df["Close"].pct_change(lookback)
    signal = (trailing_return > threshold).astype(int)
    return signal


def mean_reversion_rsi(
    df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70
) -> pd.Series:
    """
    Classic RSI mean-reversion: go long when RSI drops below `oversold`
    (asset is "cheap"), exit when RSI rises above `overbought`. Holds the
    previous position while RSI sits between the two thresholds.
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    signal = pd.Series(index=df.index, dtype=float)
    signal[rsi < oversold] = 1
    signal[rsi > overbought] = 0
    signal = signal.ffill().fillna(0)
    return signal.astype(int)


STRATEGIES = {
    "sma_crossover": sma_crossover,
    "momentum": momentum,
    "mean_reversion": mean_reversion_rsi,
}
