"""
Turns raw OHLCV data into a table of predictive features, plus the
next-day direction label the model is trained to predict.
"""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Every feature on row t is built ONLY from information available up
    to and including day t's close -- no future data leaks in here.
    (The label, in build_labels() below, is a separate matter -- see
    the docstring there.)
    """
    features = pd.DataFrame(index=df.index)
    close = df["Close"]

    # Lagged returns: how much has price moved over the last N days
    for lag in (1, 2, 3, 5, 10):
        features[f"return_{lag}d"] = close.pct_change(lag)

    # Trend context: is price above/below its recent moving averages
    sma_5 = close.rolling(5).mean()
    sma_20 = close.rolling(20).mean()
    features["price_vs_sma5"] = close / sma_5 - 1
    features["price_vs_sma20"] = close / sma_20 - 1
    features["sma5_vs_sma20"] = sma_5 / sma_20 - 1

    # RSI as a feature -- same math as strategies.mean_reversion_rsi
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rsi = pd.Series(index=df.index, dtype=float)
    valid = avg_loss > 0
    rsi[valid] = 100 - (100 / (1 + avg_gain[valid] / avg_loss[valid]))
    rsi[avg_loss == 0] = 100
    features["rsi_14"] = rsi

    # Recent volatility
    features["volatility_10d"] = close.pct_change().rolling(10).std()

    # Volume trend, if the data includes it
    if "Volume" in df.columns:
        features["volume_change_5d"] = df["Volume"].pct_change(5)

    return features


def build_labels(df: pd.DataFrame) -> pd.Series:
    """
    1 if TOMORROW's close is higher than today's, else 0. This uses
    shift(-1) -- looking one row into the future -- which is correct and
    necessary here: this is the TRAINING TARGET, not the trading signal.
    A model is trained to map "features known as of day t" -> "was day
    t+1 up", exactly mirroring what backtest.py's forward-shift already
    assumes every strategy's signal represents.
    """
    future_return = df["Close"].shift(-1) / df["Close"] - 1
    return (future_return > 0).astype(int)
