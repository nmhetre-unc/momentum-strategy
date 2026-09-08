"""
Rolling features that describe the market ENVIRONMENT rather than the
direction of the next move.

features.py answers "is tomorrow up?". This module answers a different
question: "what kind of market are we in right now?" -- calm and
trending, choppy and directionless, or violent and falling. Those are
regimes, and a strategy that prints money in one of them can bleed out
in another.

Every column here is point-in-time: the value on row t is built only
from information available at day t's close. That matters more here than
almost anywhere else in the project, because a regime label that
secretly peeked at the future makes every downstream backtest look
brilliant and be worthless. See the note on standardize_features().
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Short descriptions surfaced as tooltips in the dashboard. Keys must
# match the column names produced by build_regime_features().
FEATURE_DOCS = {
    "vol_20d": "Annualized realized volatility over the last 20 days. The single most useful regime variable — volatility clusters, so high-vol days beget high-vol days.",
    "vol_ratio": "20-day vol divided by 100-day vol. Above 1 means volatility is EXPANDING (stress building); below 1 means it's contracting (calm returning).",
    "vol_percentile": "Where today's 20-day vol sits in its own expanding history (0 = calmest ever, 1 = most violent ever). Expanding, not full-sample, so it never peeks ahead.",
    "trend_60d": "Trailing 60-day return. Crude but honest measure of which way the market has been going.",
    "price_vs_sma200": "Price relative to its 200-day moving average. The classic bull/bear dividing line.",
    "sma_slope": "20-day change in the 50-day moving average, scaled by price. Captures whether the trend is accelerating or rolling over.",
    "efficiency_ratio": "Kaufman efficiency ratio: net move divided by total path travelled over 60 days. Near 1 = clean trend, near 0 = the market went nowhere loudly (chop).",
    "autocorr_60": "Rolling lag-1 autocorrelation of daily returns. Positive = momentum-friendly (moves follow through), negative = mean-reversion-friendly (moves get given back).",
    "drawdown_252d": "How far below the trailing 1-year high we are. Distinguishes 'high vol on the way up' from 'high vol on the way down'.",
    "downside_share": "Share of total volatility coming from down days. High values mean the risk is one-sided and unpleasant.",
    "parkinson_vol": "Volatility estimated from the daily high-low range. Reacts faster than close-to-close vol because it sees intraday damage.",
    "range_pct": "Average daily high-low range as a fraction of price. A direct read on intraday turbulence.",
    "volume_z": "Volume relative to its own recent history, in standard deviations. Volume spikes usually accompany regime changes.",
    "volume_trend": "20-day average volume divided by 100-day average volume. Rising participation vs. drying up.",
    "illiquidity": "Amihud-style illiquidity: how much price moves per dollar traded. Higher = thinner, more fragile market.",
}

# Features that must exist for the naming/ordering logic in regime.py to
# work. build_regime_features() always produces these regardless of
# whether the input has High/Low/Volume columns.
CORE_FEATURES = ["vol_20d", "vol_ratio", "vol_percentile", "trend_60d", "efficiency_ratio"]


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized standard deviation of daily returns over a rolling window."""
    return close.pct_change().rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def efficiency_ratio(close: pd.Series, window: int = 60) -> pd.Series:
    """
    Kaufman's efficiency ratio: |net change| / sum(|daily changes|).

    1.0 means the market walked in a straight line; 0.0 means it thrashed
    around and ended where it started. This is the cleanest single
    distinction between "trending" and "choppy", and it's the reason a
    trend strategy can have a great Sharpe in one year and a terrible one
    in the next with identical average volatility.
    """
    net_move = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return net_move / path_length.replace(0, np.nan)


def rolling_autocorrelation(close: pd.Series, window: int = 60, lag: int = 1) -> pd.Series:
    """Rolling correlation between daily returns and their own lagged values."""
    returns = close.pct_change()
    return returns.rolling(window).corr(returns.shift(lag))


def parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Range-based volatility estimator. Uses the intraday high-low spread,
    which contains information a close-to-close estimate throws away --
    a day that crashed 5% and recovered looks quiet to close-to-close vol
    and correctly looks violent here.
    """
    log_range = np.log(df["High"] / df["Low"])
    variance = (log_range ** 2).rolling(window).mean() / (4 * np.log(2))
    return np.sqrt(variance * TRADING_DAYS_PER_YEAR)


def build_regime_features(
    df: pd.DataFrame,
    vol_window: int = 20,
    trend_window: int = 60,
    long_window: int = 252,
) -> pd.DataFrame:
    """
    Builds the full regime feature table.

    Columns depend on what the input actually has: High/Low add
    range-based turbulence measures, Volume adds liquidity measures.
    A Close-only DataFrame (like the synthetic data in test_logic.py)
    still produces every column in CORE_FEATURES, so nothing downstream
    needs to special-case it.
    """
    close = df["Close"]
    returns = close.pct_change()
    features = pd.DataFrame(index=df.index)

    # --- Volatility: level, direction of travel, and historical context ---
    vol_short = realized_volatility(close, vol_window)
    vol_long = realized_volatility(close, min(long_window // 2, 100))
    features["vol_20d"] = vol_short
    features["vol_ratio"] = vol_short / vol_long.replace(0, np.nan)

    # Expanding (not full-sample) percentile rank -- on day t this only
    # knows about days up to t. A full-sample rank would leak the future
    # into what is supposed to be a point-in-time description of today.
    features["vol_percentile"] = vol_short.expanding(min_periods=vol_window).rank(pct=True)

    # --- Trend: magnitude, position, and whether it's still accelerating ---
    features["trend_60d"] = close.pct_change(trend_window)
    sma_long = close.rolling(long_window).mean()
    features["price_vs_sma200"] = close / sma_long - 1
    sma_mid = close.rolling(50).mean()
    features["sma_slope"] = (sma_mid - sma_mid.shift(20)) / close

    # --- Character of the path: trending vs. chopping ---
    features["efficiency_ratio"] = efficiency_ratio(close, trend_window)
    features["autocorr_60"] = rolling_autocorrelation(close, trend_window)

    # --- Where we are relative to the recent peak ---
    rolling_peak = close.rolling(long_window, min_periods=vol_window).max()
    features["drawdown_252d"] = close / rolling_peak - 1

    # --- Asymmetry of risk: is the volatility one-sided? ---
    # Root-mean-square of the negative returns over the RMS of all of
    # them. Computed this way rather than as std(returns[returns < 0])
    # because that version has a NaN in most windows -- rolling() needs
    # `window` non-missing observations, and half the days are positive.
    negative = returns.clip(upper=0)
    downside_rms = np.sqrt((negative ** 2).rolling(vol_window).mean())
    total_rms = np.sqrt((returns ** 2).rolling(vol_window).mean())
    features["downside_share"] = downside_rms / total_rms.replace(0, np.nan)

    # --- Intraday turbulence, when we have the bars for it ---
    if {"High", "Low"}.issubset(df.columns):
        features["parkinson_vol"] = parkinson_volatility(df, vol_window)
        features["range_pct"] = ((df["High"] - df["Low"]) / close).rolling(vol_window).mean()

    # --- Liquidity, when we have volume ---
    if "Volume" in df.columns:
        volume = df["Volume"].astype(float)
        vol_mean = volume.rolling(long_window // 2, min_periods=vol_window).mean()
        vol_std = volume.rolling(long_window // 2, min_periods=vol_window).std()
        features["volume_z"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
        features["volume_trend"] = (
            volume.rolling(vol_window).mean() / vol_mean.replace(0, np.nan)
        )
        dollar_volume = (volume * close).replace(0, np.nan)
        features["illiquidity"] = np.log1p(returns.abs() / dollar_volume * 1e9).rolling(vol_window).mean()

    # Infinities come from divisions where the denominator was ~0 in a
    # degenerate stretch (e.g. a perfectly flat synthetic series). Treat
    # them as missing rather than letting them poison a fitted model.
    return features.replace([np.inf, -np.inf], np.nan)


def standardize_features(
    features: pd.DataFrame,
    method: str = "expanding",
    min_periods: int = 60,
    clip: float = 5.0,
) -> pd.DataFrame:
    """
    Puts every feature on a comparable scale so a clustering model isn't
    dominated by whichever column happens to have the biggest units.

    method="expanding" (default) z-scores each column against only its
    own past -- causal, and the right choice when the labels feed a
    backtest. method="full" z-scores against the entire sample, which is
    what most tutorials do and which quietly leaks the future: knowing
    the sample-wide mean volatility is knowing something you could not
    have known in 2015. It's kept here deliberately so the dashboard can
    show interns the size of the difference.
    """
    if method == "full":
        centered = features - features.mean()
        scaled = centered / features.std().replace(0, np.nan)
    elif method == "expanding":
        mean = features.expanding(min_periods=min_periods).mean()
        std = features.expanding(min_periods=min_periods).std()
        scaled = (features - mean) / std.replace(0, np.nan)
    else:
        raise ValueError(f"Unknown standardization method: {method!r}")

    return scaled.replace([np.inf, -np.inf], np.nan).clip(-clip, clip)


def reduce_dimensions(features: pd.DataFrame, n_components: int = 3, fit_rows: pd.Index = None):
    """
    Optional PCA compression of the (correlated) regime features.

    Most of these columns measure two or three underlying things --
    "how violent" and "which way" -- wearing different hats. PCA makes
    that explicit and often makes clusters cleaner. Returns
    (components_df, fitted_pca) so the caller can inspect the loadings,
    which is where the actual insight is.

    `fit_rows` restricts the PCA fit to a subset of the index (e.g. the
    in-sample period) while still transforming everything.
    """
    from sklearn.decomposition import PCA

    valid = features.dropna()
    if valid.empty:
        raise ValueError("No complete rows available to fit PCA on.")

    fit_data = valid.loc[valid.index.intersection(fit_rows)] if fit_rows is not None else valid
    if len(fit_data) <= n_components:
        raise ValueError(f"Need more than {n_components} complete rows to fit PCA; got {len(fit_data)}.")

    pca = PCA(n_components=n_components)
    pca.fit(fit_data)

    transformed = pd.DataFrame(
        pca.transform(valid),
        index=valid.index,
        columns=[f"pc_{i + 1}" for i in range(n_components)],
    )
    return transformed.reindex(features.index), pca
