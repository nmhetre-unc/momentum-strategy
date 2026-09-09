"""
Each strategy function takes a price DataFrame (with a 'Close' column)
and returns a pandas Series of positions: 1 = long, 0 = flat.
"""

from collections.abc import Callable
from typing import cast

import pandas as pd

from qbt.ml import ml_direction_signal


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

    # avg_loss == 0 means every day in the window gained, which is a
    # legitimate maximally-overbought case: RSI should be 100, not NaN.
    rsi = pd.Series(index=df.index, dtype=float)
    valid = avg_loss > 0
    rsi[valid] = 100 - (100 / (1 + avg_gain[valid] / avg_loss[valid]))
    rsi[avg_loss == 0] = 100

    signal = pd.Series(index=df.index, dtype=float)
    signal[rsi < oversold] = 1
    signal[rsi > overbought] = 0
    signal = signal.ffill().fillna(0)
    return signal.astype(int)


def _momentum_eligibility(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """
    True where a ticker has accumulated at least `lookback` non-NaN
    observations as of that date -- an expanding count, not "the trailing
    `lookback` days are all non-NaN". Kept separate from whether the
    momentum score itself happens to be computable, so this is a rule
    about the ticker's history, not an artifact of the score formula.
    An all-NaN column (a ticker whose fetch failed outright, e.g. via
    qbt.universe) self-excludes here with no special case: it never
    accumulates any valid history to be eligible with.
    """
    return prices.notna().cumsum() >= lookback


def cross_sectional_momentum(
    prices: pd.DataFrame, lookback: int = 252, skip: int = 21, decile: float = 0.1
) -> pd.DataFrame:
    """
    12-1 month cross-sectional momentum. Unlike every other strategy in
    this module, this one operates on a multi-ticker price matrix (dates
    x tickers, e.g. from qbt.universe.fetch_universe) rather than a
    single asset's `Close` column, and returns a weights DataFrame (dates
    x tickers) rather than a 1/0 position Series -- it needs
    run_portfolio_backtest(), not run_backtest(), and is intentionally
    not registered in STRATEGIES below.

    Ranks tickers by trailing return from `lookback` days ago through
    `skip` days ago -- the classic 12-1 month formula, which excludes the
    most recent month because short-term reversal contaminates raw
    momentum there. Goes equal-weight long the top decile and
    equal-weight short the bottom decile of ELIGIBLE tickers, rebalanced
    on the last trading day of each calendar month. A ticker not yet
    eligible (see _momentum_eligibility) gets zero weight and cannot be
    ranked, no matter how extreme its raw price move looks.

    The returned weights hold constant between rebalance dates -- this is
    what run_portfolio_backtest() expects to shift and apply, the same
    way a single-asset strategy's signal holds between changes. Before
    the first rebalance, and on any month with fewer than 2 eligible
    tickers, the portfolio holds flat at zero rather than take a
    degenerate one-sided position.
    """
    momentum_score = prices.shift(skip) / prices.shift(lookback) - 1
    eligible = _momentum_eligibility(prices, lookback) & momentum_score.notna()

    # Last trading day actually present in the index for each calendar
    # month -- not a fixed calendar date, which might be a holiday.
    date_index = pd.DatetimeIndex(prices.index)
    month_of: pd.Series = date_index.to_series().groupby(date_index.to_period("M")).max()
    rebalance_dates = pd.DatetimeIndex(month_of.to_numpy())

    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for date in rebalance_dates:
        # `.loc[date]` on a DataFrame is typed as possibly returning a
        # DataFrame (duplicate-index case); the index is unique here by
        # construction, so it's always actually a Series -- cast rather
        # than fight pandas-stubs' overload for a case that can't happen.
        elig_tickers = cast(pd.Series, eligible.loc[date])
        n_eligible = int(elig_tickers.sum())
        if n_eligible < 2:
            continue

        n_leg = max(1, int(n_eligible * decile))
        n_leg = min(n_leg, n_eligible // 2)

        date_scores = cast(pd.Series, momentum_score.loc[date])
        scores = date_scores[elig_tickers].sort_values()
        shorts, longs = scores.index[:n_leg], scores.index[-n_leg:]

        row = pd.Series(0.0, index=prices.columns)
        row[longs] = 1.0 / n_leg
        row[shorts] = -1.0 / n_leg
        weights.loc[date] = row

    return weights.ffill().fillna(0.0)


STRATEGIES: dict[str, Callable[..., pd.Series]] = {
    "sma_crossover": sma_crossover,
    "momentum": momentum,
    "mean_reversion": mean_reversion_rsi,
    "ml_direction": ml_direction_signal,
}


# --------------------------------------------------------------------------
# Dashboard copy
# --------------------------------------------------------------------------
# Read by the dashboard instead of hard-coded strings, so a strategy and
# its displayed explanation can't drift apart.
STRATEGY_DOCS = {
    "sma_crossover": {
        "family": "Trend-following",
        "what": (
            "Goes long when the short moving average is above the long one, flat otherwise. "
            "The two averages are a crude trend filter: the short one tracks recent price, "
            "the long one tracks the established level, and the crossover is the moment "
            "recent price decisively left the old level behind."
        ),
        "works_when": (
            "Markets trend persistently. Big, long moves in one direction are where all of "
            "its return comes from."
        ),
        "fails_when": (
            "Sideways, choppy markets. Price oscillates across the moving average, generating "
            "a buy near every local top and a sell near every local bottom. This isn't bad luck — "
            "it's structural, and it's why the strategy's equity curve looks like a staircase "
            "with long flat-to-down sections between rare large gains."
        ),
        "regime_hint": (
            "Expect strong performance in trending regimes and steady bleed in the "
            "choppy/range regime."
        ),
        "watch_for": (
            "A very low trade count. Two or three trades over ten years means your Sharpe ratio "
            "is describing three coin flips, not a repeatable process, no matter how good it looks."
        ),
    },
    "momentum": {
        "family": "Trend-following",
        "what": (
            "Goes long when the trailing N-day return is above a threshold. Same underlying bet "
            "as the crossover — that recent direction persists — expressed more directly."
        ),
        "works_when": (
            "Returns are positively autocorrelated: up days tend to be followed by up days."
        ),
        "fails_when": (
            "Mean-reverting or whipsawing markets, and immediately after sharp reversals, where "
            "it is maximally long right as the move ends."
        ),
        "regime_hint": (
            "Check the autocorr_60 regime feature. Momentum should do best in regimes "
            "where it is positive."
        ),
        "watch_for": (
            "Threshold sensitivity. If the result changes character between threshold 0.00 and "
            "0.01, you are fitting noise. A real edge is not that fragile."
        ),
    },
    "mean_reversion": {
        "family": "Mean-reversion",
        "what": (
            "RSI-based: buys when RSI falls below the oversold line, exits when it rises above "
            "overbought, and holds in between. It is a bet that short-term moves overshoot and "
            "get given back."
        ),
        "works_when": (
            "Range-bound markets with negative short-horizon autocorrelation — precisely "
            "the regime that destroys trend-following."
        ),
        "fails_when": (
            "Strong trends, especially downtrends. 'Oversold' in a crash means 'cheap and getting "
            "cheaper'; the strategy buys the whole way down. This is the classic mean-reversion "
            "death: many small wins, then one loss that erases all of them."
        ),
        "regime_hint": (
            "This is the natural complement to the trend strategies — the pair is the "
            "motivating example for regime switching."
        ),
        "watch_for": (
            "Win rate near 70% with a negative total return. High hit rate and poor payoff "
            "is the signature of picking up pennies in front of a steamroller."
        ),
    },
    "ml_direction": {
        "family": "Machine learning",
        "what": (
            "Trains a classifier (logistic regression or random forest) on lagged returns, "
            "moving-average ratios, RSI, volatility and volume to predict whether tomorrow "
            "closes higher, then goes long on every predicted up day."
        ),
        "works_when": (
            "There is a stable, learnable relationship between today's features and "
            "tomorrow's direction. On liquid index data, mostly there is not."
        ),
        "fails_when": (
            "Always, unless you check. Daily direction is close to a coin flip; a flexible model "
            "handed 12 features and 2,000 rows will find patterns in noise and report high "
            "training accuracy for them."
        ),
        "regime_hint": (
            "Try the regime-conditional variant in ml_strategy.py: one model per regime is "
            "a different bet from one model for all conditions."
        ),
        "watch_for": (
            "The train/test accuracy gap. Random forest at 85% train / 48% test is not a good "
            "model with a bug — it is a memorizer, working exactly as a high-capacity model does "
            "on noise."
        ),
    },
}

# Slider definitions, so the dashboard can build parameter controls
# generically and adaptive.py can sanity-check the values it's handed.
PARAM_SPECS = {
    "sma_crossover": [
        {
            "name": "short_window", "label": "Short SMA window",
            "min": 5, "max": 100, "default": 50, "step": 1,
            "help": "Days in the fast average. Shorter reacts sooner and whipsaws more.",
        },
        {
            "name": "long_window", "label": "Long SMA window",
            "min": 50, "max": 300, "default": 200, "step": 1,
            "help": (
                "Days in the slow average. 200 is conventional, which is itself a reason "
                "to test whether it matters."
            ),
        },
    ],
    "momentum": [
        {
            "name": "lookback", "label": "Lookback (days)",
            "min": 5, "max": 60, "default": 20, "step": 1,
            "help": "Window over which the trailing return is measured.",
        },
        {
            "name": "threshold", "label": "Return threshold",
            "min": -0.05, "max": 0.05, "default": 0.0, "step": 0.005,
            "help": (
                "How positive the trailing return must be before going long. Above 0 "
                "demands confirmation."
            ),
        },
    ],
    "mean_reversion": [
        {
            "name": "period", "label": "RSI period",
            "min": 5, "max": 30, "default": 14, "step": 1,
            "help": "Averaging window for RSI. Shorter is twitchier.",
        },
        {
            "name": "oversold", "label": "Oversold threshold",
            "min": 10, "max": 40, "default": 30, "step": 1,
            "help": "RSI level that triggers a buy.",
        },
        {
            "name": "overbought", "label": "Overbought threshold",
            "min": 60, "max": 90, "default": 70, "step": 1,
            "help": "RSI level that triggers an exit.",
        },
    ],
    "ml_direction": [
        {
            "name": "train_frac", "label": "Train fraction",
            "min": 0.5, "max": 0.9, "default": 0.7, "step": 0.05,
            "help": (
                "Share of history used for fitting. The rest is the only part whose "
                "performance means anything."
            ),
        },
    ],
}


def default_params(name: str) -> dict:
    """The default parameter set for a strategy, straight from PARAM_SPECS."""
    return {spec["name"]: spec["default"] for spec in PARAM_SPECS.get(name, [])}
