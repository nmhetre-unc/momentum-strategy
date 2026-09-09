"""
Regime-aware strategy wrappers. Each fixed strategy in strategies.py
applies one rule to every day in history; these wrappers condition on
regime instead, via four mechanisms: filtering (sit out bad regimes),
switching (a different strategy per regime), re-parameterizing (same
strategy, per-regime settings), and position sizing (scale by volatility
or regime).

Every function here returns a fractional position in [0, 1], unlike the
binary output of strategies.py; run_backtest() handles both identically.

Every `auto` choice (which regimes to allow, which strategy to run
where, how big to size) is learned only from data up to `learn_frac` of
the sample (default 0.6) and applied unchanged thereafter;
describe_choices() exposes what was learned.
"""

import numpy as np
import pandas as pd

from qbt.analytics import sharpe_ratio
from qbt.backtest import run_backtest
from qbt.ml import ml_direction_signal
from qbt.regime import UNKNOWN, detect_regimes, detect_regimes_walk_forward
from qbt.regime_features import realized_volatility
from qbt.strategies import STRATEGIES

# Strategies the `auto` selectors choose between. ml_direction is
# excluded since it would be refit inside every candidate evaluation;
# use ml_regime_conditional instead.
AUTO_CANDIDATES = ("sma_crossover", "momentum", "mean_reversion")

# Minimum days in the learning window before drawing a conclusion about a regime.
MIN_LEARN_DAYS = 60


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------
def _resolve_regimes(df: pd.DataFrame, regimes=None, regime_method: str = "hmm",
                     n_regimes: int = 3, regime_fit_frac: float = 0.7,
                     regime_smooth: str = "min_duration", min_duration: int = 5,
                     regime_walk_forward: bool = False):
    """
    Accepts either a precomputed RegimeResult / label Series, or the
    parameters needed to detect regimes from scratch. The dashboard
    passes a precomputed result; the CLI and walk_forward.py let it be
    computed here, which keeps these wrappers usable as plain
    strategy_fn(df, **params).
    """
    if regimes is not None:
        if hasattr(regimes, "labels"):
            return regimes
        labels = pd.Series(regimes).reindex(df.index).fillna(UNKNOWN).astype(int)
        names = {int(r): f"Regime {int(r)}" for r in sorted(labels.unique()) if r >= 0}
        return type("_Labels", (), {"labels": labels, "names": names, "causal": True})()

    if regime_walk_forward:
        return detect_regimes_walk_forward(
            df, method=regime_method, n_regimes=n_regimes,
            smooth=regime_smooth, min_duration=min_duration,
        )
    return detect_regimes(
        df, method=regime_method, n_regimes=n_regimes, fit_frac=regime_fit_frac,
        smooth=regime_smooth, min_duration=min_duration,
    )


def _base_signal(df: pd.DataFrame, name: str, params: dict | None = None) -> pd.Series:
    """Runs a base strategy by name and normalizes it to a float series on df.index."""
    if name == "flat":
        return pd.Series(0.0, index=df.index)
    if name not in STRATEGIES:
        raise KeyError(
            f"Unknown base strategy {name!r}. Available: {sorted(STRATEGIES)} (or 'flat')."
        )
    signal = STRATEGIES[name](df, **(params or {}))
    return signal.reindex(df.index).fillna(0).astype(float)


def _learn_cutoff(df: pd.DataFrame, learn_frac: float):
    """The last date an `auto` rule is allowed to look at."""
    return df.index[max(int(len(df) * learn_frac) - 1, 0)]


def _regime_sharpe_table(df: pd.DataFrame, labels: pd.Series, candidates,
                         learn_end, min_days: int = MIN_LEARN_DAYS) -> pd.DataFrame:
    """
    Sharpe ratio of each candidate strategy inside each regime, measured
    only on days up to learn_end. This is the evidence behind every
    automatic choice this module makes; describe_choices() surfaces it
    verbatim.
    """
    window = df.index <= learn_end
    rows = []

    for name in candidates:
        signal = _base_signal(df, name)
        result = run_backtest(df, signal)
        returns = result["strategy_return"].fillna(0)

        for regime_id in sorted(r for r in labels.unique() if r != UNKNOWN):
            mask = window & (labels == regime_id).to_numpy()
            segment = returns[mask]
            rows.append({
                "regime": int(regime_id),
                "strategy": name,
                "days": int(mask.sum()),
                "sharpe": sharpe_ratio(segment) if len(segment) >= min_days else np.nan,
                "mean_return": segment.mean() if len(segment) else np.nan,
            })

    return pd.DataFrame(rows)


def _best_per_regime(table: pd.DataFrame, allow_flat: bool = True) -> dict:
    """
    Picks the highest-Sharpe candidate per regime from the learning
    table. A regime where nothing achieved a positive Sharpe maps to
    'flat'.
    """
    choices: dict[int, str | None] = {}
    for regime_id, group in table.groupby("regime"):
        # pandas-stubs types a groupby key as a broad Hashable union (it
        # can't know from the DataFrame's static type that "regime" holds
        # plain ints); it always does here, by construction in the caller.
        regime_key = int(regime_id)  # type: ignore[arg-type]
        scored = group.dropna(subset=["sharpe"])
        if scored.empty:
            # Not enough learning data to judge this regime. Trade it
            # normally rather than inventing a rule from 12 days.
            choices[regime_key] = None
            continue
        best = scored.loc[scored["sharpe"].idxmax()]
        choices[regime_key] = (
            "flat" if (allow_flat and best["sharpe"] <= 0) else str(best["strategy"])
        )
    return choices


def _apply_by_regime(df: pd.DataFrame, labels: pd.Series, signals: dict,
                     default: pd.Series | None = None) -> pd.Series:
    """Selects, for each day, the signal belonging to that day's regime."""
    out = pd.Series(0.0, index=df.index) if default is None else default.astype(float).copy()
    for regime_id, signal in signals.items():
        mask = labels == regime_id
        out[mask] = signal.reindex(df.index)[mask].astype(float)
    # Warm-up days have no regime and therefore no basis for a position.
    out[labels == UNKNOWN] = 0.0
    return out.fillna(0.0)


# --------------------------------------------------------------------------
# 1. Filtering -- same strategy, sit out the bad regimes
# --------------------------------------------------------------------------
def regime_filtered(df: pd.DataFrame, base: str = "sma_crossover", base_params: dict | None = None,
                    allowed_regimes=None, learn_frac: float = 0.6, **regime_kwargs) -> pd.Series:
    """
    Runs `base` normally, but forces the position flat in regimes it
    isn't allowed to trade. allowed_regimes=None (default) learns the
    allow-list from the first `learn_frac` of history, keeping any regime
    where the base strategy had a positive Sharpe there; pass an explicit
    tuple to test your own hypothesis instead. Only removes existing
    positions, never invents new ones.
    """
    regime_result = _resolve_regimes(df, **regime_kwargs)
    labels = regime_result.labels
    signal = _base_signal(df, base, base_params)

    if allowed_regimes is None:
        learn_end = _learn_cutoff(df, learn_frac)
        table = _regime_sharpe_table(df, labels, [base], learn_end)
        allowed = [
            int(row["regime"]) for _, row in table.iterrows()
            if pd.isna(row["sharpe"]) or row["sharpe"] > 0
        ]
    else:
        allowed = [int(r) for r in allowed_regimes]

    gated = signal.where(labels.isin(allowed), 0.0)
    gated[labels == UNKNOWN] = 0.0
    return gated.astype(float)


# --------------------------------------------------------------------------
# 2. Switching -- a different strategy in each regime
# --------------------------------------------------------------------------
def regime_switch(df: pd.DataFrame, strategy_map: dict | None = None, candidates=AUTO_CANDIDATES,
                  learn_frac: float = 0.6, allow_flat: bool = True, **regime_kwargs) -> pd.Series:
    """
    Runs whichever strategy suits the current regime, motivated by the
    trend/mean-reversion pairing (trend-following in trending regimes,
    RSI reversion in ranges). Regime labels arrive late and switching
    flips the whole position, both of which cost real money -- run with
    cost_bps=10 before drawing conclusions. strategy_map={0: "momentum",
    ...} pins the mapping explicitly; None learns it from the first
    `learn_frac` of history.
    """
    regime_result = _resolve_regimes(df, **regime_kwargs)
    labels = regime_result.labels

    if strategy_map is None:
        learn_end = _learn_cutoff(df, learn_frac)
        table = _regime_sharpe_table(df, labels, candidates, learn_end)
        strategy_map = _best_per_regime(table, allow_flat=allow_flat)

    signals = {}
    for regime_id, name in strategy_map.items():
        chosen = candidates[0] if name is None else name
        signals[int(regime_id)] = _base_signal(df, chosen)

    return _apply_by_regime(df, labels, signals)


# --------------------------------------------------------------------------
# 3. Re-parameterizing -- same strategy, regime-specific settings
# --------------------------------------------------------------------------
def regime_parameters(df: pd.DataFrame, base: str = "sma_crossover", param_map: dict | None = None,
                      **regime_kwargs) -> pd.Series:
    """
    One strategy, different parameters per regime. The default map
    shortens trend windows as volatility rises. Three regimes times two
    window parameters is six numbers fitted to one price history; the
    equity curve will improve regardless, and walk-forward is how you
    check whether anything real did.
    """
    regime_result = _resolve_regimes(df, **regime_kwargs)
    labels = regime_result.labels
    regime_ids = sorted(r for r in labels.unique() if r != UNKNOWN)

    if param_map is None:
        param_map = _default_param_map(base, regime_ids)

    signals = {
        int(regime_id): _base_signal(df, base, params)
        for regime_id, params in param_map.items()
    }
    return _apply_by_regime(df, labels, signals)


def _default_param_map(base: str, regime_ids: list) -> dict:
    """
    Faster settings for higher-volatility regimes. Regime IDs are ordered
    by volatility (0 = calmest), which is what makes a default like this
    expressible.
    """
    n = max(len(regime_ids), 1)
    presets: dict[str, list[dict]] = {
        "sma_crossover": [{"short_window": s, "long_window": lw}
                          for s, lw in ((50, 200), (30, 120), (15, 60), (10, 40))],
        "momentum": [{"lookback": lb, "threshold": 0.0} for lb in (40, 20, 10, 5)],
        "mean_reversion": [{"period": p, "oversold": o, "overbought": 100 - o}
                           for p, o in ((14, 30), (14, 25), (10, 20), (7, 15))],
    }
    if base not in presets:
        raise ValueError(
            f"No default parameter map for base strategy {base!r}. "
            "Pass param_map={regime_id: {...}} explicitly."
        )
    ladder = presets[base]
    # Spread the presets evenly across however many regimes there are.
    return {
        int(regime_id): ladder[min(int(i * len(ladder) / n), len(ladder) - 1)]
        for i, regime_id in enumerate(regime_ids)
    }


# --------------------------------------------------------------------------
# 4. Position sizing
# --------------------------------------------------------------------------
def volatility_targeted(df: pd.DataFrame, base: str = "sma_crossover",
                        base_params: dict | None = None, target_vol: float = 0.15,
                        vol_window: int = 20, max_leverage: float = 1.0,
                        **_ignored_regime_kwargs) -> pd.Series:
    """
    Keeps the base signal, but scales the position so expected volatility
    stays near `target_vol` annualized:
    position = signal x clip(target_vol / trailing_realized_vol, 0, max_leverage).
    The trailing vol window ends at day t and the whole signal is shifted
    forward a day in run_backtest(), so nothing here sees the future.
    max_leverage caps the multiplier at 1.0 by default, keeping output in
    [0, 1]; above 1.0 means borrowing. Regime keyword arguments are
    accepted and ignored, so this drops into any comparison alongside the
    regime-aware wrappers unchanged.
    """
    signal = _base_signal(df, base, base_params)
    trailing_vol = realized_volatility(df["Close"], vol_window)

    scale = (target_vol / trailing_vol.replace(0, np.nan)).clip(upper=max_leverage)
    # Before the vol window has spun up there is no risk estimate, so
    # there is no basis for a size. Flat is the honest answer.
    scale = scale.fillna(0.0)
    return (signal * scale).clip(0.0, max_leverage).astype(float)


def regime_sized(df: pd.DataFrame, base: str = "sma_crossover", base_params: dict | None = None,
                 size_map: dict | None = None, **regime_kwargs) -> pd.Series:
    """
    Keeps the base signal, but sets position size per regime, sized
    inversely to each regime's own volatility and normalized so the
    calmest regime gets a full position. The discrete cousin of
    volatility_targeted(): coarser, but changes size only at regime
    boundaries, which is cheaper to trade.
    """
    regime_result = _resolve_regimes(df, **regime_kwargs)
    labels = regime_result.labels
    signal = _base_signal(df, base, base_params)

    if size_map is None:
        returns = df["Close"].pct_change()
        vol_by_regime = {
            int(r): returns[labels == r].std()
            for r in sorted(x for x in labels.unique() if x != UNKNOWN)
        }
        calmest = min((v for v in vol_by_regime.values() if v and not pd.isna(v)), default=None)
        size_map = (
            dict.fromkeys(vol_by_regime, 1.0) if calmest is None
            else {r: float(np.clip(calmest / v, 0.0, 1.0)) if v else 0.0
                  for r, v in vol_by_regime.items()}
        )

    sizes = labels.map(size_map).astype(float).fillna(0.0)
    return (signal * sizes).clip(0.0, 1.0).astype(float)


# --------------------------------------------------------------------------
# Composite: everything at once
# --------------------------------------------------------------------------
def adaptive_ensemble(df: pd.DataFrame, candidates=AUTO_CANDIDATES, learn_frac: float = 0.6,
                      target_vol: float = 0.15, vol_window: int = 20,
                      max_leverage: float = 1.0, allow_flat: bool = True,
                      **regime_kwargs) -> pd.Series:
    """
    Regime-based strategy switching, with the result volatility-targeted
    on top. Combines the switching and sizing mechanisms; compare against
    each individually, since stacking mechanisms stacks their assumptions
    too.
    """
    regime_result = _resolve_regimes(df, **regime_kwargs)
    labels = regime_result.labels

    learn_end = _learn_cutoff(df, learn_frac)
    table = _regime_sharpe_table(df, labels, candidates, learn_end)
    strategy_map = _best_per_regime(table, allow_flat=allow_flat)

    signals = {
        int(regime_id): _base_signal(df, candidates[0] if name is None else name)
        for regime_id, name in strategy_map.items()
    }
    switched = _apply_by_regime(df, labels, signals)

    trailing_vol = realized_volatility(df["Close"], vol_window)
    scale = (target_vol / trailing_vol.replace(0, np.nan)).clip(upper=max_leverage).fillna(0.0)
    return (switched * scale).clip(0.0, max_leverage).astype(float)


def ml_regime_conditional(df: pd.DataFrame, train_frac: float = 0.7, model_type: str = "logistic",
                          regime_mode: str = "conditional", **regime_kwargs) -> pd.Series:
    """
    The ML direction model, told which regime it's in.
    regime_mode="feature" gives one model the regime as an input column;
    "conditional" fits a separate model per regime -- see qbt/ml.py's
    module docstring for the overfitting tradeoff.
    """
    regime_result = _resolve_regimes(df, **regime_kwargs)
    return ml_direction_signal(
        df, train_frac=train_frac, model_type=model_type,
        regimes=regime_result.labels, regime_mode=regime_mode,
    ).astype(float)


# --------------------------------------------------------------------------
# Introspection -- what did `auto` actually decide?
# --------------------------------------------------------------------------
def describe_choices(df: pd.DataFrame, candidates=AUTO_CANDIDATES, learn_frac: float = 0.6,
                     allow_flat: bool = True, **regime_kwargs) -> dict:
    """
    Returns the learning-window evidence and the resulting per-regime
    choice, so an automatic decision can be inspected instead of trusted.
    Read `table` before `choices`: a winning margin of 0.05 Sharpe over
    80 days is not statistically meaningful.
    """
    regime_result = _resolve_regimes(df, **regime_kwargs)
    labels = regime_result.labels
    learn_end = _learn_cutoff(df, learn_frac)
    table = _regime_sharpe_table(df, labels, candidates, learn_end)

    return {
        "learn_end": learn_end,
        "table": table,
        "choices": _best_per_regime(table, allow_flat=allow_flat),
        "names": regime_result.names,
        "regime_method": getattr(regime_result, "method", "provided"),
        "causal": getattr(regime_result, "causal", True),
    }


def describe_filter(df: pd.DataFrame, base: str = "sma_crossover", base_params: dict | None = None,
                    learn_frac: float = 0.6, **regime_kwargs) -> dict:
    """
    The allow-list regime_filtered() would learn, plus the evidence. If
    the base strategy had a negative Sharpe in every regime during the
    learning window, the allow-list comes back empty and the strategy
    stays flat for the whole period -- this is correct output, not a bug.
    """
    regime_result = _resolve_regimes(df, **regime_kwargs)
    labels = regime_result.labels
    learn_end = _learn_cutoff(df, learn_frac)
    table = _regime_sharpe_table(df, labels, [base], learn_end)

    allowed = [
        int(row["regime"]) for _, row in table.iterrows()
        if pd.isna(row["sharpe"]) or row["sharpe"] > 0
    ]
    return {
        "base": base,
        "learn_end": learn_end,
        "table": table,
        "allowed": allowed,
        "blocked": [int(r) for r in table["regime"] if int(r) not in allowed],
        "names": regime_result.names,
        "all_blocked": len(allowed) == 0,
    }


ADAPTIVE_STRATEGIES = {
    "regime_filtered": regime_filtered,
    "regime_switch": regime_switch,
    "regime_parameters": regime_parameters,
    "volatility_targeted": volatility_targeted,
    "regime_sized": regime_sized,
    "adaptive_ensemble": adaptive_ensemble,
    "ml_regime_conditional": ml_regime_conditional,
}

# One registry for the dashboard and CLI. Kept here rather than in
# strategies.py so that strategies.py stays free of any dependency on
# regime.py -- the base strategies must remain runnable, and testable,
# with no regime machinery in the picture at all.
ALL_STRATEGIES = {**STRATEGIES, **ADAPTIVE_STRATEGIES}

ADAPTIVE_DOCS = {
    "regime_filtered": {
        "mechanism": "Filtering",
        "what": "Runs one strategy, but stands aside in regimes where it historically lost money.",
        "why": (
            "Removes a known-bad exposure without predicting anything new. The most robust "
            "of the four mechanisms, and the one most likely to survive out-of-sample."
        ),
        "watch_for": (
            "Exposure. If filtering cuts you to 30% invested, your remaining metrics are "
            "computed on far fewer days than the headline period suggests."
        ),
    },
    "regime_switch": {
        "mechanism": "Switching",
        "what": (
            "Runs a different strategy in each regime — trend-following in trends, "
            "mean-reversion in ranges."
        ),
        "why": (
            "Trend and mean-reversion profit from opposite behaviours. If you can tell "
            "which you're in, you can in principle harvest both."
        ),
        "watch_for": (
            "Turnover and lag. Switching flips the entire position, and smoothed regime "
            "labels arrive several days late — right when the new regime's move is largest. "
            "Turn on cost_bps before believing the result."
        ),
    },
    "regime_parameters": {
        "mechanism": "Re-parameterizing",
        "what": (
            "One strategy, but faster settings in high-volatility regimes and slower ones "
            "in calm markets."
        ),
        "why": (
            "Volatile markets move faster; a window tuned for calm conditions may be far "
            "too slow for them."
        ),
        "watch_for": (
            "Degrees of freedom. Every regime hands you a fresh parameter set to tune. "
            "In-sample results will improve almost by construction — that improvement is "
            "not evidence."
        ),
    },
    "volatility_targeted": {
        "mechanism": "Position sizing",
        "what": "Keeps the signal, scales position size to hold expected volatility near a target.",
        "why": (
            "Volatility is far more forecastable than direction. Sizing down before "
            "turbulence usually cuts drawdown while leaving return broadly intact."
        ),
        "watch_for": (
            "This often beats every regime-based mechanism here, using no regime model at "
            "all. If it does, that is the finding — report it."
        ),
    },
    "regime_sized": {
        "mechanism": "Position sizing",
        "what": "Position size set per regime, inversely to that regime's volatility.",
        "why": (
            "The discrete version of volatility targeting. Resizes only at regime "
            "boundaries, so it trades far less."
        ),
        "watch_for": (
            "Compare it directly against volatility_targeted. If the continuous version "
            "wins, the regime labels added nothing beyond what trailing volatility already knew."
        ),
    },
    "adaptive_ensemble": {
        "mechanism": "All of the above",
        "what": "Regime-based strategy switching with volatility targeting layered on top.",
        "why": (
            "The full system, and the right thing to benchmark the individual "
            "mechanisms against."
        ),
        "watch_for": (
            "Attribution. Run each mechanism alone before running them together, or you "
            "will not know which part earned the result — and stacked mechanisms stack "
            "their assumptions."
        ),
    },
    "ml_regime_conditional": {
        "mechanism": "Regime-conditioned ML",
        "what": (
            "The direction classifier, either given the regime as a feature or fitted "
            "separately per regime."
        ),
        "why": (
            "The relationship between today's features and tomorrow's return may "
            "genuinely differ across regimes."
        ),
        "watch_for": (
            "Sample size. Conditional mode splits your training data by regime while the "
            "parameter count stays put. Check the per-regime train_rows column in the model report."
        ),
    },
}
