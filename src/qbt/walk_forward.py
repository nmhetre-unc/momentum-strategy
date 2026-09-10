"""
Checks whether a strategy's performance holds up on data it wasn't
fitted on. Three entry points, in increasing rigor: evaluate_out_of_sample()
(one split), rolling_walk_forward() (many consecutive out-of-sample
folds), and evaluate_with_regimes() (the out-of-sample result, split by
market regime).
"""

import inspect

import numpy as np
import pandas as pd

from qbt.analytics import full_report, performance_by_regime
from qbt.backtest import run_backtest


def evaluate_out_of_sample(df: pd.DataFrame, strategy_fn, split_frac: float = 0.7,
                           cost_bps: float = 5.0, n_boot: int = 0, **strategy_params) -> dict:
    """
    Computes the signal and backtest once on the full dataset, then
    splits the result chronologically at `split_frac`: everything before
    is in-sample, everything after is out-of-sample. Reports full metrics
    for each half separately. `n_boot` is forwarded to full_report(); it
    defaults to 0 (no Sharpe CI) since this runs in loops elsewhere.
    """
    signal = strategy_fn(df, **strategy_params)
    result = run_backtest(df, signal, cost_bps=cost_bps)

    split_idx = int(len(result) * split_frac)
    split_date = result.index[split_idx]

    in_sample = result.loc[:split_date]
    out_sample = result.loc[split_date:]

    return {
        "split_date": str(split_date.date()),
        "in_sample": full_report(in_sample, n_boot=n_boot),
        "out_sample": full_report(out_sample, n_boot=n_boot),
    }


def _rolling_folds(df: pd.DataFrame, strategy_fn, train_days: int, test_days: int,
                   cost_bps: float, n_boot: int, refit_per_fold: bool,
                   fold_override: dict, strategy_params: dict) -> dict:
    """
    One pass of the rolling walk-forward loop -- shared so both the
    refit-per-fold and single-fit variants run through identical fold
    bookkeeping and differ only in how the signal for each fold's test
    window is produced.

    refit_per_fold=False: `strategy_fn` runs once on the whole series;
    every fold's test window is sliced out of that one continuous
    backtest (the original behavior).

    refit_per_fold=True: for each fold, `strategy_fn` runs fresh on just
    `df.iloc[start-train_days:start+test_days]` -- that fold's own
    train_days-length window immediately preceding its test window, plus
    the test window itself so a stateless rule has the trailing context
    it needs. `fold_override` (train_frac/learn_frac set to
    train_days/(train_days+test_days), for strategies that accept either)
    makes any internal fit boundary land exactly at the fold's own
    train/test split. Only the resulting test-window slice of that
    fold-local backtest is kept; nothing from a fold's own train_days is
    ever reported.
    """
    if not refit_per_fold:
        signal = strategy_fn(df, **strategy_params)
        full_result = run_backtest(df, signal, cost_bps=cost_bps)
    else:
        # A precomputed regime result (or anything else carrying a `.labels`
        # index aligned to the WHOLE series) can't be silently reused on a
        # fold's shorter slice -- its labels would misalign, and even if
        # they didn't, they'd have been fit on future data relative to an
        # early fold. Fail clearly here rather than let it crash confusingly
        # deep inside the strategy with a shape-mismatch error.
        fold_length = train_days + test_days
        for key, value in strategy_params.items():
            labels = getattr(value, "labels", None)
            if labels is not None and len(labels) != fold_length:
                raise ValueError(
                    f"refit_per_fold=True re-runs strategy_fn on each fold's own "
                    f"{fold_length}-row window, but strategy_params[{key!r}] carries "
                    f"labels for {len(labels)} rows -- presumably fit on the whole "
                    f"series. A precomputed regime result can't be sliced to a fold's "
                    f"own index without re-fitting it, which would defeat the point of "
                    f"refit_per_fold. Either omit {key!r} so the strategy detects "
                    f"regimes fresh on each fold, or pass refit_per_fold=False."
                )

    folds: list[dict] = []
    oos_windows = []
    start = train_days
    while start + test_days <= len(df):
        if refit_per_fold:
            fold_df = df.iloc[start - train_days:start + test_days]
            fold_signal = strategy_fn(fold_df, **{**strategy_params, **fold_override})
            fold_result = run_backtest(fold_df, fold_signal, cost_bps=cost_bps)
            window = fold_result.iloc[train_days:train_days + test_days]
        else:
            window = full_result.iloc[start:start + test_days]

        stats = full_report(window, n_boot=n_boot)
        folds.append({
            "fold": len(folds) + 1,
            "train_end": df.index[start - 1],
            "test_start": window.index[0],
            "test_end": window.index[-1],
            "total_return": stats["total_return"],
            "sharpe_ratio": stats["sharpe_ratio"],
            "max_drawdown": stats["max_drawdown"],
            "exposure": stats["exposure"],
            "num_trades": stats["num_trades"],
        })
        oos_windows.append(window)
        start += test_days

    fold_table = pd.DataFrame(folds)
    stitched_result = pd.concat(oos_windows)
    sharpes = fold_table["sharpe_ratio"]

    return {
        "folds": fold_table,
        "n_folds": len(fold_table),
        "oos_equity": (1 + stitched_result["strategy_return"].fillna(0)).cumprod(),
        "oos_returns": stitched_result["strategy_return"].fillna(0),
        "oos_report": full_report(stitched_result, n_boot=n_boot),
        # Median and pct-positive are reported alongside the mean since
        # consistency across folds matters as much as the average.
        "mean_sharpe": float(sharpes.mean()),
        "median_sharpe": float(sharpes.median()),
        "sharpe_std": float(sharpes.std()),
        "pct_folds_positive": float((sharpes > 0).mean()),
        "worst_fold_sharpe": float(sharpes.min()),
    }


def rolling_walk_forward(df: pd.DataFrame, strategy_fn, train_days: int = 756,
                         test_days: int = 126, cost_bps: float = 5.0,
                         n_boot: int = 0, refit_per_fold: bool = True,
                         **strategy_params) -> dict:
    """
    Slides a train/test window through history and evaluates each
    out-of-sample block separately.

    refit_per_fold=True (default): each fold calls `strategy_fn` fresh on
    just that fold's own train_days-length window immediately preceding
    its test window -- a genuine rolling re-fit -- and only that fold's
    own model generates its test-period signal. refit_per_fold=False
    keeps the original behavior: `strategy_fn` runs ONCE on the whole
    series, and every fold's test window is sliced out of that one
    continuous backtest. That's correct for a pure rule (sma_crossover,
    momentum, mean_reversion), but for a FITTED strategy (ml_direction,
    or any adaptive wrapper that learns a choice from an early fraction
    of history) it means later folds are judged on a model that, from a
    live-trading standpoint, could not have existed yet at that fold's
    train_end -- an optimistic approximation, not a walk-forward result.

    For a strategy that accepts `train_frac` or `learn_frac`
    (ml_direction, ml_regime_conditional, and the adaptive wrappers that
    auto-select), refit_per_fold overrides whichever parameter the
    strategy has to train_days / (train_days + test_days), so that fold's
    internal fit boundary lines up exactly with its own train/test split.
    A caller-supplied train_frac/learn_frac in **strategy_params is
    overridden in that case -- the whole point of refit_per_fold is that
    the fit boundary IS the fold boundary, so an independent value would
    silently break the alignment it's meant to guarantee. Strategies
    without such a parameter (the plain rule-based ones) are unaffected:
    they have no fit step, so recomputing them on the fold's own window
    and reading off the test tail is already the correct causal behavior.

    mean_sharpe reflects whichever mode refit_per_fold selects.
    mean_sharpe_fixed_model is always the single-fit variant's number:
    when refit_per_fold=True, the (more expensive) fixed-model pass is
    also run so the two can be compared in one call, showing exactly how
    much the old behavior overstated a fitted strategy's performance;
    when refit_per_fold=False, no second pass runs and
    mean_sharpe_fixed_model trivially equals mean_sharpe, since that IS
    the fixed-model result. Note that refit_per_fold=True is itself
    already n_folds times more expensive than a single fit, before even
    counting the comparison pass -- worth knowing before wiring this into
    a page that reruns on every widget change.

    Warm-up note: no strategy in this project leaves a literal NaN in its
    signal during warm-up -- the rule-based ones resolve it to 0/flat
    because pandas comparisons against NaN evaluate to False, and
    ml_direction_signal explicitly drops warm-up rows before predicting
    and defaults its signal to 0 everywhere else. run_backtest()'s own
    signal.fillna(0) is a further backstop regardless. What refit_per_fold
    changes is which window that warm-up is computed OVER: each fold
    recomputes it from just that fold's own train_days-length slice
    rather than reusing the full series' already-warmed-up trailing
    history. Under this function's defaults (train_days=756), every
    strategy's own warm-up (200 days for sma_crossover, the longest of
    any strategy in STRATEGIES) is fully absorbed inside train_slice, so
    it affects zero days of any fold's test window; it would only start
    eating into the test window if train_days were reduced below a
    strategy's own lookback requirement, by
    max(0, warmup_days - train_days) days per fold, capped at test_days.

    `n_boot` is forwarded to full_report() for each fold; it defaults to
    0 (no Sharpe CI), since this function's own per-fold loop multiplies
    whatever full_report costs.
    """
    if len(df) < train_days + test_days:
        raise ValueError(
            f"Need at least {train_days + test_days} rows for a {train_days}/{test_days} "
            f"walk-forward; got {len(df)}. Shorten the windows or widen the date range."
        )

    fold_frac = train_days / (train_days + test_days)
    accepted_params = set(inspect.signature(strategy_fn).parameters)
    fold_override = {
        name: fold_frac for name in ("train_frac", "learn_frac") if name in accepted_params
    }

    if refit_per_fold:
        primary = _rolling_folds(df, strategy_fn, train_days, test_days, cost_bps, n_boot,
                                 True, fold_override, strategy_params)
        fixed = _rolling_folds(df, strategy_fn, train_days, test_days, cost_bps, n_boot,
                               False, fold_override, strategy_params)
        mean_sharpe_fixed_model = fixed["mean_sharpe"]
        fitted_note = (
            "Each fold refits on its own train_days-length window and predicts only its "
            "own test window. mean_sharpe_fixed_model shows what the old single-fit "
            "behavior (refit_per_fold=False) would have reported instead."
        )
    else:
        primary = _rolling_folds(df, strategy_fn, train_days, test_days, cost_bps, n_boot,
                                 False, fold_override, strategy_params)
        mean_sharpe_fixed_model = primary["mean_sharpe"]
        fitted_note = (
            "Signal generated once on the full series, then evaluated in rolling "
            "out-of-sample folds. Fitted strategies are not refit per fold -- pass "
            "refit_per_fold=True to see the honest (usually lower) walk-forward number."
        )

    return {
        **primary,
        "mean_sharpe_fixed_model": mean_sharpe_fixed_model,
        "refit_per_fold": refit_per_fold,
        "fitted_note": fitted_note,
    }


def evaluate_with_regimes(df: pd.DataFrame, strategy_fn, regimes, split_frac: float = 0.7,
                          cost_bps: float = 5.0, n_boot: int = 0,
                          strategy_params: dict | None = None) -> dict:
    """
    Walk-forward validation with the in-sample and out-of-sample results
    each broken down by market regime, so a Sharpe drop can be attributed
    to either a shift in regime mix or genuine per-regime decay -- compare
    `regime_mix` between the two periods first. `strategy_params` is an
    explicit dict rather than **kwargs because the adaptive wrappers take
    a `regimes` argument themselves, which would collide with this
    function's own `regimes` parameter under **kwargs. `n_boot` is
    forwarded to full_report(); it defaults to 0 (no Sharpe CI) since
    this is often called once per strategy in a comparison loop.
    """
    labels = regimes.labels if hasattr(regimes, "labels") else regimes
    names = getattr(regimes, "names", None)

    signal = strategy_fn(df, **(strategy_params or {}))
    result = run_backtest(df, signal, cost_bps=cost_bps, regimes=labels)

    split_idx = int(len(result) * split_frac)
    split_date = result.index[split_idx]
    in_sample, out_sample = result.loc[:split_date], result.loc[split_date:]

    def regime_mix(segment: pd.DataFrame) -> pd.Series:
        valid = segment.loc[segment["regime"] != -1, "regime"]
        if len(valid):
            return valid.value_counts(normalize=True).sort_index()
        return pd.Series(dtype=float)

    return {
        "split_date": str(split_date.date()),
        "in_sample": full_report(in_sample, n_boot=n_boot),
        "out_sample": full_report(out_sample, n_boot=n_boot),
        "in_sample_by_regime": performance_by_regime(in_sample, labels, names),
        "out_sample_by_regime": performance_by_regime(out_sample, labels, names),
        "regime_mix": pd.DataFrame({
            "in_sample": regime_mix(in_sample),
            "out_sample": regime_mix(out_sample),
        }).fillna(0.0),
        "names": names or {},
    }


def compare_strategies(df: pd.DataFrame, strategies: dict, split_frac: float = 0.7,
                       cost_bps: float = 5.0, n_boot: int = 0) -> pd.DataFrame:
    """
    Runs several strategies over identical data, dates, costs and split,
    and returns one table of in-sample vs out-of-sample metrics.
    `strategies` maps a display name to either a callable or a
    (callable, params_dict) tuple. Selecting the best out-of-sample
    Sharpe from this table is itself an in-sample result, since the
    out-of-sample data was used to choose it -- report the whole table.
    `n_boot` is forwarded to full_report() for every strategy; it
    defaults to 0 (no Sharpe CI), since this function calls full_report
    twice per strategy.
    """
    rows = []
    for name, entry in strategies.items():
        fn, params = entry if isinstance(entry, tuple) else (entry, {})
        try:
            signal = fn(df, **params)
            result = run_backtest(df, signal, cost_bps=cost_bps)
            split_date = result.index[int(len(result) * split_frac)]
            in_stats = full_report(result.loc[:split_date], n_boot=n_boot)
            out_stats = full_report(result.loc[split_date:], n_boot=n_boot)
            rows.append({
                "strategy": name,
                "is_sharpe": in_stats["sharpe_ratio"],
                "oos_sharpe": out_stats["sharpe_ratio"],
                "sharpe_decay": in_stats["sharpe_ratio"] - out_stats["sharpe_ratio"],
                "oos_return": out_stats["total_return"],
                "oos_max_dd": out_stats["max_drawdown"],
                "oos_exposure": out_stats["exposure"],
                "turnover": out_stats["turnover"],
                "error": None,
            })
        except Exception as exc:  # keep one broken strategy from killing the table
            rows.append({
                "strategy": name, "is_sharpe": np.nan, "oos_sharpe": np.nan,
                "sharpe_decay": np.nan, "oos_return": np.nan, "oos_max_dd": np.nan,
                "oos_exposure": np.nan, "turnover": np.nan, "error": f"{type(exc).__name__}: {exc}",
            })

    return pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False, na_position="last")
