"""
Checks whether a strategy's performance holds up on data it wasn't
fitted on. Three entry points, in increasing rigor: evaluate_out_of_sample()
(one split), rolling_walk_forward() (many consecutive out-of-sample
folds), and evaluate_with_regimes() (the out-of-sample result, split by
market regime).
"""

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


def rolling_walk_forward(df: pd.DataFrame, strategy_fn, train_days: int = 756,
                         test_days: int = 126, cost_bps: float = 5.0,
                         n_boot: int = 0, **strategy_params) -> dict:
    """
    Slides a train/test window through history and evaluates each
    out-of-sample block separately. The signal is generated once on the
    full series, then sliced: correct for the rule-based strategies, but
    an approximation for fitted strategies (ml_direction, and the adaptive
    wrappers), which fit once rather than refitting per fold -- the
    returned `fitted_note` states this. Returns per-fold metrics plus a
    stitched equity curve of only the out-of-sample days. `n_boot` is
    forwarded to full_report() for each fold; it defaults to 0 (no
    Sharpe CI), since this function's own per-fold loop multiplies
    whatever full_report costs.
    """
    signal = strategy_fn(df, **strategy_params)
    result = run_backtest(df, signal, cost_bps=cost_bps)

    if len(result) < train_days + test_days:
        raise ValueError(
            f"Need at least {train_days + test_days} rows for a {train_days}/{test_days} "
            f"walk-forward; got {len(result)}. Shorten the windows or widen the date range."
        )

    folds: list[dict] = []
    oos_returns = []
    start = train_days
    while start + test_days <= len(result):
        window = result.iloc[start:start + test_days]
        stats = full_report(window, n_boot=n_boot)
        folds.append({
            "fold": len(folds) + 1,
            "train_end": result.index[start - 1],
            "test_start": window.index[0],
            "test_end": window.index[-1],
            "total_return": stats["total_return"],
            "sharpe_ratio": stats["sharpe_ratio"],
            "max_drawdown": stats["max_drawdown"],
            "exposure": stats["exposure"],
            "num_trades": stats["num_trades"],
        })
        oos_returns.append(window["strategy_return"].fillna(0))
        start += test_days

    fold_table = pd.DataFrame(folds)
    stitched = pd.concat(oos_returns)
    sharpes = fold_table["sharpe_ratio"]

    return {
        "folds": fold_table,
        "n_folds": len(fold_table),
        "oos_equity": (1 + stitched).cumprod(),
        "oos_returns": stitched,
        "oos_report": full_report(result.loc[stitched.index], n_boot=n_boot),
        # Median and pct-positive are reported alongside the mean since
        # consistency across folds matters as much as the average.
        "mean_sharpe": float(sharpes.mean()),
        "median_sharpe": float(sharpes.median()),
        "sharpe_std": float(sharpes.std()),
        "pct_folds_positive": float((sharpes > 0).mean()),
        "worst_fold_sharpe": float(sharpes.min()),
        "fitted_note": (
            "Signal generated once on the full series, then evaluated in rolling "
            "out-of-sample folds. Fitted strategies are not refit per fold."
        ),
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
