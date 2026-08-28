"""
Checks whether a strategy's performance holds up on data it wasn't
"tuned" on -- the single biggest thing separating a rigorous backtest
from one that's secretly overfit to one lucky period.

Three levels of rigour live here, in increasing order:

    evaluate_out_of_sample()  one split. Cheap, and enough to catch
                              blatant overfitting.
    rolling_walk_forward()    many consecutive out-of-sample folds.
                              Catches the subtler failure a single split
                              misses -- a strategy that works until 2018
                              and never again, whose one 30% holdout
                              happened to land in a friendly stretch.
    evaluate_with_regimes()   the out-of-sample result, split by market
                              regime. Answers *where* it broke, not just
                              *that* it broke.
"""

import numpy as np
import pandas as pd

from backtest import run_backtest
from analytics import full_report, performance_by_regime


def evaluate_out_of_sample(df: pd.DataFrame, strategy_fn, split_frac: float = 0.7, **strategy_params) -> dict:
    """
    Computes the signal and backtest ONCE on the full dataset (so rolling
    windows have full history available from the start), then splits the
    result chronologically: the first `split_frac` of days is "in-sample,"
    the rest is "out-of-sample." Reports full metrics for each separately.

    A strategy whose Sharpe ratio collapses (or flips negative) out-of-sample
    is a strategy that was fit to noise in the in-sample period, not one
    that captures something real.
    """
    signal = strategy_fn(df, **strategy_params)
    result = run_backtest(df, signal)

    split_idx = int(len(result) * split_frac)
    split_date = result.index[split_idx]

    in_sample = result.loc[:split_date]
    out_sample = result.loc[split_date:]

    return {
        "split_date": str(split_date.date()),
        "in_sample": full_report(in_sample),
        "out_sample": full_report(out_sample),
    }


def rolling_walk_forward(df: pd.DataFrame, strategy_fn, train_days: int = 756,
                         test_days: int = 126, cost_bps: float = 0.0,
                         **strategy_params) -> dict:
    """
    Slides a train/test window through history and evaluates each
    out-of-sample block separately.

    Why bother when evaluate_out_of_sample() already exists: one split
    gives you ONE out-of-sample number, and one number cannot distinguish
    "this works" from "this got lucky in one three-year window." Ten
    consecutive out-of-sample folds give you a distribution. What you
    want to see is most blocks positive and the bad ones survivable. What
    you usually see, on a strategy that looked great on a single split,
    is two spectacular blocks and eight mediocre ones -- meaning the
    headline Sharpe belongs to a specific market episode, not to the
    strategy.

    Note on the strategies in this project: the signal is generated once
    on the full series, then sliced. That's correct for the rule-based
    strategies, whose parameters are fixed and whose rolling windows are
    already backward-looking. For fitted strategies (ml_direction, and
    the auto-selecting wrappers in adaptive.py) it is an approximation --
    they fit once on their own internal split rather than refitting per
    fold. A production framework would refit every fold; the honest
    description of this one is "fixed model, rolling evaluation", and the
    `fitted_note` in the returned dict says so.

    Returns per-fold metrics plus a stitched equity curve made only of
    out-of-sample days -- the closest thing here to a paper-trading record.
    """
    signal = strategy_fn(df, **strategy_params)
    result = run_backtest(df, signal, cost_bps=cost_bps)

    if len(result) < train_days + test_days:
        raise ValueError(
            f"Need at least {train_days + test_days} rows for a {train_days}/{test_days} "
            f"walk-forward; got {len(result)}. Shorten the windows or widen the date range."
        )

    folds, oos_returns = [], []
    start = train_days
    while start + test_days <= len(result):
        window = result.iloc[start:start + test_days]
        stats = full_report(window)
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
        "oos_report": full_report(result.loc[stitched.index]),
        # Consistency matters more than the average. A strategy with mean
        # Sharpe 0.4 across 12 folds, 10 of them positive, is a far better
        # bet than one averaging 0.8 off two enormous folds.
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
                          cost_bps: float = 0.0, strategy_params: dict = None) -> dict:
    """
    Walk-forward validation, with the in-sample and out-of-sample results
    each broken down by market regime.

    This is the diagnostic that turns "it stopped working" into something
    actionable. There are two very different stories behind a Sharpe that
    halves out-of-sample:

      (a) per-regime performance is unchanged, but the MIX of regimes
          shifted -- the out-of-sample period simply contained more of
          the regime this strategy dislikes. The strategy is intact; your
          expectations were built on a biased sample of history.

      (b) per-regime performance itself deteriorated -- it now loses
          money in the regime it used to profit from. That is real decay,
          and no amount of regime timing fixes it.

    Compare the `regime_mix` shares between the two periods first. That
    single comparison usually settles which story you're in.

    `strategy_params` is an explicit dict rather than **kwargs on purpose:
    the adaptive wrappers themselves take a `regimes` argument, and with
    **kwargs that would collide with this function's own `regimes`
    parameter the moment you evaluated one of them.
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
        return valid.value_counts(normalize=True).sort_index() if len(valid) else pd.Series(dtype=float)

    return {
        "split_date": str(split_date.date()),
        "in_sample": full_report(in_sample),
        "out_sample": full_report(out_sample),
        "in_sample_by_regime": performance_by_regime(in_sample, labels, names),
        "out_sample_by_regime": performance_by_regime(out_sample, labels, names),
        "regime_mix": pd.DataFrame({
            "in_sample": regime_mix(in_sample),
            "out_sample": regime_mix(out_sample),
        }).fillna(0.0),
        "names": names or {},
    }


def compare_strategies(df: pd.DataFrame, strategies: dict, split_frac: float = 0.7,
                       cost_bps: float = 0.0) -> pd.DataFrame:
    """
    Runs several strategies over identical data and returns one table of
    in-sample vs out-of-sample metrics.

    Comparing strategies fairly is harder than it looks, and this
    function exists to remove the easy mistakes: same ticker, same dates,
    same costs, same split, same metrics, all computed the same way. What
    it cannot remove is selection bias -- if you compare twenty
    strategies and report the best one's out-of-sample Sharpe, that
    number is itself an in-sample result, because you used the
    out-of-sample data to choose. The honest report is the whole table.

    `strategies` maps a display name to either a callable or a
    (callable, params_dict) tuple.
    """
    rows = []
    for name, entry in strategies.items():
        fn, params = entry if isinstance(entry, tuple) else (entry, {})
        try:
            signal = fn(df, **params)
            result = run_backtest(df, signal, cost_bps=cost_bps)
            split_date = result.index[int(len(result) * split_frac)]
            in_stats = full_report(result.loc[:split_date])
            out_stats = full_report(result.loc[split_date:])
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
