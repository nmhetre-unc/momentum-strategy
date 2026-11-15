"""
One backtest job as a pure function: a price frame and a request in,
a result frame and a flat metrics dict out. No database, no network,
no global state -- the service's workers call this, and so can tests
and the CLI, without either side knowing about the other.

Deterministic by construction: the strategies seed their own models
(random_state=42), and the bootstrap draws from a fresh
np.random.default_rng(seed). Every bootstrap implementation gives
bit-identical intervals, so which one ran (recorded in
RunOutput.bootstrap_impl) affects speed only, never the numbers. That
is what makes re-running a job after a worker crash safe.

Only the single-asset base strategies in qbt.strategies.STRATEGIES are
runnable here for now; the adaptive wrappers take non-scalar parameters
and get their own request models later.
"""

import inspect
import math
from dataclasses import dataclass

import pandas as pd

from qbt.analytics import full_report
from qbt.backtest import run_backtest
from qbt.bootstrap import resolve_impl
from qbt.strategies import STRATEGIES


@dataclass(frozen=True)
class RunOutput:
    result: pd.DataFrame          # run_backtest() output, plus a normalized `equity` column
    metrics: dict[str, float]     # full_report() keys; a key is absent if it wasn't computed
    bootstrap_impl: str | None    # the impl that ran, or None when n_boot == 0


def validate_prices(df: pd.DataFrame) -> None:
    """Raises ValueError unless `df` is a usable single-asset price frame."""
    if "Close" not in df.columns:
        raise ValueError("price frame has no 'Close' column")
    if len(df) < 2:
        raise ValueError(f"need at least 2 price bars, got {len(df)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("price frame must have a DatetimeIndex")
    if not (df.index.is_unique and df.index.is_monotonic_increasing):
        raise ValueError("price dates must be unique and increasing")
    if df["Close"].isna().any() or (df["Close"] <= 0).any():
        raise ValueError("Close must be positive with no missing values")


def run(df: pd.DataFrame, strategy: str, params: dict, cost_bps: float = 5.0,
        n_boot: int = 0, seed: int = 0, impl: str = "auto") -> RunOutput:
    """
    Runs `strategy` with `params` over all of `df` (the caller slices the
    date range; warm-up windows such as a 200-day SMA are consumed inside
    it, as in the CLI), then reports full_report() metrics with an
    `n_boot`-replicate stationary-bootstrap Sharpe interval.

    Raises ValueError for anything wrong with the inputs -- an unknown
    strategy or parameter, a bad price frame, a negative cost -- so a
    caller can tell a request that will fail every time from a transient
    failure worth retrying.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if n_boot < 0:
        raise ValueError(f"n_boot must be >= 0, got {n_boot}")
    validate_prices(df)

    strategy_fn = STRATEGIES[strategy]
    # Checked up front rather than by catching TypeError from the call,
    # which would also swallow a genuine bug inside the strategy.
    accepted = set(inspect.signature(strategy_fn).parameters) - {"df"}
    unknown = set(params) - accepted
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {sorted(unknown)} for {strategy!r}; "
            f"accepted: {sorted(accepted)}"
        )

    signal = strategy_fn(df, **params)

    result = run_backtest(df, signal, cost_bps=cost_bps)
    # The same normalized curve full_report() measures (day 0's NaN
    # return counts as flat), so stored equity and metrics agree.
    result["equity"] = (1 + result["strategy_return"].fillna(0)).cumprod()

    bootstrap_impl = resolve_impl(impl) if n_boot > 0 else None
    report = full_report(result, n_boot=n_boot, seed=seed,
                         impl=bootstrap_impl or impl)
    metrics = {key: float(value) for key, value in report.items() if value is not None}
    return RunOutput(result=result, metrics=metrics, bootstrap_impl=bootstrap_impl)


def metrics_equal(a: dict[str, float], b: dict[str, float]) -> bool:
    """Exact equality, treating NaN as equal to NaN (determinism checks)."""
    return a.keys() == b.keys() and all(
        a[k] == b[k] or (math.isnan(a[k]) and math.isnan(b[k])) for k in a
    )
