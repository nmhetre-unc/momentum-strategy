"""
C1 baseline report for monthly SPY prediction (qbt.ml): dataset sizes,
purged walk-forward folds, baseline accuracy and the buy-and-hold
trading comparison, for both targets (next-month direction and regime),
with and without the HMM regime features. Writes everything to
results/ml/c1_baselines.json; any number quoted from C1 should come
from that file.

Also recorded:
- buy-and-hold verification: the trading numbers cover exactly the
  out-of-sample test months, checked against a direct computation from
  closing prices;
- how much of the next-month regime label is already determined at the
  decision date (the label is not changed; see qbt.ml.labels);
- how often the HMM behind the point-in-time features is refit, and the
  wall time of building each dataset.

Timings and the environment block change from run to run; every other
number is deterministic for a given cache file.

Uses the 17-year cached SPY file (2008-01-01 to 2025-01-01), the same
one as scripts/regenerate_results.py.

Run with: python scripts/ml_c1_report.py
"""

import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qbt.data import CACHE_DIR, fetch_ohlcv
from qbt.ml.evaluate import (
    MonthlyDataset,
    build_dataset,
    evaluate_baselines,
    strategy_returns,
    trading_summary,
)
from qbt.ml.features import HMM_STATES
from qbt.ml.splits import EMBARGO_MONTHS, Fold, walk_forward_splits
from qbt.regime import detect_regimes_walk_forward

TICKER, START, END = "SPY", "2008-01-01", "2025-01-01"
MIN_TRAIN = 60       # months of purged training data before the first test block
TEST_SIZE = 24       # months per test block
EMBARGO = EMBARGO_MONTHS
COST_BPS = 5.0
TREND_WINDOW, VOL_WINDOW = 60, 20   # the `rules` regime's inputs (qbt.regime_features)
OUT = Path(__file__).resolve().parents[1] / "results" / "ml" / "c1_baselines.json"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat() if value == value.normalize() else value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        return None if np.isnan(value) else float(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame.index.name is not None:
        frame = frame.reset_index()
    return [_jsonable(row) for row in frame.to_dict(orient="records")]


def _environment() -> dict:
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text()
                .splitlines() if line.startswith("model name")), platform.processor())
    return {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu_model": cpu,
        "logical_cpus": os.cpu_count(),
        "loadavg_1_5_15_at_start": list(os.getloadavg()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _data_block(df: pd.DataFrame) -> dict:
    cache_file = Path(CACHE_DIR) / f"{TICKER}_{START}_{END}.csv"
    return {
        "ticker": TICKER,
        "requested_range": [START, END],
        "first_date": df.index[0],
        "last_date": df.index[-1],
        "n_trading_days": len(df),
        "cache_file": f"data_cache/{cache_file.name}",
        "cache_sha256": hashlib.sha256(cache_file.read_bytes()).hexdigest(),
    }


def _buy_and_hold_check(df: pd.DataFrame, ds: MonthlyDataset, folds: list[Fold],
                        reported: dict) -> dict:
    """
    Recomputes buy-and-hold over the test months straight from closing
    prices and asserts it matches evaluate_baselines(); also gives the
    full-sample figure, for contrast only.
    """
    tested = np.concatenate([f.test for f in folds])
    t0 = ds.t0.iloc[tested]
    first_t0, last_t1 = t0.iloc[0], ds.t1.iloc[tested[-1]]
    close = df["Close"]
    gross_from_prices = float(close[last_t1] / close[first_t0] - 1)
    fwd = ds.forward_return.iloc[tested]
    assert np.isclose(float(np.prod(1 + fwd) - 1), gross_from_prices, rtol=1e-12, atol=0)
    assert (ds.t1.iloc[tested[:-1]].to_numpy() == t0.iloc[1:].to_numpy()).all(), \
        "test months must be contiguous"

    held = pd.Series(1.0, index=fwd.index)
    recomputed = trading_summary(strategy_returns(held, fwd, COST_BPS), held)
    for key in ("n_months", "total_return", "ann_return", "sharpe"):
        assert np.isclose(recomputed[key], reported[key], rtol=1e-12, atol=0), key

    everything = pd.Series(1.0, index=ds.forward_return.index)
    full = trading_summary(strategy_returns(everything, ds.forward_return, COST_BPS), everything)
    return {
        "statement": (
            f"buy_and_hold in `trading` is computed over exactly the {len(tested)} "
            f"out-of-sample test months: held from the close of {first_t0.date()} (first test "
            f"decision) to the close of {last_t1.date()} (last test label end), net of one "
            f"{COST_BPS:g} bps entry cost. It does not include the warm-up or the first "
            f"{len(folds[0].train) + len(folds[0].purged)} training months. Checked against "
            "a direct computation from closing prices."
        ),
        "n_test_months": len(tested),
        "first_test_decision": first_t0,
        "last_test_label_end": last_t1,
        "gross_return_from_prices": gross_from_prices,
        "net_total_return": recomputed["total_return"],
        "ann_return": recomputed["ann_return"],
        "sharpe": recomputed["sharpe"],
        "for_contrast_full_sample": {
            "n_months": len(ds.forward_return),
            "first_decision": ds.t0.iloc[0],
            "ann_return": full["ann_return"],
            "sharpe": full["sharpe"],
        },
    }


def _regime_label_determinacy(df: pd.DataFrame, ds: MonthlyDataset,
                              folds: list[Fold]) -> dict:
    """
    How much of the next-month regime label is already known at t.
    Agreement rates are empirical; the window fractions are structural
    (they depend only on where each label day falls in month t+1).
    """
    tested = np.concatenate([f.test for f in folds])
    current = ds.X["regime_rules"]           # daily rules regime on the decision date
    agree_daily = current == ds.y
    agree_month = ds.previous == ds.y        # modal rules regime of month t

    trend_known, vol_known = [], []
    days = pd.DatetimeIndex(df.index)
    for t0, t1 in zip(ds.t0, ds.t1, strict=True):
        k = np.arange(1, ((days > t0) & (days <= t1)).sum() + 1)  # day k of month t+1
        trend_known.append(np.clip((TREND_WINDOW - k) / TREND_WINDOW, 0, None).mean())
        vol_known.append(np.clip((VOL_WINDOW - k) / VOL_WINDOW, 0, None).mean())

    return {
        "note": (
            "The next-month regime label is the modal daily `rules` regime over (t, t_next]. "
            f"Each of those days classifies on a {TREND_WINDOW}-day trend and a {VOL_WINDOW}-day "
            "volatility ranked against its expanding history, so on day k of month t+1, "
            f"{TREND_WINDOW}-k of the trend's daily returns and {VOL_WINDOW}-k of the "
            "volatility's were already known at the decision date. The agreement rates show "
            "how often the label simply equals the regime already observable at t; a model "
            "must beat these, not the majority class, to show it predicts regime changes."
        ),
        "n_samples_all": len(ds.y),
        "n_samples_oos": len(tested),
        "label_equals_rules_regime_on_decision_date": {
            "all_samples": float(agree_daily.mean()),
            "oos_test_months": float(agree_daily.iloc[tested].mean()),
        },
        "label_equals_modal_regime_of_month_t": {
            "all_samples": float(agree_month.mean()),
            "oos_test_months": float(agree_month.iloc[tested].mean()),
        },
        "mean_share_of_trend_window_known_at_t": float(np.mean(trend_known)),
        "mean_share_of_vol_window_known_at_t": float(np.mean(vol_known)),
    }


def _hmm_refits(df: pd.DataFrame) -> dict:
    """The HMM run behind the hmm_* features, with qbt.ml.features' defaults."""
    started = time.perf_counter()
    result = detect_regimes_walk_forward(df, method="hmm", n_regimes=HMM_STATES)
    seconds = time.perf_counter() - started
    refits = result.meta["refit_dates"]
    return {
        "method": "detect_regimes_walk_forward(method='hmm', n_regimes=3), library defaults",
        "initial_train_rows": result.meta["initial_train"],
        "refit_every_trading_days": result.meta["refit_every"],
        "n_refits": result.meta["n_refits"],
        "first_refit_train_end": refits[0],
        "last_refit_train_end": refits[-1],
        "first_labelled_day": result.meta["first_label"],
        "note": (
            "Expanding window: each refit fits on every day up to its train end, then labels "
            "the next refit_every trading days with the forward filter. That is about one refit "
            "per quarter, so each refit serves about three monthly decisions."
        ),
        "wall_seconds": seconds,
    }


def main() -> None:
    environment = _environment()
    df = fetch_ohlcv(TICKER, START, END)
    report: dict = {
        "report": "scripts/ml_c1_report.py",
        "data": _data_block(df),
        "parameters": {
            "decision_dates": "last trading day of each complete calendar month",
            "min_train": MIN_TRAIN, "test_size": TEST_SIZE, "embargo": EMBARGO,
            "embargo_note": "walk-forward has no training data after a test block, so the "
                            "embargo removes nothing here; purging removes one boundary "
                            "month per fold",
            "cost_bps": COST_BPS,
            "accuracy_interval": "Wilson 95%",
            "mcnemar": "exact, two-sided, previous_label vs majority_class",
        },
        "runs": [],
        "environment": environment,
    }

    timings = []
    for include_hmm in (False, True):
        for target in ("direction", "regime"):
            started = time.perf_counter()
            ds = build_dataset(df, target=target, include_hmm=include_hmm)
            build_seconds = time.perf_counter() - started
            timings.append({"target": target, "include_hmm": include_hmm,
                            "build_dataset_wall_seconds": build_seconds})

            folds = walk_forward_splits(ds.t0, ds.t1, min_train=MIN_TRAIN,
                                        test_size=TEST_SIZE, embargo=EMBARGO)
            rep = evaluate_baselines(ds, folds, cost_bps=COST_BPS)
            run: dict = {
                "target": target,
                "include_hmm": include_hmm,
                "n_samples": len(ds.y),
                "first_decision": ds.t0.iloc[0],
                "last_decision": ds.t0.iloc[-1],
                "label_balance": {str(k): float(v) for k, v in
                                  ds.y.value_counts(normalize=True).sort_index().items()},
                "n_folds": len(folds),
                "n_oos_months": int(rep.folds["n_test"].sum()),
                "folds": _records(rep.folds),
                "summary": _records(rep.summary),
                "mcnemar_p_previous_vs_majority": rep.mcnemar_p,
            }
            if rep.trading is not None:
                run["trading"] = _records(rep.trading.astype(float).rename_axis("strategy"))
                run["buy_and_hold_check"] = _buy_and_hold_check(
                    df, ds, folds, rep.trading.loc["buy_and_hold"].astype(float).to_dict())
            else:
                run["regime_label_determinacy"] = _regime_label_determinacy(df, ds, folds)
            report["runs"].append(run)
            print(f"{target:9s} hmm={include_hmm!s:5s} n={len(ds.y)} "
                  f"folds={len(folds)} build={build_seconds:.1f}s")

    report["hmm_features"] = {**_hmm_refits(df), "build_dataset_timings": timings}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    print(f"wrote {OUT.relative_to(OUT.parents[2])}")


if __name__ == "__main__":
    main()
