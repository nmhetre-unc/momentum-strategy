"""
Audit: is any strategy scored on data it learned from?

For every strategy in qbt.adaptive.ALL_STRATEGIES on SPY 2008-2025,
records what it learns from history (a fitted model, a regime model, a
choice selected on past performance) with file:line references into the
code being measured, which rows that makes in-sample, and how much of
the in-sample period it trades and gets scored on:

- first_out_of_sample_day: the first day after everything the strategy
  learned from, computed here from first principles (the ML split, the
  regime model's causal_from, the day after learn_end), not read back
  from the strategy.
- in_sample_rows: rows before that day. in_sample_days_with_position:
  how many of them the strategy holds a position on. Full-period metrics
  score every one of them.
- evaluate_out_of_sample_in_sample_days: days in evaluate_out_of_sample's
  out-of-sample half that are before first_out_of_sample_day.
- rolling_fixed_model_folds_scored_in_sample: rolling walk-forward folds
  (single whole-series fit) scored although their test window starts
  before first_out_of_sample_day.
- days_changed_by_later_prices: positions before 2022-01-03 that change
  when every price from that date on is rewritten. Nonzero means a
  position used later data (lookahead), which is worse than in-sample.

Run on the code to measure:
    python scripts/audit_learning_strategies.py --section after
writes that section of results/ml/learning_strategies_audit.json and
keeps the other. "after" asserts every in-sample count is 0; "before"
asserts the leaks the fix addresses were there.
"""

import argparse
import inspect
import json
import subprocess
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

import qbt.adaptive as adaptive
import qbt.ml as ml
import qbt.regime as regime
from qbt.data import fetch_ohlcv
from qbt.timesplit import split_date
from qbt.walk_forward import evaluate_out_of_sample, rolling_walk_forward

TICKER, START, END = "SPY", "2008-01-01", "2025-01-01"
TRAIN_FRAC, LEARN_FRAC, REGIME_FIT_FRAC = 0.7, 0.6, 0.7
ROLLING_TRAIN_DAYS, ROLLING_TEST_DAYS = 756, 126
REWRITE_FROM = "2022-01-03"
OUT = Path(__file__).resolve().parents[1] / "results" / "ml" / "learning_strategies_audit.json"

LEARNS_A_CHOICE = ("regime_filtered", "regime_switch", "regime_sized", "adaptive_ensemble")
REGIME_WRAPPERS = (*LEARNS_A_CHOICE, "regime_parameters")
ML = ("ml_direction", "ml_regime_conditional")


def _where(func, *snippets: str) -> str:
    """file:line of the first of `snippets` found in `func`'s source."""
    lines, first = inspect.getsourcelines(func)
    path = Path(inspect.getsourcefile(func) or "")
    shown = "/".join(path.parts[path.parts.index("src") + 1:]) if "src" in path.parts else path.name
    for snippet in snippets:
        for offset, line in enumerate(lines):
            if snippet in line:
                return f"{shown}:{first + offset}"
    raise LookupError(f"none of {snippets} in {func.__name__}")


def _returns_line(func) -> str:
    """file:line of `func`'s final return: where its positions leave it."""
    lines, first = inspect.getsourcelines(func)
    last = max(i for i, line in enumerate(lines) if line.strip().startswith("return"))
    path = Path(inspect.getsourcefile(func) or "")
    return f"{'/'.join(path.parts[path.parts.index('src') + 1:])}:{first + last}"


def _learning(name: str, df: pd.DataFrame, regime_fit_end, learn_end) -> list[dict]:
    hmm = {"what": "HMM regime model (fit on the first 70% of post-warm-up rows)",
           "where": _where(regime.detect_regimes, "fit_index = X.index[:split_idx]"),
           "learns_from_rows_through": regime_fit_end.date().isoformat()}
    choice = {"what": "", "where": "", "learns_from_rows_through": learn_end.date().isoformat()}
    if name == "regime_filtered":
        return [hmm, {**choice, "what": "allow-list: regimes where the base had Sharpe > 0",
                      "where": _where(adaptive.regime_filtered, "_regime_sharpe_table(")}]
    if name == "regime_switch":
        return [hmm, {**choice, "what": "strategy per regime: best learning-window Sharpe",
                      "where": _where(adaptive.regime_switch, "_regime_sharpe_table(")}]
    if name == "adaptive_ensemble":
        return [hmm, {**choice, "what": "strategy per regime: best learning-window Sharpe",
                      "where": _where(adaptive.adaptive_ensemble, "_regime_sharpe_table(")}]
    if name == "regime_sized":
        return [hmm, {**choice, "what": "size per regime: inverse of that regime's volatility",
                      "where": _where(adaptive.regime_sized, ".std()")}]
    if name == "regime_parameters":
        return [hmm]
    if name in ML:
        ml_split = split_date(df.index, TRAIN_FRAC)
        model = {"what": "logistic regression on next-day direction",
                 "where": _where(ml._split, "train_mask = "),
                 "learns_from_rows_through":
                     df.index[df.index < ml_split][-1].date().isoformat()}
        if name == "ml_direction":
            return [model]
        return [model, {**hmm, "what": "HMM regime model, fit capped before the ML split",
                        "where": _where(adaptive.ml_regime_conditional, "_resolve_regimes("),
                        "learns_from_rows_through":
                            df.index[df.index < ml_split][-1].date().isoformat()}]
    return []


def _first_out_of_sample_day(name: str, df: pd.DataFrame, regime_oos, learn_end):
    if name in ML:
        return split_date(df.index, TRAIN_FRAC)
    if name in REGIME_WRAPPERS:
        after_learning = df.index[df.index > learn_end][0]
        return max(regime_oos, after_learning) if name in LEARNS_A_CHOICE else regime_oos
    return None


def _rewrite_from(df: pd.DataFrame, date: str) -> pd.DataFrame:
    out = df.copy()
    later = out.index >= pd.Timestamp(date)
    returns = out["Close"].pct_change().to_numpy()
    start = out.loc[~later, "Close"].iloc[-1]
    out.loc[later, "Close"] = start * np.cumprod(1 - 3 * returns[later])
    return out


def measure(source: str) -> dict:
    warnings.filterwarnings("ignore")
    df = fetch_ohlcv(TICKER, START, END)
    rewritten = _rewrite_from(df, REWRITE_FROM)
    detected = regime.detect_regimes(df, fit_frac=REGIME_FIT_FRAC)
    regime_fit_end = pd.Timestamp(str(detected.fit_end))
    regime_oos = pd.Timestamp(str(detected.causal_from))
    learn_end = pd.Timestamp(adaptive._learn_cutoff(df, LEARN_FRAC))
    shared_split = split_date(df.index, TRAIN_FRAC)

    strategies = {}
    for name, strategy in adaptive.ALL_STRATEGIES.items():
        fn = cast(Callable[..., pd.Series], strategy)
        oos = _first_out_of_sample_day(name, df, regime_oos, learn_end)
        signal = fn(df).reindex(df.index).fillna(0.0).astype(float)
        changed = fn(rewritten).reindex(df.index).fillna(0.0).astype(float)
        before_cut = df.index < pd.Timestamp(REWRITE_FROM)
        entry: dict = {
            "learns": _learning(name, df, regime_fit_end, learn_end),
            "positions_returned_at": _returns_line(fn),
            "first_out_of_sample_day": None if oos is None else oos.date().isoformat(),
            "days_changed_by_later_prices":
                int((signal[before_cut] != changed[before_cut]).sum()),
        }
        if oos is not None:
            in_sample = df.index < oos
            evaluated = pd.Timestamp(evaluate_out_of_sample(df, fn)["split_date"])
            rolled = rolling_walk_forward(df, fn, train_days=ROLLING_TRAIN_DAYS,
                                          test_days=ROLLING_TEST_DAYS, refit_per_fold=False)
            folds = rolled["folds"]
            scored = (folds["can_hold_position"] if "can_hold_position" in folds
                      else pd.Series(True, index=folds.index))
            entry.update({
                "in_sample_rows": int(in_sample.sum()),
                "in_sample_first_last": [df.index[0].date().isoformat(),
                                         df.index[in_sample][-1].date().isoformat()],
                "in_sample_days_with_position": int((signal[in_sample] != 0).sum()),
                "evaluate_out_of_sample_split": evaluated.date().isoformat(),
                "evaluate_out_of_sample_in_sample_days":
                    int(((df.index >= evaluated) & in_sample).sum()),
                "rolling_fixed_model_folds": int(len(folds)),
                "rolling_fixed_model_folds_scored_in_sample":
                    int((scored & (folds["test_start"] < oos)).sum()),
            })
        strategies[name] = entry

    return {
        "source": source,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {"ticker": TICKER, "first_date": df.index[0].date().isoformat(),
                 "last_date": df.index[-1].date().isoformat(), "n_trading_days": len(df)},
        "windows": {"ml_split_first_out_of_sample_day": shared_split.date().isoformat(),
                    "regime_fit_end": regime_fit_end.date().isoformat(),
                    "regime_causal_from": regime_oos.date().isoformat(),
                    "learn_end": learn_end.date().isoformat(),
                    "rewrite_from": REWRITE_FROM},
        "strategies": strategies,
    }


def check(section: str, m: dict) -> None:
    learning = {k: v for k, v in m["strategies"].items() if v["first_out_of_sample_day"]}
    counts = ("in_sample_days_with_position", "evaluate_out_of_sample_in_sample_days",
              "rolling_fixed_model_folds_scored_in_sample")
    if section == "after":
        for name, v in learning.items():
            for key in counts:
                assert v[key] == 0, (name, key, v[key])
        for name, v in m["strategies"].items():
            assert v["days_changed_by_later_prices"] == 0, name
    else:
        for name in REGIME_WRAPPERS:
            v = m["strategies"][name]
            assert all(v[key] > 0 for key in counts), (name, v)
        assert m["strategies"]["regime_sized"]["days_changed_by_later_prices"] > 0
        assert all(m["strategies"][name]["in_sample_days_with_position"] == 0 for name in ML)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=OUT.parents[2]).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("before", "after"), required=True)
    parser.add_argument("--source", help="what code is being measured (default: git HEAD)")
    args = parser.parse_args()
    source = args.source or f"git {_git('rev-parse', '--short', 'HEAD')} + working tree"
    measured = measure(source)
    check(args.section, measured)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing["audit"] = __doc__.strip().splitlines()[0]
    existing[args.section] = measured
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(existing, indent=2) + "\n")
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()
