"""
Lookahead and overlap audit of the daily ML strategy (qbt.ml's
ml_direction_signal and the paths that call it), on the cached SPY
file. Writes one section of results/ml/daily_ml_audit.json:

    python scripts/audit_daily_ml.py --section after
    python scripts/audit_daily_ml.py --section before --source "<what code was measured>"

The "before" section was produced by running this same script against
the pre-fix code (an export of commit 1a51dc6) instead of the editable
install; the other section in the file is left as it is. Every
measurement goes through functions that exist in both versions, so the
two sections count the same things the same way:

1. in_sample_signal_days: signal days before the model's first
   out-of-sample date that carry a model prediction, i.e. days predicted
   by a model fitted on their own labels.
2. evaluate_out_of_sample_overlap_days: days evaluate_out_of_sample()
   calls out-of-sample that the model trained on.
3. rolling_refit_overlap_days: over every rolling_walk_forward
   (refit_per_fold=True) fold, test-window days the fold's model trained
   on.
4. regime_fit_overlap_days: days on or after the ML split that the
   regime model behind ml_regime_conditional was fitted on; and whether
   that regime result's `causal` flag is true.
5. last_label: the final row's label (no next close exists) and whether
   that row reaches the training data.
6. model_report_overlap_rows: rows scored in both model_report()'s
   train and test accuracy.
7. dummy_columns_from_test_only_regimes: regime dummy columns created
   for a regime that never occurs in the training rows (probe: regime 0
   before the split, regime 1 after).

In the "after" section every count must be 0, the last label NaN and
the causal flag truthful; the script asserts this. In the "before"
section it asserts the opposite, so the audit is shown to detect each
problem.

Run with: python scripts/audit_daily_ml.py --section after
"""

import argparse
import functools
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import qbt.adaptive as adaptive
import qbt.ml as ml
from qbt.data import fetch_ohlcv
from qbt.features import build_labels
from qbt.regime import UNKNOWN
from qbt.walk_forward import evaluate_out_of_sample, rolling_walk_forward

TICKER, START, END = "SPY", "2008-01-01", "2025-01-01"
TRAIN_FRAC = 0.7
ROLLING_TRAIN_DAYS, ROLLING_TEST_DAYS = 756, 126
OUT = Path(__file__).resolve().parents[1] / "results" / "ml" / "daily_ml_audit.json"


def _predictions(df: pd.DataFrame) -> pd.Series:
    """Model predictions behind ml_direction_signal; NaN where it makes none."""
    if hasattr(ml, "ml_direction_predictions"):
        return ml.ml_direction_predictions(df, train_frac=TRAIN_FRAC)
    # Pre-fix code: ml_direction_signal predicted every valid row
    # (qbt/ml/__init__.py:179 at 1a51dc6), so this is what it used.
    model, cols, _split, data = ml.train_ml_model(df, train_frac=TRAIN_FRAC)
    return pd.Series(model.predict(data[cols]), index=data.index, dtype=float)


def _in_sample(df: pd.DataFrame, split: pd.Timestamp) -> dict:
    predictions = _predictions(df).dropna()
    signal = ml.ml_direction_signal(df, train_frac=TRAIN_FRAC)
    before_split = signal.index < split
    return {
        "first_out_of_sample_date": split.date().isoformat(),
        "in_sample_signal_days": int((predictions.index < split).sum()),
        "in_sample_share_of_all_days": float((predictions.index < split).sum() / len(df)),
        "days_long_before_split": int((signal[before_split] != 0).sum()),
    }


def _evaluate_overlap(df: pd.DataFrame, split: pd.Timestamp) -> dict:
    eval_split = pd.Timestamp(evaluate_out_of_sample(df, ml.ml_direction_signal)["split_date"])
    overlap = int(((df.index >= eval_split) & (df.index < split)).sum())
    return {"evaluate_out_of_sample_split": eval_split.date().isoformat(),
            "model_split": split.date().isoformat(),
            "evaluate_out_of_sample_overlap_days": overlap}


def _rolling_overlap(df: pd.DataFrame) -> dict:
    per_fold = []

    @functools.wraps(ml.ml_direction_signal)
    def recording(fold_df: pd.DataFrame, **params):
        # rolling_walk_forward also runs the strategy once on the whole
        # series (its fixed-model comparison); only fold windows count here.
        if len(fold_df) != ROLLING_TRAIN_DAYS + ROLLING_TEST_DAYS:
            return ml.ml_direction_signal(fold_df, **params)
        _model, _cols, fold_split, data = ml.train_ml_model(
            fold_df, train_frac=params.get("train_frac", TRAIN_FRAC))
        test_start = fold_df.index[ROLLING_TRAIN_DAYS]
        trained = data.index[data.index < fold_split]
        per_fold.append(int((trained >= test_start).sum()))
        return ml.ml_direction_signal(fold_df, **params)

    rolling_walk_forward(df, recording, train_days=ROLLING_TRAIN_DAYS,
                         test_days=ROLLING_TEST_DAYS, refit_per_fold=True)
    return {"n_folds": len(per_fold), "per_fold": per_fold,
            "rolling_refit_overlap_days": int(sum(per_fold))}


def _regime_overlap(df: pd.DataFrame, split: pd.Timestamp) -> dict:
    captured = []
    originals = (adaptive.detect_regimes, adaptive.detect_regimes_walk_forward)

    def capture(fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            captured.append(result)
            return result
        return wrapper

    adaptive.detect_regimes = capture(originals[0])
    adaptive.detect_regimes_walk_forward = capture(originals[1])
    try:
        adaptive.ml_regime_conditional(df)
    finally:
        adaptive.detect_regimes, adaptive.detect_regimes_walk_forward = originals
    (result,) = captured
    fit_end = pd.Timestamp(result.fit_end)
    labelled = result.labels != UNKNOWN
    # Truth: a fitted regime result is causal only if no labelled day is
    # on or before the last day its model was fitted on.
    in_fit = bool((labelled & (result.labels.index <= fit_end)).any())
    return {
        "ml_split": split.date().isoformat(),
        "regime_fit_end": fit_end.date().isoformat(),
        "regime_fit_overlap_days": int(((df.index >= split) & (df.index <= fit_end)).sum()),
        "causal_flag": bool(result.causal),
        "causal_truth": not in_fit,
        "causal_flag_truthful": bool(result.causal) == (not in_fit),
        "causal_from": (None if getattr(result, "causal_from", None) is None
                        else pd.Timestamp(result.causal_from).date().isoformat()),
    }


def _last_label(df: pd.DataFrame) -> dict:
    labels = build_labels(df)
    data, _cols = ml._prepare_data(df)
    last = labels.iloc[-1]
    return {"last_label": None if pd.isna(last) else float(last),
            "last_row_in_training_data": bool(df.index[-1] in data.index)}


def _model_report_overlap(df: pd.DataFrame) -> dict:
    scored = []
    original = ml.accuracy_score

    def recording(y_true, y_pred, *args, **kwargs):
        scored.append(pd.Index(y_true.index))
        return original(y_true, y_pred, *args, **kwargs)

    ml.accuracy_score = recording
    try:
        ml.model_report(df, train_frac=TRAIN_FRAC)
    finally:
        ml.accuracy_score = original
    train_rows, test_rows = scored[0], scored[1]
    return {"train_rows_scored": len(train_rows), "test_rows_scored": len(test_rows),
            "model_report_overlap_rows": len(train_rows.intersection(test_rows))}


def _dummy_columns(df: pd.DataFrame, split: pd.Timestamp) -> dict:
    probe = pd.Series(np.where(df.index < split, 0, 1), index=df.index)
    _model, cols, _split, _data = ml.train_ml_model(df, train_frac=TRAIN_FRAC, regimes=probe)
    regime_cols = sorted(c for c in cols if c.startswith("regime_"))
    return {"probe": "regime 0 on every row before the ML split, regime 1 on every row after",
            "regime_columns": regime_cols,
            "dummy_columns_from_test_only_regimes": int("regime_1" in regime_cols)}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=OUT.parents[2]).stdout.strip()


def measure(source: str) -> dict:
    df = fetch_ohlcv(TICKER, START, END)
    split = pd.Timestamp(ml.train_ml_model(df, train_frac=TRAIN_FRAC)[2])
    return {
        "source": source,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {"ticker": TICKER, "first_date": df.index[0].date().isoformat(),
                 "last_date": df.index[-1].date().isoformat(), "n_trading_days": len(df)},
        "parameters": {"train_frac": TRAIN_FRAC, "rolling_train_days": ROLLING_TRAIN_DAYS,
                       "rolling_test_days": ROLLING_TEST_DAYS},
        "1_in_sample_signal": _in_sample(df, split),
        "2_evaluate_out_of_sample": _evaluate_overlap(df, split),
        "3_rolling_refit_per_fold": _rolling_overlap(df),
        "4_regime_fit_window": _regime_overlap(df, split),
        "5_last_label": _last_label(df),
        "6_model_report": _model_report_overlap(df),
        "7_regime_dummy_columns": _dummy_columns(df, split),
    }


def check(section: str, m: dict) -> None:
    counts = {
        "in_sample_signal_days": m["1_in_sample_signal"]["in_sample_signal_days"],
        "days_long_before_split": m["1_in_sample_signal"]["days_long_before_split"],
        "evaluate_out_of_sample_overlap_days":
            m["2_evaluate_out_of_sample"]["evaluate_out_of_sample_overlap_days"],
        "rolling_refit_overlap_days": m["3_rolling_refit_per_fold"]["rolling_refit_overlap_days"],
        "regime_fit_overlap_days": m["4_regime_fit_window"]["regime_fit_overlap_days"],
        "model_report_overlap_rows": m["6_model_report"]["model_report_overlap_rows"],
        "dummy_columns_from_test_only_regimes":
            m["7_regime_dummy_columns"]["dummy_columns_from_test_only_regimes"],
    }
    truthful = m["4_regime_fit_window"]["causal_flag_truthful"]
    last = m["5_last_label"]
    if section == "after":
        for name, value in counts.items():
            assert value == 0, f"{name} = {value}, expected 0 after the fix"
        assert truthful, "regime result's causal flag disagrees with its fit window"
        assert last["last_label"] is None and not last["last_row_in_training_data"]
    else:
        for name, value in counts.items():
            assert value > 0, f"{name} = 0: the audit failed to detect it on the pre-fix code"
        assert not truthful
        assert last["last_label"] == 0.0 and last["last_row_in_training_data"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("before", "after"), required=True)
    parser.add_argument("--source", help="what code is being measured (default: git HEAD)")
    args = parser.parse_args()

    source = args.source or (f"git {_git('rev-parse', '--short', 'HEAD')}"
                             + (" + uncommitted changes" if _git("status", "--porcelain")
                                else ""))
    measured = measure(source)
    check(args.section, measured)

    report = json.loads(OUT.read_text()) if OUT.exists() else {}
    report["audit"] = "scripts/audit_daily_ml.py"
    report[args.section] = measured
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({k: report[k] for k in ("audit", "before", "after")
                               if k in report}, indent=2) + "\n")
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()
