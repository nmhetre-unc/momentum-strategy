"""
C2 report for monthly SPY prediction: logistic regression and LightGBM
(qbt.ml.models) trained inside the C1 purged walk-forward folds, tuned
on inner purged folds only, for both targets (next-month direction and
regime), with and without the HMM features. Same data, folds and
baselines as scripts/ml_c1_report.py, plus, for regime, the rules
regime on the decision date.

Writes:
- results/ml/c2_trials.jsonl: one line per (target, feature set, model,
  outer fold, hyperparameter configuration) inner-tuning evaluation,
  then one line per (target, feature set, model) out-of-sample result;
- results/ml/c2_results.json: per fold and pooled AUC, log loss, Brier
  score and accuracy vs every baseline (stationary-bootstrap intervals,
  exact McNemar), calibration bins, the effect of the HMM features, and
  the economic test of the direction models (5 bps, Sharpe vs
  buy-and-hold over identical months, block-bootstrap intervals,
  deflated Sharpe over every configuration tried), and a verdict
  computed from those numbers;
- results/ml/c2_calibration.png and results/ml/c2_equity.png.

Every number is deterministic for a given cache file (fixed seeds,
deterministic LightGBM, seeded bootstrap); the environment block and
timings change between runs.

Trial count. The deflated Sharpe counts every configuration tried,
including a first pass run before the grids were widened (FIRST_PASS:
both targets, base features, the narrower grids). That pass was a
separate nested selection whose out-of-sample accuracy was seen, so it
counts as separate trials even though each of its configurations is
also in the final grids. It is re-run here from the same code and
logged to c2_trials.jsonl as "first_pass_*" records.

The bar. For regime, the bar a model has to clear is
rules_regime_lookup: for each test month, the most frequent next-month
regime among training months that had the same rules regime on their
decision date (ties to the lower class). It fits a 4x4 table and beats
the raw rules regime. For direction, the bar is majority_class (always
up, i.e. buy-and-hold).

Run with: python scripts/ml_c2_report.py   (needs the `ml` extra)
"""

import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kurtosis, skew  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_c1_report import (  # noqa: E402
    COST_BPS,
    EMBARGO,
    END,
    MIN_TRAIN,
    START,
    TEST_SIZE,
    TICKER,
    _data_block,
    _environment,
    _jsonable,
)

from qbt.data import fetch_ohlcv  # noqa: E402
from qbt.ml.baselines import (  # noqa: E402
    conditional_prior_proba,
    prior_proba,
)
from qbt.ml.evaluate import (  # noqa: E402
    BAR,
    MonthlyDataset,
    balanced_accuracy,
    baseline_predictions,
    build_dataset,
    mcnemar_exact,
    strategy_returns,
    trading_summary,
)
from qbt.ml.features import clear_hmm_cache, hmm_cache_info  # noqa: E402
from qbt.ml.metrics import (  # noqa: E402
    CI_LEVEL,
    MEAN_BLOCK,
    N_BOOT,
    auc,
    brier,
    brier_binary,
    calibration_bins,
    log_loss,
    log_loss_per_month,
    paired_mean_bootstrap,
    paired_sharpe_bootstrap,
    predicted_class,
)
from qbt.ml.models import (  # noqa: E402
    INNER_SPLITS,
    LIGHTGBM_FIXED,
    MODELS,
    PROB_FLOOR,
    SEED,
    WalkForwardPredictions,
    classes_for,
    grid_for,
    walk_forward_predict,
)
from qbt.ml.splits import Fold, walk_forward_splits  # noqa: E402
from qbt.stats import deflated_sharpe_ratio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "ml"
C1_FILE = OUT_DIR / "c1_baselines.json"
TARGETS = ("direction", "regime")
FEATURE_SETS = {"base": False, "base+hmm": True}
ALPHA = 0.05
# The grids before they were widened (see qbt.ml.models): run on the base
# features for both targets, then discarded. Counted as trials.
FIRST_PASS: dict[str, Any] = {
    "targets": ("direction", "regime"),
    "feature_sets": ("base",),
    "grids": {
        "logistic": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
        "lightgbm": [{"num_leaves": leaves, "min_child_samples": child, "n_estimators": trees}
                     for leaves, child, trees in product((3, 7), (5, 15), (50, 200))],
    },
}
SERIES_COLORS = {"logistic": "#2a78d6", "lightgbm": "#eb6834"}   # categorical slots 1, 2
LINESTYLES = {"base": "-", "base+hmm": "--"}


def _references(ds: MonthlyDataset, folds: list[Fold]) -> dict[str, np.ndarray]:
    """Probabilistic baselines, fitted on each outer training set like the models."""
    classes = classes_for(ds.target)
    refs = {"prior": np.vstack([
        prior_proba(ds.y.iloc[f.train], len(f.test), classes, PROB_FLOOR) for f in folds])}
    if ds.target == "regime":
        rules = ds.X["regime_rules"]
        refs["prior_given_rules_regime"] = np.vstack([
            conditional_prior_proba(ds.y.iloc[f.train], rules.iloc[f.train],
                                    rules.iloc[f.test], classes, PROB_FLOOR)
            for f in folds])
    return refs


def _check_against_c1(c1: dict, target: str, include_hmm: bool,
                      base: pd.DataFrame) -> None:
    """The baselines here must be exactly C1's (same data, same folds)."""
    run = next(r for r in c1["runs"]
               if r["target"] == target and r["include_hmm"] == include_hmm)
    assert run["n_oos_months"] == len(base), (target, include_hmm)
    for row in run["summary"]:
        mine = float((base[row["predictor"]] == base["y"]).mean())
        assert np.isclose(mine, row["accuracy"], rtol=0, atol=1e-12), row["predictor"]
    if target == "regime":
        reported = run["regime_label_determinacy"][
            "label_equals_rules_regime_on_decision_date"]["oos_test_months"]
        mine = float((base["rules_regime_at_decision"] == base["y"]).mean())
        assert np.isclose(mine, reported, rtol=0, atol=1e-12)


def _scores(y: np.ndarray, proba: np.ndarray, classes: tuple[float, ...]) -> dict:
    value, n_classes = auc(y, proba, classes)
    pred = predicted_class(proba, classes)
    out = {"n": len(y), "auc": value, "auc_classes_averaged": n_classes,
           "log_loss": log_loss(y, proba, classes), "brier": brier(y, proba, classes),
           "accuracy": float(np.mean(pred == y)),
           "balanced_accuracy": balanced_accuracy(y, pred)}
    if len(classes) == 2:
        out["brier_binary"] = brier_binary(y, proba[:, 1])
    return out


def _evaluate_model(ds: MonthlyDataset, wf: WalkForwardPredictions, base: pd.DataFrame,
                    refs: dict[str, np.ndarray], baselines: list[str]) -> dict:
    classes = classes_for(ds.target)
    proba = wf.proba.loc[base.index].to_numpy()
    y = base["y"].to_numpy()
    pred = predicted_class(proba, classes)
    correct = pred == y

    per_fold = []
    for number, rows in base.groupby("fold").indices.items():
        row = {"fold": int(str(number)), **_scores(y[rows], proba[rows], classes)}
        for name, ref in refs.items():
            row[f"{name}_log_loss"] = log_loss(y[rows], ref[rows], classes)
            row[f"{name}_brier"] = brier(y[rows], ref[rows], classes)
        for b in baselines:
            row[f"acc_{b}"] = float(np.mean(base[b].to_numpy()[rows] == y[rows]))
        per_fold.append(row)

    accuracy = paired_mean_bootstrap(
        {"model": correct, **{b: base[b].to_numpy() == y for b in baselines}}, "model")
    accuracy_block12 = paired_mean_bootstrap(
        {"model": correct, **{b: base[b].to_numpy() == y for b in baselines}}, "model",
        mean_block=12)
    for b in baselines:
        accuracy["vs"][b]["mcnemar_p"] = mcnemar_exact(correct, base[b].to_numpy() == y)
        accuracy["vs"][b]["difference_ci_mean_block_12"] = \
            accuracy_block12["vs"][b]["difference_ci"]
    loss = paired_mean_bootstrap(
        {"model": log_loss_per_month(y, proba, classes),
         **{name: log_loss_per_month(y, ref, classes) for name, ref in refs.items()}},
        "model", higher_is_better=False)

    if len(classes) == 2:
        calibration = {"forecast": "P(up)",
                       "bins": calibration_bins(proba[:, 1], (y == classes[1]).astype(float))}
    else:
        calibration = {"forecast": "top-label confidence (max class probability)",
                       "bins": calibration_bins(proba.max(axis=1), correct.astype(float))}
    return {
        "selected_per_fold": wf.selected,
        "per_fold": per_fold,
        "pooled": _scores(y, proba, classes),
        "references_pooled": {name: _scores(y, ref, classes) for name, ref in refs.items()},
        "accuracy_vs_baselines": accuracy,
        "log_loss_vs_references": loss,
        "calibration": calibration,
        "predicted_class_share": {str(c): float(np.mean(pred == c)) for c in classes},
    }


def _economic(ds: MonthlyDataset, wf: WalkForwardPredictions, index: pd.Index,
              n_trials: int) -> tuple[dict, pd.Series, pd.Series]:
    fwd = ds.forward_return.loc[index]
    positions = pd.Series((wf.proba.loc[index, 1.0] > 0.5).astype(float), index=index)
    held = pd.Series(1.0, index=index)
    strat = strategy_returns(positions, fwd, COST_BPS)
    bench = strategy_returns(held, fwd, COST_BPS)
    s, b = strat.to_numpy(), bench.to_numpy()
    boot = paired_sharpe_bootstrap(s, b)
    per_period = s.mean() / s.std(ddof=1)
    bench_per_period = b.mean() / b.std(ddof=1)
    sk, ku = float(skew(s)), float(kurtosis(s, fisher=False))
    identical = bool(np.array_equal(s, b))
    vs_bh = None if identical else deflated_sharpe_ratio(
        per_period, n_trials, sk, ku, len(s), sharpe_benchmark=bench_per_period)
    vs_bh_single = None if identical else deflated_sharpe_ratio(
        per_period, 1, sk, ku, len(s), sharpe_benchmark=bench_per_period)
    return ({
        "window": {"first_decision": index[0], "last_label_end": ds.t1.loc[index[-1]],
                   "n_months": len(index)},
        "rule": "long when P(up) > 0.5, else flat; held to the next decision date",
        "strategy": trading_summary(strat, positions),
        "buy_and_hold": trading_summary(bench, held),
        "n_position_changes": int(positions.diff().abs().fillna(positions.abs()).sum()),
        "identical_to_buy_and_hold": identical,
        "sharpe_bootstrap": boot,
        "deflated_sharpe": {
            "n_trials": n_trials,
            "per_period_sharpe": float(per_period),
            "skew": sk, "kurtosis_pearson": ku,
            "vs_zero": deflated_sharpe_ratio(per_period, n_trials, sk, ku, len(s)),
            "vs_buy_and_hold": vs_bh,
            "single_trial_vs_buy_and_hold": vs_bh_single,
            "note": ("unset vs buy-and-hold: the strategy's returns are buy-and-hold's "
                     "(always long), so there is no hypothesis to deflate"
                     if identical else
                     "P(true Sharpe > threshold) after deflating for n_trials; per-period "
                     "(monthly) Sharpe, Pearson kurtosis"),
        },
    }, (1 + strat).cumprod(), (1 + bench).cumprod())


def _hmm_effect(preds: dict, datasets: dict) -> list[dict]:
    """With vs without HMM features, same target and model, on the months both test."""
    rows = []
    for target in TARGETS:
        classes = classes_for(target)
        for model in MODELS:
            a, b = preds[(target, "base", model)], preds[(target, "base+hmm", model)]
            common = a.proba.index.intersection(b.proba.index)
            y = datasets[(target, "base")].y.loc[common].to_numpy()
            pa, pb = a.proba.loc[common].to_numpy(), b.proba.loc[common].to_numpy()
            ca, cb = predicted_class(pa, classes) == y, predicted_class(pb, classes) == y
            rows.append({
                "target": target, "model": model, "n_common_months": len(common),
                "first_common": common[0], "last_common": common[-1],
                "accuracy_base": float(ca.mean()), "accuracy_base_hmm": float(cb.mean()),
                "log_loss_base": log_loss(y, pa, classes),
                "log_loss_base_hmm": log_loss(y, pb, classes),
                "mcnemar_p": mcnemar_exact(cb, ca),
                "log_loss_difference_hmm_minus_base": paired_mean_bootstrap(
                    {"hmm": log_loss_per_month(y, pb, classes),
                     "base": log_loss_per_month(y, pa, classes)},
                    "hmm", higher_is_better=False)["vs"]["base"],
            })
    return rows


def _verdict(runs: list[dict], economic: list[dict]) -> dict:
    """Mechanical: a model beats a baseline only if the paired interval excludes 0 and p < ALPHA."""
    classification = []
    for run in runs:
        for model, rep in run["models"].items():
            vs = rep["accuracy_vs_baselines"]["vs"]
            beaten = [b for b, v in vs.items()
                      if v["difference_ci"][0] > 0 and v["mcnemar_p"] < ALPHA]
            loss = rep["log_loss_vs_references"]["vs"]
            bar = BAR[run["target"]]
            classification.append({
                "target": run["target"], "feature_set": run["feature_set"], "model": model,
                "accuracy": rep["pooled"]["accuracy"],
                "bar": bar, "bar_accuracy": vs[bar]["other_mean"],
                "difference_vs_bar": vs[bar]["difference"],
                "difference_vs_bar_ci": vs[bar]["difference_ci"],
                "mcnemar_p_vs_bar": vs[bar]["mcnemar_p"],
                "beats_bar": bar in beaten,
                "best_baseline": max(vs, key=lambda b: vs[b]["other_mean"]),
                "best_baseline_accuracy": max(v["other_mean"] for v in vs.values()),
                "baselines_beaten": beaten,
                "beats_every_baseline": len(beaten) == len(vs),
                "log_loss_better_than": [r for r, v in loss.items()
                                         if v["difference_ci"][1] < 0],
                "log_loss_references": list(loss),
            })
    trading = [{
        "feature_set": e["feature_set"], "model": e["model"],
        "sharpe": e["sharpe_bootstrap"]["sharpe"],
        "buy_and_hold_sharpe": e["sharpe_bootstrap"]["benchmark_sharpe"],
        "beats_buy_and_hold": e["sharpe_bootstrap"]["difference_ci"][0] > 0,
        "deflated_sharpe_vs_buy_and_hold": e["deflated_sharpe"]["vs_buy_and_hold"],
    } for e in economic]
    any_cls = [c for c in classification if c["beats_bar"]]
    any_trd = [t for t in trading if t["beats_buy_and_hold"]]
    statement = []
    statement.append(
        "No model beats its bar on accuracy (direction: majority_class; regime: "
        "rules_regime_lookup)." if not any_cls else
        "Beats its bar on accuracy: " + ", ".join(
            f"{c['target']}/{c['feature_set']}/{c['model']}" for c in any_cls) + ".")
    statement.append(
        "No direction strategy beats buy-and-hold on Sharpe (paired interval includes or is "
        "below 0)." if not any_trd else
        "Beats buy-and-hold on Sharpe: " + ", ".join(
            f"{t['feature_set']}/{t['model']}" for t in any_trd) + ".")
    return {"bar": BAR,
            "rule": f"beats = paired {CI_LEVEL:.0%} stationary-bootstrap interval of the "
                    f"difference above 0 and exact McNemar p < {ALPHA} (classification); "
                    "paired Sharpe-difference interval above 0 (trading)",
            "statement": " ".join(statement),
            "classification": classification, "trading": trading}


def _plot_calibration(runs: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    titles = {"direction": "Direction: forecast P(up) vs share of up months",
              "regime": "Regime: top-label confidence vs share correct"}
    for ax, target in zip(axes, TARGETS, strict=True):
        ax.plot([0, 1], [0, 1], color="#8a8a86", linewidth=1, label="perfectly calibrated")
        for run in (r for r in runs if r["target"] == target):
            for model, rep in run["models"].items():
                bins = rep["calibration"]["bins"]
                ax.plot([b["mean_forecast"] for b in bins],
                        [b["observed_frequency"] for b in bins],
                        color=SERIES_COLORS[model], linestyle=LINESTYLES[run["feature_set"]],
                        linewidth=2, marker="o", markersize=5,
                        label=f"{model}, {run['feature_set']}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(titles[target], fontsize=10)
        ax.set_xlabel("mean forecast (5 equal-count bins)")
        ax.set_ylabel("observed frequency")
        ax.grid(color="#e4e4e0", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("C2 out-of-sample calibration, SPY monthly (results/ml/c2_results.json)",
                 fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_equity(curves: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    for ax, feature_set in zip(axes, FEATURE_SETS, strict=True):
        bench = None
        for model in MODELS:
            strat, bench = curves[(feature_set, model)]
            ax.plot(strat.index, strat.to_numpy(), color=SERIES_COLORS[model], linewidth=2,
                    label=f"{model} (long if P(up) > 0.5)")
        assert bench is not None
        if all(np.allclose(curves[(feature_set, m)][0].to_numpy(), bench.to_numpy())
               for m in MODELS):
            ax.text(0.98, 0.04, "both models were long every month:\nall three lines coincide",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    color="#5c5c58")
        ax.plot(bench.index, bench.to_numpy(), color="#5c5c58", linewidth=2,
                linestyle=":", label="buy-and-hold, same months")
        ax.set_title(f"Features: {feature_set}, {COST_BPS:g} bps per change", fontsize=10)
        ax.set_ylabel("growth of 1 (monthly, net)")
        ax.grid(color="#e4e4e0", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("C2 economic test, out-of-sample months only (results/ml/c2_results.json)",
                 fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    environment = _environment()
    df = fetch_ohlcv(TICKER, START, END)
    c1 = json.loads(C1_FILE.read_text())
    n_configs = {m: len(grid_for(m)) for m in MODELS}
    n_final = len(TARGETS) * len(FEATURE_SETS) * sum(n_configs.values())
    n_first_pass = (len(FIRST_PASS["targets"]) * len(FIRST_PASS["feature_sets"])
                    * sum(len(g) for g in FIRST_PASS["grids"].values()))
    n_trials = n_final + n_first_pass

    clear_hmm_cache()
    datasets: dict[tuple[str, str], MonthlyDataset] = {}
    build_seconds = {}
    for feature_set, include_hmm in FEATURE_SETS.items():
        for target in TARGETS:
            started = time.perf_counter()
            datasets[(target, feature_set)] = build_dataset(df, target, include_hmm)
            build_seconds[f"{target}/{feature_set}"] = time.perf_counter() - started
    cache = hmm_cache_info()
    assert cache["computed"] == 1, cache

    trial_lines: list[dict] = []
    first_pass_accuracy = []
    for target in FIRST_PASS["targets"]:
        for feature_set in FIRST_PASS["feature_sets"]:
            ds = datasets[(target, feature_set)]
            folds = walk_forward_splits(ds.t0, ds.t1, min_train=MIN_TRAIN,
                                        test_size=TEST_SIZE, embargo=EMBARGO)
            for model, grid in FIRST_PASS["grids"].items():
                wf = walk_forward_predict(model, ds.X, ds.y, ds.t1, folds, target, grid=grid)
                for trial in wf.trials:
                    trial_lines.append({"record": "first_pass_inner_tuning", "target": target,
                                        "feature_set": feature_set, "model": model, **trial})
                y = ds.y.loc[wf.proba.index].to_numpy()
                accuracy = float(np.mean(
                    predicted_class(wf.proba.to_numpy(), classes_for(target)) == y))
                first_pass_accuracy.append({"target": target, "feature_set": feature_set,
                                            "model": model, "accuracy": accuracy})
                trial_lines.append({"record": "first_pass_outer_test", "target": target,
                                    "feature_set": feature_set, "model": model,
                                    "selected_per_fold": [x["config"] for x in wf.selected],
                                    "accuracy": accuracy, "n": len(y)})
    preds: dict[tuple[str, str, str], WalkForwardPredictions] = {}
    runs, economic, curves = [], [], {}
    fit_seconds = {}
    for (target, feature_set), ds in datasets.items():
        folds = walk_forward_splits(ds.t0, ds.t1, min_train=MIN_TRAIN,
                                    test_size=TEST_SIZE, embargo=EMBARGO)
        base = baseline_predictions(ds, folds, PROB_FLOOR)
        _check_against_c1(c1, target, FEATURE_SETS[feature_set], base)
        baselines = [c for c in base.columns if c not in ("fold", "y")]
        refs = _references(ds, folds)
        run: dict = {
            "target": target, "feature_set": feature_set,
            "features": list(ds.X.columns), "n_samples": len(ds.y),
            "n_folds": len(folds), "n_oos_months": len(base),
            "first_test_decision": base.index[0], "last_test_decision": base.index[-1],
            "baseline_accuracy": {b: float((base[b] == base["y"]).mean()) for b in baselines},
            "models": {},
        }
        for model in MODELS:
            started = time.perf_counter()
            wf = walk_forward_predict(model, ds.X, ds.y, ds.t1, folds, target)
            fit_seconds[f"{target}/{feature_set}/{model}"] = time.perf_counter() - started
            preds[(target, feature_set, model)] = wf
            for trial in wf.trials:
                trial_lines.append({"record": "inner_tuning", "target": target,
                                    "feature_set": feature_set, "model": model, **trial})
            rep = _evaluate_model(ds, wf, base, refs, baselines)
            run["models"][model] = rep
            trial_lines.append({"record": "outer_test", "target": target,
                                "feature_set": feature_set, "model": model,
                                "selected_per_fold": [s["config"] for s in wf.selected],
                                **rep["pooled"]})
            if target == "direction":
                block, strat_curve, bench_curve = _economic(ds, wf, base.index, n_trials)
                economic.append({"feature_set": feature_set, "model": model, **block})
                curves[(feature_set, model)] = (strat_curve, bench_curve)
            print(f"{target:9s} {feature_set:8s} {model:8s} "
                  f"acc={rep['pooled']['accuracy']:.3f} auc={rep['pooled']['auc']:.3f} "
                  f"logloss={rep['pooled']['log_loss']:.3f}")
        runs.append(run)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_calibration(runs, OUT_DIR / "c2_calibration.png")
    _plot_equity(curves, OUT_DIR / "c2_equity.png")

    verdict = _verdict(runs, economic)
    report = {
        "report": "scripts/ml_c2_report.py",
        "data": _data_block(df),
        "parameters": {
            "outer_splits": {"kind": "walk_forward_splits (C1)", "min_train": MIN_TRAIN,
                             "test_size": TEST_SIZE, "embargo": EMBARGO},
            "inner_splits": {"kind": "purged_kfold_splits inside each outer training set",
                             "n_splits": INNER_SPLITS, "embargo": EMBARGO,
                             "selection": "lowest pooled inner log loss"},
            "grids": {m: grid_for(m) for m in MODELS},
            "lightgbm_fixed": LIGHTGBM_FIXED,
            "logistic": "StandardScaler on numeric features, one-hot on regime codes, "
                        "lbfgs, max_iter=10000",
            "seed": SEED, "prob_floor": PROB_FLOOR,
            "bootstrap": {"kind": "stationary (qbt.bootstrap), paired", "n_boot": N_BOOT,
                          "mean_block_months": MEAN_BLOCK, "ci_level": CI_LEVEL, "seed": 0,
                          "sensitivity": "accuracy differences also at mean_block 12"},
            "mcnemar": "exact, two-sided, model vs each baseline on the same months",
            "brier": "multiclass form: sum over classes of (p_k - 1[y=k])^2, averaged over "
                     "months; brier_binary is the usual binary score for direction",
            "auc": "binary for direction; macro one-vs-rest over classes defined in the "
                   "sample for regime",
            "cost_bps": COST_BPS,
        },
        "trials": {
            "configurations_per_model": n_configs,
            "targets": list(TARGETS), "feature_sets": list(FEATURE_SETS),
            "final_grid_configurations": n_final,
            "first_pass": {
                "targets": list(FIRST_PASS["targets"]),
                "feature_sets": list(FIRST_PASS["feature_sets"]),
                "configurations_per_model": {m: len(g) for m, g in
                                             FIRST_PASS["grids"].items()},
                "grids": FIRST_PASS["grids"],
                "configurations": n_first_pass,
                "out_of_sample_accuracy": first_pass_accuracy,
                "note": "run before the grids were widened, then discarded; each of these "
                        "configurations is also in the final grids, but the pass was a "
                        "separate selection whose out-of-sample results were seen, so it "
                        "counts as separate trials",
            },
            "total_configurations": n_trials,
            "previous_total_without_first_pass": n_final,
            "note": "the deflated Sharpe counts every (target, feature set, model, "
                    "hyperparameter configuration) combination tried, including the "
                    "first pass and the regime target's, which trades nothing",
        },
        "hmm_cache": {**cache, "build_dataset_wall_seconds": build_seconds},
        "runs": runs,
        "hmm_effect": _hmm_effect(preds, datasets),
        "economic": economic,
        "verdict": verdict,
        "plots": ["results/ml/c2_calibration.png", "results/ml/c2_equity.png"],
        "timings": {"walk_forward_wall_seconds": fit_seconds},
        "environment": environment,
    }
    (OUT_DIR / "c2_results.json").write_text(json.dumps(_jsonable(report), indent=2) + "\n")
    with (OUT_DIR / "c2_trials.jsonl").open("w") as fh:
        for line in trial_lines:
            fh.write(json.dumps(_jsonable(line)) + "\n")
    print(verdict["statement"])
    print(f"wrote results/ml/c2_results.json, c2_trials.jsonl ({len(trial_lines)} lines), "
          "c2_calibration.png, c2_equity.png")


if __name__ == "__main__":
    main()
