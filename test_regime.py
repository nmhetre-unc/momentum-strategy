"""
Sanity-checks regime.py, regime_features.py, adaptive.py and the new
regime-aware paths through backtest.py / ml_strategy.py / walk_forward.py,
against synthetic data with KNOWN regime structure.

Same approach as test_logic.py -- assertions plus a readable printout, no
network required. The synthetic series here cycles through three
deliberately different states (calm drift up, flat and choppy, violent
selloff), so a working detector should recover something close to them.
That's the point: on real data you can never check a regime label against
ground truth, so the only place you can verify the machinery is where you
built the truth yourself.

Run with: python test_regime.py
"""

import numpy as np
import pandas as pd

from adaptive import ADAPTIVE_STRATEGIES, ALL_STRATEGIES, describe_choices, describe_filter
from analytics import full_report, performance_by_regime
from backtest import run_backtest
from ml_strategy import model_report
from regime import (
    REGIME_METHODS, UNKNOWN, detect_regimes, detect_regimes_walk_forward,
    regime_episodes, regime_stability, regime_summary, smooth_labels,
)
from regime_features import build_regime_features, standardize_features
from strategies import STRATEGIES
from walk_forward import (
    compare_strategies, evaluate_out_of_sample, evaluate_with_regimes, rolling_walk_forward,
)

# --------------------------------------------------------------------------
# Synthetic data with known regimes
# --------------------------------------------------------------------------
RNG = np.random.default_rng(20240826)
N = 2600
SPECS = [(0.0009, 0.0065), (0.0000, 0.0110), (-0.0016, 0.0290)]


def make_series(with_ohlcv: bool = True) -> pd.DataFrame:
    segments, total, i = [], 0, 0
    while total < N:
        drift, vol = SPECS[i % 3]
        length = int(RNG.integers(150, 320))
        segments.append(RNG.normal(drift, vol, length))
        total += length
        i += 1

    returns = np.concatenate(segments)[:N]
    close = 100 * np.cumprod(1 + returns)
    index = pd.date_range("2008-01-01", periods=N, freq="B")

    if not with_ohlcv:
        return pd.DataFrame({"Close": close}, index=index)
    return pd.DataFrame({
        "Close": close,
        "High": close * (1 + np.abs(RNG.normal(0, 0.004, N))),
        "Low": close * (1 - np.abs(RNG.normal(0, 0.004, N))),
        "Volume": RNG.lognormal(15, 0.4, N),
    }, index=index)


df = make_series(with_ohlcv=True)
df_close_only = make_series(with_ohlcv=False)

# --------------------------------------------------------------------------
# 1. Features must be point-in-time and survive a Close-only frame
# --------------------------------------------------------------------------
print("=== Regime features ===")
features = build_regime_features(df)
features_close_only = build_regime_features(df_close_only)

for name, table in (("OHLCV", features), ("Close-only", features_close_only)):
    assert not table.empty, f"{name}: no features produced"
    all_nan = [c for c in table.columns if table[c].notna().sum() == 0]
    assert not all_nan, f"{name}: columns are entirely NaN: {all_nan}"
    clean = standardize_features(table).dropna()
    assert len(clean) > 500, f"{name}: only {len(clean)} usable rows after warm-up"
    print(f"{name:12s} {len(table.columns):2d} features, {len(clean):4d} usable rows")

# The expanding percentile must never see the future: recomputing it on a
# truncated series has to give the same answer for the rows that survive.
truncated = build_regime_features(df.iloc[:1500])
overlap = truncated.index.intersection(features.index)
for column in ("vol_percentile", "vol_20d", "trend_60d", "efficiency_ratio"):
    left = features.loc[overlap, column].dropna()
    right = truncated.loc[overlap, column].dropna()
    shared = left.index.intersection(right.index)
    assert np.allclose(left[shared], right[shared], equal_nan=True), (
        f"{column} changed when later data was removed -- it is looking ahead"
    )
print("Point-in-time check passed: truncating the series doesn't change earlier feature values.")

# --------------------------------------------------------------------------
# 2. Every detection method produces usable, persistent labels
# --------------------------------------------------------------------------
print("\n=== Regime detection ===")
results = {}
for method in REGIME_METHODS:
    result = detect_regimes(df, method=method, n_regimes=3, fit_frac=0.7)
    labels = result.labels

    assert labels.index.equals(df.index), f"{method}: labels not aligned to the price index"
    assert not labels.isna().any(), f"{method}: NaN in labels"
    assert labels.isin(list(result.names) + [UNKNOWN]).all(), f"{method}: label outside the named set"
    assert result.valid().sum() > 400, f"{method}: only {result.valid().sum()} labelled days"
    assert set(result.names) == set(range(len(result.names))), f"{method}: regime IDs are not 0..k-1"

    stability = regime_stability(labels)
    results[method] = result
    print(f"{method:11s} k={len(result.names)} labelled={stability['labelled_days']:4d} "
          f"episodes={stability['n_episodes']:3d} avg_duration={stability['avg_duration']:5.1f}d "
          f"causal={result.causal}")

# Regime IDs are ordered by volatility -- regime 0 must be the calmest.
# Everything downstream (color ramps, default parameter ladders, size maps)
# depends on this holding.
for method, result in results.items():
    returns = df["Close"].pct_change()
    vols = [returns[result.labels == r].std() for r in sorted(result.names)]
    vols = [v for v in vols if not pd.isna(v)]
    assert vols == sorted(vols), f"{method}: regime IDs not ordered by volatility: {vols}"
print("Ordering check passed: regime 0 is the calmest under every method.")

# --------------------------------------------------------------------------
# 3. The HMM should find persistent states, not noise
# --------------------------------------------------------------------------
print("\n=== HMM persistence ===")
hmm_result = results["hmm"]
matrix = hmm_result.transition_matrix()
diagonal = np.diag(matrix.to_numpy())
print(matrix.round(3).to_string())
assert (diagonal > 0.85).all(), f"HMM states are not persistent enough: diagonal={diagonal}"
assert hmm_result.model.converged_, "HMM did not converge"
print(f"Converged in {hmm_result.model.n_iter_} EM iterations; "
      f"min persistence {diagonal.min():.3f}.")

summary = regime_summary(hmm_result, df)
assert len(summary) >= 2, "HMM produced fewer than two populated regimes"
vol_spread = summary["ann_volatility"].max() - summary["ann_volatility"].min()
assert vol_spread > 0.05, (
    f"Detected regimes differ by only {vol_spread:.1%} in volatility -- on data built with "
    "three very different states, that means detection failed"
)
print(f"\n{summary[['name', 'days', 'ann_return', 'ann_volatility', 'max_drawdown']].round(3).to_string(index=False)}")

# --------------------------------------------------------------------------
# 4. Smoothing is causal and actually reduces flicker
# --------------------------------------------------------------------------
print("\n=== Label smoothing ===")
raw = detect_regimes(df, method="kmeans", n_regimes=3, fit_frac=0.7, smooth="none")
raw_episodes = len(regime_episodes(raw.labels))
for method, kwargs in (("min_duration", {"min_duration": 5}), ("median", {"window": 5})):
    smoothed = smooth_labels(raw.labels, method=method, **kwargs)
    smoothed_episodes = len(regime_episodes(smoothed))
    assert smoothed_episodes <= raw_episodes, f"{method} increased the episode count"
    print(f"{method:13s} episodes {raw_episodes} -> {smoothed_episodes}")

# Causality: smoothing row t must not depend on anything after t.
prefix = smooth_labels(raw.labels.iloc[:1200], method="min_duration", min_duration=5)
full = smooth_labels(raw.labels, method="min_duration", min_duration=5)
assert (prefix == full.iloc[:1200]).all(), "min_duration smoothing is using future labels"
print("Causality check passed: smoothed labels don't change when later data is removed.")

# --------------------------------------------------------------------------
# 5. Walk-forward detection labels nothing it hasn't earned
# --------------------------------------------------------------------------
print("\n=== Walk-forward regime detection ===")
wf_regimes = detect_regimes_walk_forward(df, method="hmm", n_regimes=3)
assert wf_regimes.causal, "walk-forward result is not marked causal"
first_labelled = wf_regimes.labels[wf_regimes.labels != UNKNOWN].index[0]
assert wf_regimes.labels.loc[:first_labelled].iloc[:-1].eq(UNKNOWN).all(), (
    "walk-forward produced labels before its initial training window"
)
print(f"Refits: {wf_regimes.meta['n_refits']}, first label {first_labelled.date()}, "
      f"labelled days {int(wf_regimes.valid().sum())}")

# --------------------------------------------------------------------------
# 6. backtest.py stays backward compatible and costs behave
# --------------------------------------------------------------------------
print("\n=== Backtest: costs and regime passthrough ===")
signal = STRATEGIES["sma_crossover"](df)
free = run_backtest(df, signal)
costed = run_backtest(df, signal, cost_bps=10, regimes=hmm_result.labels)

assert np.allclose(free["cost"], 0), "cost_bps=0 charged a cost"
assert costed["cost"].sum() > 0, "cost_bps=10 charged nothing"
assert costed["strategy_return"].sum() < free["strategy_return"].sum(), "costs did not reduce returns"
assert "regime" in costed.columns and costed["regime"].isin(list(hmm_result.names) + [UNKNOWN]).all()
print(f"Total cost drag at 10bps: {costed['cost'].sum():.2%} of notional over the period.")

by_regime = performance_by_regime(costed, hmm_result.labels, hmm_result.names)
assert len(by_regime) >= 2, "per-regime performance table is degenerate"
assert (by_regime["days"].sum() == int(hmm_result.valid().sum())), (
    "per-regime day counts don't add up to the labelled days"
)
print(f"\n{by_regime[['name', 'days', 'sharpe_ratio', 'max_drawdown', 'exposure']].round(3).to_string(index=False)}")

# --------------------------------------------------------------------------
# 7. Adaptive strategies keep the strategy interface
# --------------------------------------------------------------------------
print("\n=== Adaptive strategies ===")
for name, fn in ADAPTIVE_STRATEGIES.items():
    position = fn(df, regimes=hmm_result)

    assert position.index.equals(df.index), f"{name}: position not aligned to the price index"
    assert not position.isna().any(), f"{name}: NaN in position"
    assert ((position >= 0) & (position <= 1)).all(), f"{name}: position outside [0, 1]"

    stats = full_report(run_backtest(df, position, cost_bps=5))
    assert -1.0 <= stats["max_drawdown"] <= 0.0, f"{name}: max_drawdown out of range"
    assert 0.0 <= stats["exposure"] <= 1.0, f"{name}: exposure out of range"
    print(f"{name:22s} return={stats['total_return']:8.2%} sharpe={stats['sharpe_ratio']:6.2f} "
          f"max_dd={stats['max_drawdown']:7.2%} exposure={stats['exposure']:.2f} "
          f"turnover={stats['turnover']:5.1f}")

# The base strategies must still be strictly binary -- adaptive.py exists
# precisely so that the fractional-position logic never leaked into them.
for name, fn in STRATEGIES.items():
    assert fn(df).isin([0, 1]).all(), f"{name} is no longer binary"
print("Base strategies are still binary; fractional positions are confined to adaptive.py.")

# --------------------------------------------------------------------------
# 8. Automatic choices are learned from the past only
# --------------------------------------------------------------------------
print("\n=== Auto-selection honesty ===")
described = describe_choices(df, regimes=hmm_result)
learn_end = described["learn_end"]
assert learn_end < df.index[-1], "the learning window covers the whole sample"
assert (described["table"]["days"] > 0).any(), "no learning-window days found"
assert set(described["choices"]) == set(hmm_result.names), "a regime got no decision"
print(f"Learning window ends {learn_end.date()} ({len(df.loc[:learn_end])} of {len(df)} rows).")
print("Choices:", {hmm_result.names[r]: v for r, v in described["choices"].items()})

filtered = describe_filter(df, base="sma_crossover", regimes=hmm_result)
print(f"Filter allow-list: {[hmm_result.names[r] for r in filtered['allowed']] or 'nothing (flat throughout)'}")

# Changing data AFTER the learning window must not change what was learned.
mutated = df.copy()
tail = mutated.index > learn_end
mutated.loc[tail, "Close"] = mutated.loc[tail, "Close"] * 1.5
mutated_choices = describe_choices(mutated, regimes=hmm_result)["choices"]
assert mutated_choices == described["choices"], (
    "the learned per-regime choice changed when only post-learning data changed -- it is looking ahead"
)
print("Lookahead check passed: post-learning data doesn't change what the auto-selector learned.")

# --------------------------------------------------------------------------
# 9. Regime-aware ML paths
# --------------------------------------------------------------------------
print("\n=== Regime-aware ML ===")
baseline = model_report(df, model_type="logistic")
for mode in ("feature", "conditional"):
    report = model_report(df, model_type="logistic", regimes=hmm_result.labels, regime_mode=mode)
    assert 0 <= report["test_accuracy"] <= 1, f"{mode}: implausible test accuracy"
    assert "by_regime" in report, f"{mode}: missing per-regime breakdown"
    print(f"{mode:12s} train={report['train_accuracy']:.1%} test={report['test_accuracy']:.1%} "
          f"base_rate={report['test_base_rate']:.1%} regimes_reported={len(report['by_regime'])}")
print(f"{'no regimes':12s} train={baseline['train_accuracy']:.1%} test={baseline['test_accuracy']:.1%} "
      f"base_rate={baseline['test_base_rate']:.1%}")

# --------------------------------------------------------------------------
# 10. Validation layer
# --------------------------------------------------------------------------
print("\n=== Validation layer ===")
rolling = rolling_walk_forward(df, STRATEGIES["sma_crossover"], train_days=756, test_days=126)
assert rolling["n_folds"] >= 5, f"only {rolling['n_folds']} folds"
assert 0 <= rolling["pct_folds_positive"] <= 1
assert len(rolling["oos_equity"]) == rolling["n_folds"] * 126
print(f"Rolling walk-forward: {rolling['n_folds']} folds, "
      f"{rolling['pct_folds_positive']:.0%} positive, "
      f"median Sharpe {rolling['median_sharpe']:.2f}, worst {rolling['worst_fold_sharpe']:.2f}")

comparison = compare_strategies(
    df,
    {name: (fn, {"regimes": hmm_result} if name in ADAPTIVE_STRATEGIES else {})
     for name, fn in ALL_STRATEGIES.items()},
    cost_bps=5,
)
assert comparison["error"].isna().all(), f"strategies errored: {comparison[comparison['error'].notna()]}"
assert len(comparison) == len(ALL_STRATEGIES)
print(f"\n{comparison[['strategy', 'is_sharpe', 'oos_sharpe', 'sharpe_decay', 'turnover']].round(2).to_string(index=False)}")

# Every strategy must survive every validation entry point. This loop is
# here because it wasn't: evaluate_with_regimes() originally took
# **strategy_params, which collided with its own `regimes` argument the
# moment an adaptive wrapper (which takes `regimes` too) was passed
# through it. Nothing caught that until this ran.
print("\n=== Validation entry points x every strategy ===")
for name, fn in ALL_STRATEGIES.items():
    strategy_params = {"regimes": hmm_result} if name in ADAPTIVE_STRATEGIES else {}

    evaluation = evaluate_with_regimes(
        df, fn, hmm_result, cost_bps=5, strategy_params=strategy_params
    )
    assert set(evaluation) >= {"in_sample", "out_sample", "in_sample_by_regime",
                               "out_sample_by_regime", "regime_mix"}
    assert not evaluation["regime_mix"].empty, f"{name}: empty regime mix"

    fold_result = rolling_walk_forward(df, fn, cost_bps=5, **strategy_params)
    assert fold_result["n_folds"] >= 5, f"{name}: too few folds"

    mix = evaluation["regime_mix"]
    shift = float((mix["out_sample"] - mix["in_sample"]).abs().max())
    print(f"{name:22s} decay={evaluation['in_sample']['sharpe_ratio'] - evaluation['out_sample']['sharpe_ratio']:+6.2f} "
          f"folds={fold_result['n_folds']:2d} positive={fold_result['pct_folds_positive']:.0%} "
          f"max regime-mix shift={shift:.0%}")

# The original single-split entry point must be untouched.
legacy = evaluate_out_of_sample(df, STRATEGIES["sma_crossover"])
assert set(legacy) == {"split_date", "in_sample", "out_sample"}, "evaluate_out_of_sample changed shape"
print("\nevaluate_out_of_sample() signature and return shape unchanged.")

print("\nAll regime, adaptive and validation checks passed on synthetic data.")
