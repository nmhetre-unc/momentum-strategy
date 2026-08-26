"""
Quant intern exercises, with automated checks that run against whatever
data the user actually loaded.

Each exercise has a prompt, an expected observation, hints, a written
explanation, and -- where it's possible to verify mechanically -- a
`check` function that computes the relevant numbers and reports whether
the phenomenon showed up.

A deliberate design choice: a check returning `passed=False` is not a
failure grade. Several of these exercises test whether a well-known
effect is present in YOUR data, and sometimes it isn't. A random forest
that didn't overfit on your ticker, or a choppy regime where trend
following happened to do fine, is a real result and the message says so.
Interns who learn to read "the expected effect is absent here, and here
is the number" as information rather than as an error are learning the
actual job.

Checks take a context dict:

    {"df": DataFrame, "ticker": str, "answer": <optional user answer>}

and return a CheckResult. Nothing here imports streamlit, so the whole
set can be run from a script -- see run_all() at the bottom.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from adaptive import volatility_targeted
from analytics import full_report, performance_by_regime
from backtest import run_backtest
from ml_strategy import model_report
from regime import detect_regimes, detect_regimes_walk_forward
from strategies import STRATEGIES
from walk_forward import evaluate_out_of_sample


@dataclass
class CheckResult:
    passed: bool
    message: str
    evidence: object = None      # DataFrame or dict, rendered under the message
    detail: str = ""


@dataclass
class Exercise:
    key: str
    title: str
    level: str
    prompt: str
    expected: str
    hints: list = field(default_factory=list)
    explanation: str = ""
    check: object = None
    answer_prompt: str = ""      # non-empty means the check wants a user answer
    answer_options: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _regimes(df: pd.DataFrame, method: str = "hmm", n_regimes: int = 3, fit_frac: float = 0.6):
    return detect_regimes(df, method=method, n_regimes=n_regimes, fit_frac=fit_frac)


def _regime_table(df: pd.DataFrame, strategy: str = "sma_crossover", params: dict = None,
                  cost_bps: float = 0.0):
    """Per-regime performance for one strategy, plus the regime result."""
    regimes = _regimes(df)
    signal = STRATEGIES[strategy](df, **(params or {}))
    result = run_backtest(df, signal, cost_bps=cost_bps, regimes=regimes.labels)
    table = performance_by_regime(result, regimes.labels, regimes.names)
    return table, regimes, result


def _choppiest_regime(regimes) -> int:
    """
    The regime with the lowest average efficiency ratio -- the one where
    price travelled the most and got the least distance from it. That is
    the definition of 'sideways', and it's what a trend follower dislikes.
    """
    scores = {}
    for regime_id in sorted(r for r in regimes.labels.unique() if r >= 0):
        mask = regimes.labels == regime_id
        scores[int(regime_id)] = regimes.features.loc[mask, "efficiency_ratio"].mean()
    valid = {k: v for k, v in scores.items() if not pd.isna(v)}
    return min(valid, key=valid.get) if valid else -1


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_sma_sideways(ctx: dict) -> CheckResult:
    df = ctx["df"]
    table, regimes, _ = _regime_table(df, "sma_crossover")
    if table.empty:
        return CheckResult(False, "No regimes were labelled — try a longer date range.")

    choppy = _choppiest_regime(regimes)
    choppy_row = table[table["regime"] == choppy]
    others = table[table["regime"] != choppy]
    if choppy_row.empty or others.empty:
        return CheckResult(False, "Not enough distinct regimes to compare.", table)

    choppy_sharpe = float(choppy_row["sharpe_ratio"].iloc[0])
    best_other = float(others["sharpe_ratio"].max())
    name = choppy_row["name"].iloc[0]
    days = int(choppy_row["days"].iloc[0])

    passed = choppy_sharpe < best_other
    if passed:
        message = (
            f"Confirmed. The most sideways regime is **{name}** (lowest efficiency ratio), where "
            f"the crossover's Sharpe is {choppy_sharpe:.2f} over {days} days, against "
            f"{best_other:.2f} in its best regime."
        )
    else:
        message = (
            f"Not confirmed on this data. The crossover's Sharpe in the sideways regime "
            f"(**{name}**, {days} days) is {choppy_sharpe:.2f}, which is not below its best "
            f"other regime at {best_other:.2f}. Check the `days` column — a short regime gives "
            f"a Sharpe with a very wide error bar."
        )
    return CheckResult(passed, message, table[["name", "days", "sharpe_ratio", "total_return", "max_drawdown", "exposure"]])


def check_ml_accuracy_gap(ctx: dict) -> CheckResult:
    df = ctx["df"]
    rows = []
    for model_type in ("logistic", "random_forest"):
        report = model_report(df, model_type=model_type)
        rows.append({
            "model": model_type,
            "train_accuracy": report["train_accuracy"],
            "test_accuracy": report["test_accuracy"],
            "gap": report["train_accuracy"] - report["test_accuracy"],
            "base_rate": report["test_base_rate"],
            "beats_base_rate": report["test_accuracy"] > report["test_base_rate"],
        })
    table = pd.DataFrame(rows)

    logistic, forest = table.iloc[0], table.iloc[1]
    passed = forest["gap"] > logistic["gap"]
    message = (
        f"Random forest gap: {forest['gap']:+.1%} (train {forest['train_accuracy']:.1%} → "
        f"test {forest['test_accuracy']:.1%}). Logistic gap: {logistic['gap']:+.1%}. "
    )
    message += (
        "The higher-capacity model has the larger gap, which is the overfitting signature."
        if passed else
        "The forest's gap is not larger here — unusual, and worth investigating before you trust it."
    )
    if not table["beats_base_rate"].any():
        message += (
            f" Note that **neither model beats the base rate** of {forest['base_rate']:.1%}, "
            "so neither has learned anything usable."
        )
    return CheckResult(passed, message, table)


def check_best_regime(ctx: dict) -> CheckResult:
    df = ctx["df"]
    strategy = ctx.get("strategy", "sma_crossover")
    table, regimes, _ = _regime_table(df, strategy)
    if table.empty:
        return CheckResult(False, "No regimes were labelled — try a longer date range.")

    best = table.loc[table["sharpe_ratio"].idxmax()]
    answer = ctx.get("answer")
    display = table[["name", "days", "sharpe_ratio", "total_return", "max_drawdown"]]

    if answer is None:
        return CheckResult(False, "Pick a regime above, then re-run the check.", display)

    correct = str(answer) == str(best["name"])
    message = (
        f"Correct — **{best['name']}** has the highest Sharpe ({best['sharpe_ratio']:.2f}) "
        f"over {int(best['days'])} days."
        if correct else
        f"Not quite. The highest Sharpe is **{best['name']}** at {best['sharpe_ratio']:.2f} "
        f"over {int(best['days'])} days; you picked {answer}."
    )
    if best["days"] < 200:
        message += (
            f" Read that with suspicion though: {int(best['days'])} days gives an annualized "
            f"Sharpe a standard error of roughly ±{np.sqrt(252 / best['days']):.1f}."
        )
    return CheckResult(correct, message, display)


def check_random_forest_overfit(ctx: dict) -> CheckResult:
    df = ctx["df"]
    report = model_report(df, model_type="random_forest")
    wf = evaluate_out_of_sample(df, STRATEGIES["ml_direction"], model_type="random_forest")

    gap = report["train_accuracy"] - report["test_accuracy"]
    passed = gap > 0.10
    evidence = {
        "train_accuracy": f"{report['train_accuracy']:.1%}",
        "test_accuracy": f"{report['test_accuracy']:.1%}",
        "test_base_rate": f"{report['test_base_rate']:.1%}",
        "accuracy_gap": f"{gap:+.1%}",
        "in_sample_sharpe": f"{wf['in_sample']['sharpe_ratio']:.2f}",
        "out_of_sample_sharpe": f"{wf['out_sample']['sharpe_ratio']:.2f}",
    }
    message = (
        f"Overfitting confirmed: a {gap:.1%} accuracy gap, and Sharpe falling from "
        f"{wf['in_sample']['sharpe_ratio']:.2f} in-sample to {wf['out_sample']['sharpe_ratio']:.2f} "
        f"out-of-sample."
        if passed else
        f"The gap here is only {gap:.1%}, below the 10% threshold. Either this ticker has more "
        f"learnable structure than most, or the training window is small enough that even the "
        f"in-sample accuracy is modest. Compare against the base rate of {report['test_base_rate']:.1%}."
    )
    return CheckResult(passed, message, evidence)


def check_walk_forward(ctx: dict) -> CheckResult:
    df = ctx["df"]
    strategy = ctx.get("strategy", "sma_crossover")
    wf = evaluate_out_of_sample(df, STRATEGIES[strategy])
    decay = wf["in_sample"]["sharpe_ratio"] - wf["out_sample"]["sharpe_ratio"]
    overfit = decay > 0.5

    table = pd.DataFrame({
        "In-sample": wf["in_sample"], "Out-of-sample": wf["out_sample"],
    }).loc[["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "num_trades", "exposure"]]

    answer = ctx.get("answer")
    verdict = "Yes — signs of overfitting" if overfit else "No — held up reasonably"
    if answer is None:
        return CheckResult(False, "Give your reading above, then re-run the check.", table)

    correct = str(answer) == verdict
    message = (
        f"Sharpe went {wf['in_sample']['sharpe_ratio']:.2f} → {wf['out_sample']['sharpe_ratio']:.2f}, "
        f"a decay of {decay:.2f}. Verdict: **{verdict}**. "
    )
    message += "Your reading matches." if correct else f"You said: {answer}."
    if wf["out_sample"]["num_trades"] < 10:
        message += (
            f" Caveat worth stating in any write-up: only {wf['out_sample']['num_trades']} trades "
            "out-of-sample, so this verdict rests on very few independent bets."
        )
    return CheckResult(correct, message, table)


def check_parameter_sensitivity(ctx: dict) -> CheckResult:
    df = ctx["df"]
    rows = []
    for short_window in (10, 20, 30, 50, 70, 90):
        wf = evaluate_out_of_sample(
            df, STRATEGIES["sma_crossover"], short_window=short_window, long_window=200
        )
        rows.append({
            "short_window": short_window,
            "is_sharpe": wf["in_sample"]["sharpe_ratio"],
            "oos_sharpe": wf["out_sample"]["sharpe_ratio"],
            "oos_max_dd": wf["out_sample"]["max_drawdown"],
        })
    table = pd.DataFrame(rows)

    spread = float(table["oos_sharpe"].max() - table["oos_sharpe"].min())
    best_is = int(table.loc[table["is_sharpe"].idxmax(), "short_window"])
    best_oos = int(table.loc[table["oos_sharpe"].idxmax(), "short_window"])
    unstable = best_is != best_oos

    message = (
        f"Out-of-sample Sharpe ranges {table['oos_sharpe'].min():.2f} to "
        f"{table['oos_sharpe'].max():.2f} across these windows — a spread of {spread:.2f} "
        f"from a parameter you have no principled reason to set one way or the other. "
    )
    message += (
        f"The best in-sample window ({best_is}) is **not** the best out-of-sample one "
        f"({best_oos}), which is precisely why picking parameters on in-sample results "
        f"does not work."
        if unstable else
        f"Here the best window is {best_is} both in- and out-of-sample. That's a good sign, "
        f"but check the spread: if every window works, the parameter isn't doing much."
    )
    return CheckResult(unstable, message, table)


def check_lookahead_gap(ctx: dict) -> CheckResult:
    """
    The headline demonstration: the same regime-filtered strategy run on
    full-sample labels and on walk-forward labels.
    """
    df = ctx["df"]
    from adaptive import regime_filtered

    try:
        honest_regimes = detect_regimes_walk_forward(df, method="hmm", n_regimes=3)
    except ValueError as exc:
        return CheckResult(False, f"Not enough history for walk-forward regimes: {exc}")

    leaky_regimes = detect_regimes(df, method="hmm", n_regimes=3, fit_frac=1.0)

    rows = []
    for label, regimes in (("Full-sample labels (leaky)", leaky_regimes),
                           ("Walk-forward labels (honest)", honest_regimes)):
        signal = regime_filtered(df, base="sma_crossover", regimes=regimes)
        stats = full_report(run_backtest(df, signal))
        rows.append({
            "labels": label,
            "sharpe_ratio": stats["sharpe_ratio"],
            "total_return": stats["total_return"],
            "max_drawdown": stats["max_drawdown"],
            "exposure": stats["exposure"],
        })
    table = pd.DataFrame(rows)

    gap = float(table.iloc[0]["sharpe_ratio"] - table.iloc[1]["sharpe_ratio"])
    passed = gap > 0
    message = (
        f"Full-sample labels give a Sharpe of {table.iloc[0]['sharpe_ratio']:.2f}; honest "
        f"walk-forward labels give {table.iloc[1]['sharpe_ratio']:.2f}. The difference of "
        f"**{gap:+.2f}** is lookahead bias — performance that exists only because the regime "
        f"model was allowed to see the future."
        if passed else
        f"The honest version scored higher here ({gap:+.2f}), which happens: walk-forward "
        f"labels are noisier but not automatically worse, and the walk-forward version also "
        f"trades a shorter effective history. Compare the exposure column before concluding."
    )
    return CheckResult(passed, message, table)


def check_cost_sensitivity(ctx: dict) -> CheckResult:
    df = ctx["df"]
    rows = []
    for name in ("sma_crossover", "momentum", "mean_reversion", "ml_direction"):
        signal = STRATEGIES[name](df)
        free = full_report(run_backtest(df, signal, cost_bps=0))
        costed = full_report(run_backtest(df, signal, cost_bps=10))
        rows.append({
            "strategy": name,
            "turnover": free["turnover"],
            "sharpe_0bps": free["sharpe_ratio"],
            "sharpe_10bps": costed["sharpe_ratio"],
            "sharpe_lost": free["sharpe_ratio"] - costed["sharpe_ratio"],
        })
    table = pd.DataFrame(rows)

    rank_free = list(table.sort_values("sharpe_0bps", ascending=False)["strategy"])
    rank_costed = list(table.sort_values("sharpe_10bps", ascending=False)["strategy"])
    changed = rank_free != rank_costed
    worst = table.loc[table["sharpe_lost"].idxmax()]

    message = (
        f"**{worst['strategy']}** loses the most to costs: {worst['sharpe_lost']:.2f} of Sharpe "
        f"at 10bps, on turnover of {worst['turnover']:.0f}x a year. "
    )
    message += (
        f"The ranking changes once costs are on — frictionless: {' > '.join(rank_free)}; "
        f"at 10bps: {' > '.join(rank_costed)}."
        if changed else
        "The ranking survives costs here, but note how unevenly the damage lands."
    )
    return CheckResult(changed, message, table)


def check_vol_targeting(ctx: dict) -> CheckResult:
    df = ctx["df"]
    base_signal = STRATEGIES["sma_crossover"](df)
    base = full_report(run_backtest(df, base_signal, cost_bps=5))
    sized = full_report(run_backtest(df, volatility_targeted(df, base="sma_crossover"), cost_bps=5))

    table = pd.DataFrame({"Unsized": base, "Vol-targeted": sized}).loc[
        ["total_return", "cagr", "annualized_volatility", "sharpe_ratio",
         "max_drawdown", "exposure", "turnover"]
    ]
    dd_better = sized["max_drawdown"] > base["max_drawdown"]   # both negative
    sharpe_better = sized["sharpe_ratio"] > base["sharpe_ratio"]

    message = (
        f"Max drawdown {base['max_drawdown']:.1%} → {sized['max_drawdown']:.1%}, "
        f"Sharpe {base['sharpe_ratio']:.2f} → {sized['sharpe_ratio']:.2f}, "
        f"exposure {base['exposure']:.2f} → {sized['exposure']:.2f}. "
    )
    if dd_better and sharpe_better:
        message += "Both improved — the usual result, and it came from sizing alone, with no change to the entry signal."
    elif dd_better:
        message += "Drawdown improved but Sharpe didn't. Check exposure: if targeting cut you to a small average position, absolute return falls even as risk does."
    else:
        message += "Neither improved here. Try a target_vol closer to the asset's own realized volatility — targeting 15% on an asset that realizes 12% mostly just caps the position."
    return CheckResult(dd_better, message, table)


def check_beats_benchmark(ctx: dict) -> CheckResult:
    df = ctx["df"]
    strategy = ctx.get("strategy", "sma_crossover")
    signal = STRATEGIES[strategy](df)
    result = run_backtest(df, signal, cost_bps=5)
    split = result.index[int(len(result) * 0.7)]
    oos = result.loc[split:]

    strategy_stats = full_report(oos)
    benchmark = oos.copy()
    benchmark["strategy_return"] = oos["daily_return"]
    benchmark["position"] = 1.0
    benchmark_stats = full_report(benchmark)

    table = pd.DataFrame({strategy: strategy_stats, "buy_and_hold": benchmark_stats}).loc[
        ["total_return", "cagr", "sharpe_ratio", "max_drawdown", "exposure"]
    ]
    passed = strategy_stats["sharpe_ratio"] > benchmark_stats["sharpe_ratio"]
    message = (
        f"Out-of-sample, {strategy} has Sharpe {strategy_stats['sharpe_ratio']:.2f} against "
        f"buy-and-hold's {benchmark_stats['sharpe_ratio']:.2f} "
        f"({strategy_stats['total_return']:.1%} vs {benchmark_stats['total_return']:.1%} return). "
    )
    message += (
        "It clears the bar — now check whether it does so with less drawdown too."
        if passed else
        "It does not clear the bar. This is the most common outcome and the most commonly "
        "omitted comparison. A strategy that underperforms buy-and-hold has to justify itself "
        "on drawdown or exposure instead."
    )
    return CheckResult(passed, message, table)


# --------------------------------------------------------------------------
# The exercise set
# --------------------------------------------------------------------------
EXERCISES = [
    Exercise(
        key="sma_sideways",
        title="Why does SMA crossover fail in sideways markets?",
        level="1 · Foundations",
        prompt=(
            "Run the SMA crossover on SPY (2010-2025). Then open the Regimes tab and look at "
            "its performance broken down by regime. Identify the sideways/choppy regime and "
            "explain, mechanically, what the strategy does there."
        ),
        expected=(
            "The crossover's Sharpe ratio in the choppy regime should be clearly worse than in "
            "the trending regimes, often negative. Look for a stair-step equity curve: long flat-"
            "to-down stretches punctuated by a few large gains."
        ),
        hints=[
            "Find the sideways regime by its efficiency ratio — the ratio of net move to total distance travelled. Low means the market went nowhere loudly.",
            "Draw the two moving averages on a range-bound stretch and mark each crossing. Where does each buy and each sell land relative to the local high and low?",
            "Count the trades in that regime versus the trending one.",
        ],
        explanation=(
            "The crossover has no concept of 'range'. It only knows whether the fast average is "
            "above the slow one, and in a sideways market price oscillates across both. Each "
            "oscillation triggers a buy near a local top (price has just risen enough to pull the "
            "fast average up) and a sell near a local bottom. The strategy systematically buys "
            "high and sells low, and it does so repeatedly.\n\n"
            "This is not bad luck or a tuning problem. It is what the rule does by construction, "
            "and it is why trend-following systems are described as paying an insurance premium in "
            "quiet markets to collect a large payout in trending ones. The whole return profile is "
            "many small losses and a few large gains — which is also why the win rate is a useless "
            "metric here, and why the strategy is psychologically hard to hold.\n\n"
            "The practical response is regime filtering: don't trade the crossover in the regime "
            "where it structurally loses. Whether you can identify that regime *in real time* is a "
            "separate and much harder question."
        ),
        check=check_sma_sideways,
    ),
    Exercise(
        key="ml_accuracy",
        title="Compare in-sample and out-of-sample ML accuracy",
        level="2 · Validation",
        prompt=(
            "Train both the logistic regression and the random forest. Record train accuracy, "
            "test accuracy, and the base rate for each. Then state which model you would deploy "
            "and why."
        ),
        expected=(
            "Random forest: high train accuracy (often 80%+), test accuracy near or below 50%. "
            "Logistic: train accuracy in the high 50s, test accuracy close to it. The correct "
            "deployment choice is the model with the smaller gap, not the higher train accuracy."
        ),
        hints=[
            "Compare test accuracy against the base rate, not against 50%. On daily equity data the base rate is about 53%.",
            "The gap between train and test accuracy is the quantity of interest, not either number alone.",
            "Ask what each model's capacity is: how many effectively free parameters does it have relative to the number of training rows?",
        ],
        explanation=(
            "Both models see the same features and the same rows. The difference is capacity. The "
            "random forest can carve the feature space into hundreds of regions and memorize what "
            "happened in each; logistic regression can only fit one linear boundary.\n\n"
            "When the signal-to-noise ratio is as low as it is in daily equity direction, capacity "
            "is a liability. The forest's 86% train accuracy is not 86% of a real relationship — it "
            "is a near-complete description of the noise in the training rows, which does not "
            "transfer.\n\n"
            "The counterintuitive rule this teaches: **less signal means you should use a simpler "
            "model, not a more powerful one.** Most people's instinct on a disappointing result is "
            "to reach for a bigger model, which makes the problem strictly worse.\n\n"
            "And one more step: even the better model probably doesn't beat the base rate. A model "
            "at 53% test accuracy has matched 'always predict up' and is not worth deploying at all."
        ),
        check=check_ml_accuracy_gap,
    ),
    Exercise(
        key="best_regime",
        title="Which regime does your strategy perform best in?",
        level="3 · Regimes",
        prompt=(
            "Pick a strategy and detect regimes on the same data. Before looking at the table, "
            "predict which regime it will do best in and write down why. Then check."
        ),
        expected=(
            "Trend strategies should do best in the low-volatility uptrend regime; mean-reversion "
            "in the choppy or range regime. Your prediction being wrong is more interesting than "
            "it being right — it means the strategy isn't doing what you assumed."
        ),
        hints=[
            "Look at the regime summary table first: which regime has positive drift and low volatility?",
            "A strategy that is long-only will do well in any regime where the asset went up. Separate 'the strategy worked' from 'the market went up'.",
            "Check the days column before trusting any per-regime Sharpe.",
        ],
        explanation=(
            "For a long-only trend strategy the answer is usually the calm uptrend regime, but the "
            "reason matters: partly the strategy works there, and partly the asset simply rose. "
            "Comparing against the benchmark's per-regime numbers separates the two.\n\n"
            "The more useful finding is normally the worst regime rather than the best. Knowing "
            "where a strategy loses is directly actionable — you can stop trading it there — while "
            "knowing where it wins mostly tells you what you already suspected.\n\n"
            "Watch the sample size. A regime covering 120 days gives an annualized Sharpe with a "
            "standard error near ±1.5. Two regimes whose Sharpes differ by 0.4 are, on that "
            "evidence, indistinguishable."
        ),
        check=check_best_regime,
        answer_prompt="Which regime do you think has the highest Sharpe?",
    ),
    Exercise(
        key="rf_overfit",
        title="Explain why the random forest overfits",
        level="2 · Validation",
        prompt=(
            "Run the random forest and examine train accuracy, test accuracy, and the resulting "
            "strategy's in- and out-of-sample Sharpe. Then write a three-sentence explanation of "
            "the mechanism — not just the observation."
        ),
        expected=(
            "A large positive gap between train and test accuracy, and an out-of-sample Sharpe "
            "well below the in-sample one, often negative. Test accuracy below the base rate is "
            "common and is the strongest version of the finding."
        ),
        hints=[
            "How many leaves does a 200-tree forest of depth 5 have in total, and how many training rows are there per leaf?",
            "What would test accuracy be if the model had learned nothing at all? Compare against that, not against zero.",
            "Try max_depth=2 mentally: would you expect the gap to grow or shrink?",
        ],
        explanation=(
            "Mechanism, in three steps.\n\n"
            "**1. The target is nearly unpredictable.** Daily direction is roughly 53/47. There is "
            "very little structure to learn.\n\n"
            "**2. The model has far more capacity than the signal justifies.** 200 trees of depth 5 "
            "give thousands of leaves; each fits a handful of training rows. With enough regions you "
            "can label almost any training set perfectly, and the forest does — that is the 86%.\n\n"
            "**3. What it fitted doesn't exist.** The regions were carved around noise, so on new "
            "data they are arbitrary. Test accuracy lands at chance, and can land below it, because "
            "the memorized patterns are not neutral — they encode relationships that were true only "
            "in the training rows.\n\n"
            "The dashboard's train/test warning triggers on this. Treat that warning as the model "
            "working correctly and telling you something true, rather than as a problem to tune away."
        ),
        check=check_random_forest_overfit,
    ),
    Exercise(
        key="walk_forward",
        title="Run walk-forward validation and interpret it",
        level="2 · Validation",
        prompt=(
            "Run walk-forward validation on a strategy of your choice. Compare in-sample and "
            "out-of-sample Sharpe, drawdown and trade count, then decide: is this evidence of "
            "overfitting, or of a strategy that held up?"
        ),
        expected=(
            "A modest decay is normal. A collapse — or a sign flip — indicates the in-sample "
            "number was largely fitted. Also check the out-of-sample trade count: too few trades "
            "means the verdict rests on very little evidence either way."
        ),
        hints=[
            "The size of the gap matters more than the level of either number.",
            "Look at drawdown too. A strategy that keeps its return but doubles its drawdown out-of-sample has still degraded.",
            "How many independent bets are in the out-of-sample period? Under 10 and you cannot conclude much.",
        ],
        explanation=(
            "Walk-forward answers one question: does performance survive on data that played no "
            "part in building the strategy?\n\n"
            "A **small gap** means the process generalizes — that's what you want, and it matters "
            "more than a high out-of-sample number. A **large gap** means the in-sample result was "
            "substantially a description of that specific period.\n\n"
            "Two things this cannot do. It cannot protect you if you look at the out-of-sample "
            "result, adjust the strategy, and look again — do that three times and the holdout is "
            "in-sample. And a single split is one draw from a distribution: use "
            "`rolling_walk_forward()` for ten or fifteen consecutive holdouts, and read the share "
            "of positive folds rather than their average.\n\n"
            "The trade count is the caveat people skip. Two trades out-of-sample is two coin flips, "
            "and no Sharpe ratio computed on it means anything."
        ),
        check=check_walk_forward,
        answer_prompt="Your reading of the result:",
        answer_options=["Yes — signs of overfitting", "No — held up reasonably"],
    ),
    Exercise(
        key="param_sensitivity",
        title="Modify strategy parameters and observe the risk changes",
        level="1 · Foundations",
        prompt=(
            "Sweep the SMA short window from 10 to 90 and record out-of-sample Sharpe and max "
            "drawdown for each. Then answer: how would you have chosen this parameter in advance, "
            "and what does the spread tell you about the result you'd have got?"
        ),
        expected=(
            "A wide spread of outcomes across windows, and — usually — a best in-sample window "
            "that is not the best out-of-sample one. That mismatch is the entire argument against "
            "choosing parameters on in-sample performance."
        ),
        hints=[
            "Look at the spread between the best and worst out-of-sample Sharpe. Would you have picked the best one in advance?",
            "Is there a plateau of similar values, or a single spike? Spikes are noise; plateaus are closer to real.",
            "Try the same sweep on a different ticker. Does the best window move?",
        ],
        explanation=(
            "Two lessons here, and the second is the important one.\n\n"
            "**Parameters change risk, not just return.** Shorter windows react faster, trade more, "
            "and usually deepen drawdowns through whipsaw. You are not tuning a return dial; you are "
            "moving along a return/risk/turnover surface.\n\n"
            "**A spread of outcomes across parameters is a measure of your own uncertainty.** If "
            "Sharpe ranges from -0.2 to 0.9 across reasonable windows, then reporting the 0.9 is "
            "reporting the luckiest draw from a distribution you had no way to sample in advance. "
            "The honest summary is the middle of the distribution, not its maximum.\n\n"
            "This is why quants prefer broad plateaus over sharp peaks. A peak means the result "
            "depends on getting a number exactly right, which you cannot do prospectively. A plateau "
            "means the effect is robust to the choice, which is what a real effect looks like."
        ),
        check=check_parameter_sensitivity,
    ),
    Exercise(
        key="lookahead",
        title="Measure the regime lookahead bias on your own data",
        level="3 · Regimes",
        prompt=(
            "Run the same regime-filtered strategy twice: once with regimes fitted on the full "
            "sample, once with walk-forward regimes. Compare the equity curves and Sharpe ratios. "
            "The difference is lookahead bias, measured on your data."
        ),
        expected=(
            "The full-sample version should look distinctly better. That gap is entirely artificial "
            "— it comes from the regime model knowing what happened next."
        ),
        hints=[
            "Where exactly does the future enter? Think about how a cluster centre is computed.",
            "Check the exposure column: the walk-forward version has no labels at all for its first two years.",
            "Which of the two could you actually have traded in 2013?",
        ],
        explanation=(
            "Fitting a clustering model or an HMM on all of history embeds the whole history in "
            "every label. A day in 2015 labelled 'low volatility regime' was assigned to a cluster "
            "whose centre was computed partly from 2020 and 2022. In 2015 that cluster did not "
            "exist.\n\n"
            "The resulting backtest looks excellent for a boring reason: the strategy knows which "
            "regime it is in with a precision that was never available.\n\n"
            "`detect_regimes_walk_forward()` refits on an expanding window and labels only forward. "
            "Its labels are noisier, they lag transitions, and it produces nothing for the first "
            "two years — because you genuinely had no model then. That is what honest looks like, "
            "and the gap between the two curves is the price of the honesty.\n\n"
            "This same bias appears anywhere a preprocessing step is fitted on the full sample: "
            "scalers, PCA, feature selection, outlier clipping. Every one of them must be fitted on "
            "training data only. It is one of the most common serious errors in submitted quant work."
        ),
        check=check_lookahead_gap,
    ),
    Exercise(
        key="costs",
        title="Turn on transaction costs and re-rank the strategies",
        level="4 · Adaptive",
        prompt=(
            "Compare all strategies at 0bps and at 10bps. Note the turnover of each and how much "
            "Sharpe each one loses. Does the ranking change?"
        ),
        expected=(
            "High-turnover strategies (ML direction, regime switching) lose far more than low-"
            "turnover ones (SMA crossover). The ranking frequently reorders, and the post-cost "
            "ranking is the real one."
        ),
        hints=[
            "Turnover x cost_bps x 2 is roughly the annual drag. Compute it by hand for the ML strategy.",
            "Which strategies looked best at zero cost? Is that still true at 10bps?",
            "What cost level would make your preferred strategy break even?",
        ],
        explanation=(
            "Costs are not a rounding error and they are not applied uniformly. A strategy turning "
            "over 80x a year at 10bps pays roughly 1.6% annually — which, against a 4% expected "
            "return, is 40% of the edge.\n\n"
            "The important part is that costs penalize precisely the strategies that look best "
            "without them. Frictionless backtests systematically favour high-frequency signals, "
            "because more trades means more opportunities to be slightly right. Add costs and many "
            "of those signals invert.\n\n"
            "A useful habit: for any strategy, compute the cost level at which it breaks even. If "
            "the answer is 3bps, the strategy is a theoretical object. If it survives 20bps, it is "
            "worth more work."
        ),
        check=check_cost_sensitivity,
    ),
    Exercise(
        key="vol_targeting",
        title="Does position sizing beat signal engineering?",
        level="4 · Adaptive",
        prompt=(
            "Take the SMA crossover unchanged and apply volatility targeting to it. Compare "
            "return, Sharpe, max drawdown, exposure and turnover against the unsized version."
        ),
        expected=(
            "Usually a materially smaller max drawdown for a similar or better Sharpe — achieved "
            "without touching the entry signal at all."
        ),
        hints=[
            "Why can trailing realized volatility forecast tomorrow's volatility when trailing return can't forecast tomorrow's return?",
            "Look at what the position size was doing in the month before the largest drawdown.",
            "Check exposure: how much of the improvement is just 'held less'?",
        ],
        explanation=(
            "Volatility clusters; direction doesn't. Today's realized volatility is a genuinely "
            "useful forecast of tomorrow's, which is why every clustering model in regime.py leans "
            "so heavily on the volatility features. Direction has no comparable persistence.\n\n"
            "So the honest edge available is in sizing rather than timing. Scaling inversely to "
            "trailing volatility means the position is already small when turbulence arrives, "
            "rather than being cut afterwards at the worst prices.\n\n"
            "For a training platform this is the most valuable single lesson in the adaptive "
            "section, because it inverts where most effort goes. Retail quant work is almost "
            "entirely about entry signals. Professional risk management is substantially about "
            "sizing.\n\n"
            "Do check exposure before celebrating. If targeting cut your average position to 0.4, "
            "some of the drawdown improvement is simply holding less — the interesting question is "
            "whether Sharpe improved, which is the risk-adjusted version of the same comparison."
        ),
        check=check_vol_targeting,
    ),
    Exercise(
        key="benchmark",
        title="Does it actually beat buy-and-hold?",
        level="1 · Foundations",
        prompt=(
            "For your favourite strategy, compare out-of-sample performance against simply holding "
            "the asset over the same period, with costs on. Report both."
        ),
        expected=(
            "Most strategies here will not beat buy-and-hold on a rising index over the last "
            "decade. That is the correct and useful finding."
        ),
        hints=[
            "Compare Sharpe and max drawdown, not just total return.",
            "Consider exposure: a strategy invested 40% of the time and matching buy-and-hold's return has done something impressive.",
            "Would the answer change on a ticker that didn't rise over the period? Try one.",
        ],
        explanation=(
            "The benchmark is the comparison most backtests omit, and omitting it is how strategies "
            "get published that are strictly worse than doing nothing.\n\n"
            "On a long-only strategy trading a rising index, most of the return is simply the index. "
            "The strategy's contribution is whatever it added or subtracted relative to holding — "
            "which is frequently negative once you count costs and the days it sat out of good moves.\n\n"
            "A strategy can still be worth having while underperforming on total return: if it "
            "matches most of the return with half the drawdown, or with 40% average exposure "
            "(leaving capital free for other things), that is a real result. But you have to make "
            "that argument explicitly on risk-adjusted terms, rather than quietly leaving the "
            "benchmark off the chart.\n\n"
            "And check a second asset. Testing only on an index that tripled over your sample is a "
            "form of survivorship bias in the test itself."
        ),
        check=check_beats_benchmark,
    ),
]

EXERCISES_BY_KEY = {exercise.key: exercise for exercise in EXERCISES}
LEVELS = ["1 · Foundations", "2 · Validation", "3 · Regimes", "4 · Adaptive"]


def run_all(df: pd.DataFrame, ticker: str = "?", keys=None) -> pd.DataFrame:
    """
    Runs every automated check (skipping the ones that need a user answer)
    and returns a summary table. Handy from a script or a notebook:

        from data_loader import fetch_ohlcv
        from exercises import run_all
        run_all(fetch_ohlcv("SPY", "2010-01-01", "2025-01-01"))
    """
    rows = []
    for exercise in EXERCISES:
        if keys and exercise.key not in keys:
            continue
        if exercise.check is None or exercise.answer_prompt:
            continue
        try:
            outcome = exercise.check({"df": df, "ticker": ticker})
            rows.append({"exercise": exercise.key, "observed": outcome.passed,
                         "message": outcome.message})
        except Exception as exc:
            rows.append({"exercise": exercise.key, "observed": None,
                         "message": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)
