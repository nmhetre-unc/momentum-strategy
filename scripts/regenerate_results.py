"""
Strategy comparison tables for SPY: every strategy in STRATEGIES and
ADAPTIVE_STRATEGIES, against buy-and-hold, the benchmark every "vs
benchmark" column below is measured against.

No strategy is scored on data it learned from. A strategy that learns
anything (a fitted model, a regime model, a choice selected on past
performance) marks its first out-of-sample day on its signal
(qbt.timesplit.out_of_sample_start) and is scored only from that day,
against buy-and-hold over the identical window, in its own table. The
full-period table holds only the strategies that learn nothing. Both
tables use the same n_trials (N_TRIALS) for the deflated Sharpe,
whichever table a strategy lands in; results/ml/
learning_strategies_audit.json lists what each strategy learns.

Uses the 17-year cached SPY file (2008-01-01 to 2025-01-01) rather than
the 10-year one: several adaptive wrappers only learn their rules from
the first 60-70% of history, and the longer window leaves them more to
learn from and a longer out-of-sample tail to be judged on.

Two Sharpe significance questions are reported side by side, and they
are not the same question: DSR vs zero asks whether the strategy beats
doing nothing; DSR vs benchmark asks whether it beats simply holding the
asset, which is the bar that actually matters for a strategy that costs
more effort to run than one click. See the "Diff vs buy_and_hold" and
"DSR vs bench" columns for the second, harder question.

The paired difference against buy-and-hold resamples both return series
with the SAME block indices per replicate (drawn once, applied to both),
so the comparison holds day-to-day pairing fixed rather than comparing
two independently-resampled distributions -- the same construction used
in the standalone paired-bootstrap analysis this table formalizes.

Run with: python scripts/regenerate_results.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

from qbt.adaptive import ADAPTIVE_STRATEGIES
from qbt.analytics import TRADING_DAYS_PER_YEAR, full_report, sharpe_ratio
from qbt.backtest import run_backtest
from qbt.bootstrap import stationary_bootstrap_indices
from qbt.data import fetch_ohlcv
from qbt.stats import EULER_MASCHERONI, deflated_sharpe_ratio, stationary_bootstrap_sharpe
from qbt.strategies import STRATEGIES
from qbt.timesplit import out_of_sample_start
from qbt.walk_forward import rolling_walk_forward

TICKER = "SPY"
START, END = "2008-01-01", "2025-01-01"
COST_BPS = 5.0
N_BOOT = 2000
N_TRIALS = 5
MEAN_BLOCK = 20
MEAN_BLOCK_SWEEP = (10, 20, 40)
SENSITIVITY_STRATEGIES = ("sma_crossover", "adaptive_ensemble")
PASSIVE_EQUIVALENT_STRATEGY = "volatility_targeted"
ROLLING_TRAIN_DAYS = 756
ROLLING_TEST_DAYS = 126

TRIAL_SWEEP = (1, 5, 11)  # 11: every strategy in both tables
TRIAL_SWEEP_STRATEGIES = ("sma_crossover", "adaptive_ensemble")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_PATH = RESULTS_DIR / "summary.md"

COLUMNS = [
    ("strategy", "Strategy"),
    ("gross_sharpe", "Gross Sharpe"),
    ("net_sharpe", "Net Sharpe"),
    ("net_sharpe_ci", "Net Sharpe 90% CI"),
    ("diff_vs_bh", "Diff vs buy_and_hold (90% CI)"),
    ("dsr_zero", "DSR vs 0"),
    ("dsr_bench", "DSR vs bench"),
    ("annualized_return", "Ann. Return"),
    ("annualized_volatility", "Ann. Vol"),
    ("net_return", "Net return"),
    ("max_drawdown", "Max DD"),
    ("turnover", "Turnover"),
    ("num_trades", "Trades"),
    ("skew", "Skew"),
    ("kurtosis", "Kurtosis"),
]


def per_period(annualized_sharpe: float) -> float:
    return annualized_sharpe / np.sqrt(TRADING_DAYS_PER_YEAR)


def dsr_vs_benchmark(sharpe: float, sr_benchmark: float, n_trials: int,
                     skew_: float, kurt_: float, n_obs: int) -> float:
    """
    Same construction as qbt.stats.deflated_sharpe_ratio, but the
    threshold is sr_benchmark + sr0 instead of just sr0 -- P(true Sharpe
    > sr_benchmark) after the same selection-bias and non-normality
    correction. `sharpe` and `sr_benchmark` must both be per-period
    (non-annualized), matching `n_obs`.
    """
    sr_std = np.sqrt((1 - skew_ * sharpe + (kurt_ - 1) / 4 * sharpe ** 2) / (n_obs - 1))
    if n_trials <= 1:
        sr0 = 0.0
    else:
        sr0 = sr_std * (
            (1 - EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n_trials)
            + EULER_MASCHERONI * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    return float(norm.cdf((sharpe - sr_benchmark - sr0) / sr_std))


def paired_stationary_bootstrap(strat_returns, bench_returns, n_boot=N_BOOT,
                                mean_block=MEAN_BLOCK, seed=0, impl="auto"):
    """
    Draws one shared set of geometric-block, wraparound-resampled indices
    per replicate (Politis-Romano stationary bootstrap) and applies it to
    both series, so the strategy's own Sharpe CI and its paired
    difference against the benchmark come from identical resamples
    rather than two independently-resampled distributions. Returns
    (strat_lo, strat_hi, diff_point, diff_lo, diff_hi). `impl` selects the
    qbt.bootstrap index-generator implementation; all are bit-identical.
    """
    strat_values = np.asarray(strat_returns)
    bench_values = np.asarray(bench_returns)
    n = len(strat_values)
    assert len(bench_values) == n, "series must be paired/aligned"
    indices = stationary_bootstrap_indices(
        n, n_boot, mean_block, np.random.default_rng(seed), impl=impl
    )

    diff_point = sharpe_ratio(strat_returns) - sharpe_ratio(bench_returns)
    strat_sharpes = np.empty(n_boot)
    diffs = np.empty(n_boot)
    for i, idx in enumerate(indices):
        s_sharpe = sharpe_ratio(strat_values[idx])
        b_sharpe = sharpe_ratio(bench_values[idx])
        strat_sharpes[i] = s_sharpe
        diffs[i] = s_sharpe - b_sharpe

    strat_lo, strat_hi = np.percentile(strat_sharpes, [5, 95])
    diff_lo, diff_hi = np.percentile(diffs, [5, 95])
    return strat_lo, strat_hi, diff_point, diff_lo, diff_hi


def build_row(name, result, strat_returns, bench_returns, gross_sharpe, is_benchmark):
    report = full_report(result, n_boot=0)
    net_sharpe = report["sharpe_ratio"]
    n_obs = len(strat_returns)
    strat_skew = float(_skew(strat_returns))
    strat_kurtosis = float(_kurtosis(strat_returns, fisher=False))

    dsr_zero = deflated_sharpe_ratio(
        per_period(net_sharpe), N_TRIALS, strat_skew, strat_kurtosis, n_obs
    )

    if is_benchmark:
        # buy_and_hold IS the benchmark here, so "beats the benchmark" and
        # "beats the benchmark after correcting for having tried n_trials
        # candidates" are both undefined for this row -- there was no
        # candidate selection to correct for. Compute the Sharpe CI (that
        # question is well-posed), but leave the vs-benchmark columns as
        # None rather than the specious sub-0.5 number the formula would
        # otherwise produce from subtracting a nonzero selection-bias term
        # from a strategy being compared to itself.
        _, net_lo, net_hi = stationary_bootstrap_sharpe(
            strat_returns, n_boot=N_BOOT, mean_block=MEAN_BLOCK, seed=0,
        )
        diff_vs_bh = None
        dsr_bench = None
    else:
        net_lo, net_hi, diff_point, diff_lo, diff_hi = paired_stationary_bootstrap(
            strat_returns, bench_returns, n_boot=N_BOOT, mean_block=MEAN_BLOCK, seed=0,
        )
        diff_vs_bh = (diff_point, diff_lo, diff_hi)
        sr_benchmark_pp = per_period(sharpe_ratio(bench_returns))
        dsr_bench = dsr_vs_benchmark(
            per_period(net_sharpe), sr_benchmark_pp, N_TRIALS, strat_skew, strat_kurtosis, n_obs
        )

    return {
        "strategy": name,
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "net_sharpe_ci": (net_lo, net_hi),
        "diff_vs_bh": diff_vs_bh,
        "dsr_zero": dsr_zero,
        "dsr_bench": dsr_bench,
        "annualized_return": report["cagr"],
        "annualized_volatility": report["annualized_volatility"],
        "net_return": report["total_return"],
        "max_drawdown": report["max_drawdown"],
        "turnover": report["turnover"],
        "num_trades": report["num_trades"],
        "skew": strat_skew,
        "kurtosis": strat_kurtosis,
    }


def format_cell(key, value):
    if value is None:
        return "—"
    if key == "net_sharpe_ci":
        lo, hi = value
        return f"[{lo:.2f}, {hi:.2f}]"
    if key == "diff_vs_bh":
        point, lo, hi = value
        return f"{point:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    if key == "strategy":
        return value
    if key == "bench_sharpe_ci":
        lo, hi = value
        return f"[{lo:.2f}, {hi:.2f}]"
    if key in ("oos_from",):
        return value
    if key == "days":
        return f"{value:d}"
    if key in ("gross_sharpe", "net_sharpe", "skew", "bench_sharpe"):
        return f"{value:.2f}"
    if key in ("dsr_zero", "dsr_bench"):
        return f"{value:.3f}"
    if key in ("annualized_return", "annualized_volatility", "net_return", "max_drawdown",
               "bench_annualized_return", "bench_max_drawdown", "exposure"):
        return f"{value:.1%}"
    if key == "turnover":
        return f"{value:.1f}"
    if key == "kurtosis":
        return f"{value:.1f}"
    if key == "num_trades":
        return f"{value:d}"
    raise KeyError(key)


OOS_COLUMNS = [
    ("strategy", "Strategy"),
    ("oos_from", "Out of sample from"),
    ("days", "Days"),
    ("net_sharpe", "Net Sharpe"),
    ("net_sharpe_ci", "Net Sharpe 90% CI"),
    ("bench_sharpe", "buy_and_hold Sharpe, same window"),
    ("bench_sharpe_ci", "buy_and_hold 90% CI"),
    ("diff_vs_bh", "Diff vs buy_and_hold (90% CI)"),
    ("dsr_zero", "DSR vs 0"),
    ("dsr_bench", "DSR vs bench"),
    ("annualized_return", "Ann. Return"),
    ("bench_annualized_return", "buy_and_hold Ann. Return"),
    ("max_drawdown", "Max DD"),
    ("bench_max_drawdown", "buy_and_hold Max DD"),
    ("exposure", "Exposure"),
    ("num_trades", "Trades"),
]


def all_strategies():
    return {**STRATEGIES, **ADAPTIVE_STRATEGIES}


def out_of_sample_window(df, fn):
    """(first out-of-sample day or None, net result, net returns) for one
    strategy, restricted to its out-of-sample window when it has one."""
    signal = fn(df)
    start = out_of_sample_start(signal)
    net_result = run_backtest(df, signal, cost_bps=COST_BPS)
    if start is not None:
        net_result = net_result.loc[start:]
    return start, net_result, net_result["strategy_return"].dropna()


def build_oos_row(name, start, net_result, net_returns, bench_full):
    """A learning strategy against buy-and-hold over the identical window."""
    bench_window = bench_full.loc[start:]
    bench_returns = bench_window["strategy_return"].dropna()
    row = build_row(name, net_result, net_returns, bench_returns, gross_sharpe=None,
                    is_benchmark=False)
    bench_report = full_report(bench_window, n_boot=0)
    _, bench_lo, bench_hi = stationary_bootstrap_sharpe(
        bench_returns, n_boot=N_BOOT, mean_block=MEAN_BLOCK, seed=0,
    )
    return {
        **row,
        "oos_from": start.date().isoformat(),
        "days": len(net_returns),
        "bench_sharpe": bench_report["sharpe_ratio"],
        "bench_sharpe_ci": (bench_lo, bench_hi),
        "bench_annualized_return": bench_report["cagr"],
        "bench_max_drawdown": bench_report["max_drawdown"],
        "exposure": full_report(net_result, n_boot=0)["exposure"],
    }


def render_oos_table(rows):
    headers = [label for _, label in OOS_COLUMNS]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [format_cell(key, row[key]) for key, _ in OOS_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_trial_sweep(df, bench_full):
    """
    Deflated Sharpe at several trial counts, from each strategy's scored
    window: the full period for a strategy that learns nothing, its
    out-of-sample window otherwise (against buy-and-hold over the same
    window).
    """
    lines = [
        "## Deflated Sharpe by trial count",
        "",
        f"n_trials in {TRIAL_SWEEP}. 11 is every strategy in both tables above. Each "
        "strategy is scored on the window it appears in above.",
        "",
        "| Strategy | Window | "
        + " | ".join(f"DSR vs 0 @{n}" for n in TRIAL_SWEEP) + " | "
        + " | ".join(f"DSR vs bench @{n}" for n in TRIAL_SWEEP) + " |",
        "|---|---|" + "|".join(["---"] * (2 * len(TRIAL_SWEEP))) + "|",
    ]
    strategies = all_strategies()
    for name in TRIAL_SWEEP_STRATEGIES:
        start, net_result, returns = out_of_sample_window(df, strategies[name])
        bench_window = bench_full if start is None else bench_full.loc[start:]
        bench_returns = bench_window["strategy_return"].dropna()
        sharpe_pp = per_period(full_report(net_result, n_boot=0)["sharpe_ratio"])
        skew_, kurt_ = float(_skew(returns)), float(_kurtosis(returns, fisher=False))
        zero = [deflated_sharpe_ratio(sharpe_pp, n, skew_, kurt_, len(returns))
                for n in TRIAL_SWEEP]
        bench = [dsr_vs_benchmark(sharpe_pp, per_period(sharpe_ratio(bench_returns)), n,
                                  skew_, kurt_, len(returns)) for n in TRIAL_SWEEP]
        window = ("full period" if start is None
                  else f"{start.date()} to {returns.index[-1].date()}")
        lines.append(f"| {name} | {window} | " + " | ".join(f"{v:.3f}" for v in zero + bench)
                     + " |")
    return "\n".join(lines)


def render_markdown_table(rows):
    headers = [label for _, label in COLUMNS]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [format_cell(key, row[key]) for key, _ in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_mean_block_sensitivity(df, bench_returns):
    lines = [
        "## Mean block length sensitivity",
        "",
        f"Net Sharpe 90% CI at n_boot={N_BOOT}, for mean_block in {MEAN_BLOCK_SWEEP}. "
        "A CI that moves a lot with mean_block means the width itself isn't trustworthy "
        "at any single block length. Each strategy on the window it is scored on above.",
        "",
        "| Strategy | " + " | ".join(f"mean_block={mb}" for mb in MEAN_BLOCK_SWEEP) + " |",
        "|---|" + "|".join(["---"] * len(MEAN_BLOCK_SWEEP)) + "|",
    ]
    strategies = all_strategies()
    for name in SENSITIVITY_STRATEGIES:
        _start, _result, returns = out_of_sample_window(df, strategies[name])
        cells = []
        for mean_block in MEAN_BLOCK_SWEEP:
            _, lo, hi = stationary_bootstrap_sharpe(
                returns, n_boot=N_BOOT, mean_block=mean_block, seed=0
            )
            cells.append(f"[{lo:.2f}, {hi:.2f}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_passive_equivalent(df):
    """
    Isolates dynamic sizing from simply holding less: a constant position
    at the strategy's own average exposure has, by construction, the
    same Sharpe as buy-and-hold (Sharpe is scale-invariant under a fixed
    weight), so any Sharpe or drawdown improvement over that passive
    baseline is attributable to the timing of the sizing, not its average
    level.
    """
    name = PASSIVE_EQUIVALENT_STRATEGY
    signal = all_strategies()[name](df)
    assert out_of_sample_start(signal) is None, f"{name} learns; compare it out of sample"
    result = run_backtest(df, signal, cost_bps=COST_BPS)
    report = full_report(result, n_boot=0)
    avg_exposure = report["exposure"]

    passive_signal = pd.Series(avg_exposure, index=df.index)
    passive_result = run_backtest(df, passive_signal, cost_bps=COST_BPS)
    passive_report = full_report(passive_result, n_boot=0)

    lines = [
        f"## Passive-equivalent comparison: {name}",
        "",
        f"{name}'s average exposure is {avg_exposure:.1%}. Comparing it against a "
        "constant position held at that same exposure the whole period isolates "
        "the value of dynamic sizing from the trivial effect of just holding less.",
        "",
        "| | Sharpe | Ann. Vol | Max DD |",
        "|---|---|---|---|",
        f"| {name} (actual) | {report['sharpe_ratio']:.3f} | "
        f"{report['annualized_volatility']:.1%} | {report['max_drawdown']:.1%} |",
        f"| Passive @ {avg_exposure:.1%} exposure | {passive_report['sharpe_ratio']:.3f} | "
        f"{passive_report['annualized_volatility']:.1%} | {passive_report['max_drawdown']:.1%} |",
    ]
    return "\n".join(lines)


def render_rolling_comparison(df):
    """
    Mean out-of-sample FOLD Sharpe under genuine per-fold refitting versus
    the old fixed-model approach (one fit on the whole series, reused
    across every fold) -- a separate question from the tables above,
    which come from single backtests and never touch
    rolling_walk_forward(). A fold is averaged only if the strategy can
    hold a position on every day of its test window: a learning strategy
    fitted once on the whole series is flat until its first
    out-of-sample day, so the fixed-model pass scores only the folds
    after it. The "same folds" column restricts refit-per-fold to those
    folds too, so the change compares like with like.
    """
    lines = [
        "## Rolling walk-forward: refit-per-fold vs fixed-model",
        "",
        f"train_days={ROLLING_TRAIN_DAYS}, test_days={ROLLING_TEST_DAYS}, "
        f"cost_bps={COST_BPS:.0f}. Mean Sharpe across out-of-sample folds -- not the "
        "Sharpe in the tables above, and not affected by them either way. "
        "'Fixed-model' reuses one fit across every fold (the old default); 'refit "
        "per fold' refits fresh on each fold's own training window, which is honest "
        "but n_folds times more expensive. Only folds where the strategy can hold a "
        "position on every test day are averaged; the fold columns count them. "
        "'Change' is refit minus fixed-model over the same folds.",
        "",
        "| Strategy | Fixed-model folds | Fixed-model mean fold Sharpe | "
        "Refit-per-fold, same folds | Change | Refit-per-fold folds | "
        "Refit-per-fold mean fold Sharpe |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = []
    for name, fn in all_strategies().items():
        result = rolling_walk_forward(
            df, fn, train_days=ROLLING_TRAIN_DAYS, test_days=ROLLING_TEST_DAYS,
            cost_bps=COST_BPS, refit_per_fold=True,
        )
        refit_folds, fixed_folds = result["folds"], result["fixed_model_folds"]
        same = fixed_folds.loc[fixed_folds["can_hold_position"], "fold"]
        refit_same = float(refit_folds.loc[refit_folds["fold"].isin(same) &
                                           refit_folds["can_hold_position"],
                                           "sharpe_ratio"].mean())
        rows.append({
            "name": name,
            "fixed_n": len(same), "n": len(fixed_folds),
            "fixed": result["mean_sharpe_fixed_model"],
            "refit_same": refit_same,
            "refit_n": result["n_folds_scored"],
            "refit": result["mean_sharpe"],
        })

    # Most overstated by the fixed-model approach (largest drop) first.
    rows.sort(key=lambda r: r["refit_same"] - r["fixed"])
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['fixed_n']} of {r['n']} | {r['fixed']:.2f} | "
            f"{r['refit_same']:.2f} | {r['refit_same'] - r['fixed']:+.2f} | "
            f"{r['refit_n']} of {r['n']} | {r['refit']:.2f} |"
        )

    return "\n".join(lines), rows


def main():
    df = fetch_ohlcv(TICKER, START, END)

    bench_result = run_backtest(df, pd.Series(1.0, index=df.index), cost_bps=0)
    bench_returns = bench_result["strategy_return"].dropna()
    bench_gross_sharpe = full_report(bench_result, n_boot=0)["sharpe_ratio"]

    rows = [build_row("buy_and_hold", bench_result, bench_returns, bench_returns,
                      gross_sharpe=bench_gross_sharpe, is_benchmark=True)]
    oos_rows = []

    for name, fn in all_strategies().items():
        signal = fn(df)
        start = out_of_sample_start(signal)
        if start is not None:
            _start, net_result, net_returns = out_of_sample_window(df, fn)
            oos_rows.append(build_oos_row(name, start, net_result, net_returns, bench_result))
            continue
        gross_sharpe = full_report(run_backtest(df, signal, cost_bps=0), n_boot=0)["sharpe_ratio"]
        net_result = run_backtest(df, signal, cost_bps=COST_BPS)
        net_returns = net_result["strategy_return"].dropna()
        rows.append(build_row(name, net_result, net_returns, bench_returns,
                              gross_sharpe=gross_sharpe, is_benchmark=False))

    header = (
        f"# {TICKER} strategy comparison\n\n"
        f"Ticker: {TICKER} · Window: {START} to {END} · "
        f"Cost: {COST_BPS:.0f}bps one-way (buy_and_hold at 0bps) · "
        f"n_boot={N_BOOT} · n_trials={N_TRIALS} · mean_block={MEAN_BLOCK}\n"
    )
    full_heading = (
        "## Full period: strategies that learn nothing\n\n"
        "Rules with fixed parameters, scored on every day.\n"
    )
    oos_heading = (
        "## Out of sample only: strategies that learn\n\n"
        "Each strategy scored only from its first out-of-sample day, the day after "
        "everything it learned from (a fitted model, a regime model's fit window, a "
        "choice learned on past performance; see "
        "results/ml/learning_strategies_audit.json). buy_and_hold is measured over the "
        "identical window, and the difference and DSR vs bench are against it. Same "
        f"n_trials={N_TRIALS}. The first day's return is zero by construction (a position "
        "decided that day earns from the next).\n"
    )
    table = render_markdown_table(rows)
    oos_table = render_oos_table(oos_rows)
    trial_sweep = render_trial_sweep(df, bench_result)
    sensitivity = render_mean_block_sensitivity(df, bench_returns)
    passive_equivalent = render_passive_equivalent(df)
    rolling_comparison, rolling_rows = render_rolling_comparison(df)

    document = (
        f"{header}\n{full_heading}\n{table}\n\n{oos_heading}\n{oos_table}\n\n"
        f"{trial_sweep}\n\n{sensitivity}\n\n{passive_equivalent}\n\n{rolling_comparison}\n"
    )

    print(document)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(document, encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")

    print("\nMost overstated by the fixed-model approach (largest drop first):")
    for r in rolling_rows[:3]:
        print(f"  {r['name']}: {r['fixed']:.2f} -> {r['refit_same']:.2f} "
              f"({r['refit_same'] - r['fixed']:+.2f}) over {r['fixed_n']} folds")


if __name__ == "__main__":
    main()
