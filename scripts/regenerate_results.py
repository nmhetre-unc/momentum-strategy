"""
Full strategy comparison table for SPY: every strategy in STRATEGIES and
ADAPTIVE_STRATEGIES, plus buy-and-hold as the first row and the benchmark
every "vs benchmark" column below is measured against.

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
from qbt.data import fetch_ohlcv
from qbt.stats import EULER_MASCHERONI, deflated_sharpe_ratio, stationary_bootstrap_sharpe
from qbt.strategies import STRATEGIES

TICKER = "SPY"
START, END = "2008-01-01", "2025-01-01"
COST_BPS = 5.0
N_BOOT = 2000
N_TRIALS = 5
MEAN_BLOCK = 20
MEAN_BLOCK_SWEEP = (10, 20, 40)
SENSITIVITY_STRATEGIES = ("sma_crossover", "adaptive_ensemble")
PASSIVE_EQUIVALENT_STRATEGY = "volatility_targeted"

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
                                mean_block=MEAN_BLOCK, seed=0):
    """
    Draws one shared set of geometric-block, wraparound-resampled indices
    per replicate (Politis-Romano stationary bootstrap) and applies it to
    both series, so the strategy's own Sharpe CI and its paired
    difference against the benchmark come from identical resamples
    rather than two independently-resampled distributions. Returns
    (strat_lo, strat_hi, diff_point, diff_lo, diff_hi).
    """
    strat_values = np.asarray(strat_returns)
    bench_values = np.asarray(bench_returns)
    n = len(strat_values)
    assert len(bench_values) == n, "series must be paired/aligned"
    p = 1.0 / mean_block
    rng = np.random.default_rng(seed)

    diff_point = sharpe_ratio(strat_returns) - sharpe_ratio(bench_returns)
    strat_sharpes = np.empty(n_boot)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pieces = []
        total = 0
        while total < n:
            start = rng.integers(0, n)
            length = rng.geometric(p)
            pieces.append((start + np.arange(length)) % n)
            total += length
        idx = np.concatenate(pieces)[:n]
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
    if key in ("gross_sharpe", "net_sharpe", "skew"):
        return f"{value:.2f}"
    if key in ("dsr_zero", "dsr_bench"):
        return f"{value:.3f}"
    if key in ("annualized_return", "annualized_volatility", "net_return", "max_drawdown"):
        return f"{value:.1%}"
    if key == "turnover":
        return f"{value:.1f}"
    if key == "kurtosis":
        return f"{value:.1f}"
    if key == "num_trades":
        return f"{value:d}"
    raise KeyError(key)


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
        "at any single block length.",
        "",
        "| Strategy | " + " | ".join(f"mean_block={mb}" for mb in MEAN_BLOCK_SWEEP) + " |",
        "|---|" + "|".join(["---"] * len(MEAN_BLOCK_SWEEP)) + "|",
    ]
    all_strategies = {**STRATEGIES, **ADAPTIVE_STRATEGIES}
    for name in SENSITIVITY_STRATEGIES:
        signal = all_strategies[name](df)
        result = run_backtest(df, signal, cost_bps=COST_BPS)
        returns = result["strategy_return"].dropna()
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
    all_strategies = {**STRATEGIES, **ADAPTIVE_STRATEGIES}
    signal = all_strategies[name](df)
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


def main():
    df = fetch_ohlcv(TICKER, START, END)

    bench_result = run_backtest(df, pd.Series(1.0, index=df.index), cost_bps=0)
    bench_returns = bench_result["strategy_return"].dropna()
    bench_gross_sharpe = full_report(bench_result, n_boot=0)["sharpe_ratio"]

    rows = [build_row("buy_and_hold", bench_result, bench_returns, bench_returns,
                      gross_sharpe=bench_gross_sharpe, is_benchmark=True)]

    for name, fn in {**STRATEGIES, **ADAPTIVE_STRATEGIES}.items():
        signal = fn(df)
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
    table = render_markdown_table(rows)
    sensitivity = render_mean_block_sensitivity(df, bench_returns)
    passive_equivalent = render_passive_equivalent(df)

    document = f"{header}\n{table}\n\n{sensitivity}\n\n{passive_equivalent}\n"

    print(document)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(document, encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
