# momentum-strategy

[![CI](https://github.com/nmhetre-unc/momentum-strategy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nmhetre-unc/momentum-strategy/actions/workflows/ci.yml)

**[Live demo: Streamlit dashboard](https://momentum-strategy-hbht78g3wpzyfyhjggwhjt.streamlit.app/)**

A backtesting engine for single-asset and cross-sectional equity strategies,
built to test one question: does any of them beat buy-and-hold? None of them
do by a detectable margin. The findings below are the point of the project.

All results use SPY or the S&P 100 over 2008-2025, 5bps one-way costs,
90% confidence intervals from a stationary block bootstrap, and deflated
Sharpe ratios following Bailey & López de Prado (2014).

## At a glance

- 11 timing strategies tested on SPY (4,278 trading days, 2008-2025): none beats
  buy-and-hold at 90% confidence.
- The best of them, `adaptive_ensemble`, has a deflated Sharpe against the
  benchmark of 0.906 at one trial, 0.549 at five, and 0.380 at eleven.
- 12-1 momentum on the S&P 100: long-leg Sharpe 1.071 [0.727, 1.441], short leg
  -0.882; 88.7% of the loss traces to a -0.214 net beta mismatch, not the signal.
- The stationary bootstrap, 62% of the full results run, now runs in C++ and
  gives bit-identical output. The bootstrap call is 19.4x faster and the whole
  pipeline 2.4x faster, against a ceiling of about 2.7x.
  See [Performance](#performance).
- 184 tests at 82.9% line coverage, with CI on every push (building the C++
  extension) and an 82% coverage gate.

## Performance

**What was hot.** Profiling the full results run (`scripts/regenerate_results.py`:
11 strategies, walk-forward refits, 2000-replicate bootstraps) picked out the
stationary bootstrap. Measured on AC power, it takes **62% of the run**
[58-63%] (`benchmarks/results/full_run.json`). Its Python loop made about 4 tiny
NumPy calls per block, with roughly 214 blocks per replicate. The hand-written
HMM's forward and backward loops cost about 5 s, under 10%. So the
bootstrap was the one worth porting.

**Three implementations, one answer.** `qbt.bootstrap` has three
implementations, all producing bit-identical index arrays:
- `python`: the original loop, kept as the reference.
- `numpy`: the same scalar draws in the same order, with each replicate's
  indices built in one vectorized step.
- `cpp`: a pybind11 extension that draws from the caller's own
  `np.random.Generator` through NumPy's C routines in `libnpyrandom`. It holds
  the generator's lock for each replicate and writes into a preallocated
  buffer with no allocation inside the loop.

The default, `impl="auto"`, picks `cpp` when the extension is available.

**How identity is proven** (`tests/test_bootstrap.py`):
- **Pinned outputs:** Sharpe intervals and SHA-256 hashes of the index stream
  were captured from the original inline loops *before* the refactor and are
  compared with `==`.
- **Hypothesis tests** compare every implementation with the reference across
  `n` up to 5000, integer and float `mean_block` in `[1, n]`, and seeds up to
  2^63−1. Both the index arrays and the generator's state afterwards must match.
- **Generator-state checks:** stopping after k replicates must leave the
  generator exactly where the reference leaves it.
- **Mutation testing:** the equivalence tests were checked against deliberately
  broken copies of the NumPy and C++ versions (rotated output, one extra draw,
  swapped draw order). All three were caught. This was done by hand and isn't
  part of the suite.
- **Full regeneration:** `results/summary.md` regenerates identically under every
  implementation.

**NumPy-version guard.** NumPy doesn't promise the same random stream across
releases. So the extension records the NumPy it was built against, and
`qbt.bootstrap` only uses it when the installed NumPy is that same version.
- On a mismatch, or with no extension, `impl="auto"` falls back to `numpy` with
  a `BootstrapFallbackWarning`.
- An explicit `impl="cpp"` raises instead.
- NumPy is pinned exactly in both the build and runtime requirements.
- With no C++ compiler, the package installs as pure Python.

**Results.** Measured on an Intel Core 7 150U (12 logical CPUs) under WSL2
(Linux 6.18) on AC power, with Python 3.12.16, NumPy 2.5.4 and GNU 15.2.0 at
`-O3`. Workload: n=4278, mean_block=20, n_boot=2000. Times are medians; micro rows show the IQR,
and full-run rows show `[min-max]` of 3 runs. Speedups are paired within
each repeat, shown as median `[min-max]`.

| | python | numpy | cpp | cpp vs numpy | cpp vs python |
|---|---|---|---|---|---|
| One replicate (indices + Sharpe) | 680 µs (IQR 35) | 337 µs (IQR 16) | 36.7 µs (IQR 4.3) | 9.0x [6.7-12.3] | 18.5x [13.2-24.0] |
| `stationary_bootstrap_sharpe` call | 1306 ms (IQR 58) | 644 ms (IQR 26) | 68 ms (IQR 4) | 9.5x [8.6-10.2] | 19.4x [17.6-21.1] |
| Bootstrap time inside the full run | 31.8 s [29.1-35.3] | 16.3 s [14.0-17.4] | 2.35 s [2.16-2.69] | 6.5x [6.0-7.5] | 13.1x [12.4-14.7] |
| Bootstrap share of the full run | 62% | 45% | 11% | | |
| **Full run, end to end** | **55.1 s [47.0-55.7]** | **36.6 s [32.0-37.1]** | **20.0 s [19.8-25.5]** | **1.6x [1.5-1.8]** | **2.4x [2.2-2.8]** |

**Amdahl's law.** The other ~20 s of the run (HMM fitting, walk-forward
refits, sklearn) didn't change. So even an instant bootstrap would cap the
end-to-end speedup at about 55/20, roughly 2.7x, and 2.4x is most of
what was available. Within the full run the bootstrap gains 13x rather than
the call's 19x. That's because the script's paired bootstrap computes two
Sharpe ratios per replicate, and that part is still NumPy. The hottest code
now is the HMM's forward and backward loops, at 17% and 9% of the C++ run.

**Methodology.**
- **Power:** everything ran on AC power, which is recorded in every result.
  An earlier batch on battery was thrown out: throttling made one Python run
  beat every NumPy run. It's kept as
  `benchmarks/results/INVALID_on_battery_full_run.json`.
- **Ordering:** every repeat runs each implementation once, with the order
  rotating between repeats, and speedups are paired ratios within a repeat.
  That keeps CPU-speed drift from landing on one implementation.
- **Isolation:** each full run is a fresh subprocess, its output hash is
  checked, and the CPU used by other processes is recorded (0.06-0.23 CPUs
  throughout).
- **Residual noise:** the identical non-bootstrap work still varied 17.5-23.3 s
  across runs, which is why every number carries a spread.

Full details, scripts and raw JSON are in [benchmarks/](benchmarks/README.md).

## Findings

**No timing strategy beats buy-and-hold detectably.** Eleven strategies were
tested on SPY. Every one produces a 90% confidence interval on the Sharpe
difference against buy-and-hold that contains zero. The closest are
`adaptive_ensemble` at +0.324 [-0.045, +0.686] and `ml_direction` at +0.10
[-0.01, +0.20]. Buy-and-hold's own Sharpe over the period is 0.607.

**The choice of benchmark inverts the conclusion.** Each strategy's own
Sharpe interval excludes zero — `sma_crossover` [0.35, 1.17],
`adaptive_ensemble` [0.59, 1.32] — and read alone both look like clear wins.
But 0.607 falls inside both, so neither is distinguishable from holding the
index. The same inversion appears in the deflated Sharpe: against zero both
stay above 0.90 at any trial count, while against the benchmark
`adaptive_ensemble` falls to 0.549 at five trials and `sma_crossover` to
0.236. Reporting only the against-zero number is what a backtest implicitly
does when it cites a Sharpe ratio and calls it significant.

**Volatility targeting is the one mechanism that survives.** It cut maximum
drawdown from -51.9% to -18.7% at a Sharpe of 0.832, holding 71.7% average
exposure. A constant 71.7% position achieves a Sharpe of exactly 0.607 —
identical to buy-and-hold, since Sharpe is invariant under constant rescaling
— and still draws down -39.7%. The improvement comes from when it sizes down,
not how much. This is the only result here whose mechanism does not require
forecasting returns: volatility clusters and is predictable where direction
is not.

**Cross-sectional momentum works long and fails short.** 12-1 momentum on the
S&P 100, decile long-short, monthly rebalance: combined Sharpe 0.001, max
drawdown -77.0%. The long leg alone has a Sharpe of 1.071 [0.727, 1.441], the
only interval in this project that clearly excludes zero. The short leg is
-0.882 [-1.281, -0.483] and exactly cancels it.

Equal dollar weights did not produce equal market exposure. The winner decile
skewed defensive, giving the long leg a beta of +0.789, while shorting the
loser decile gave a normal -1.003 — a net -0.214 against a universe that
returned +1059%. A log-space decomposition attributes 88.7% of the loss to
that beta mismatch and 11.3% to signal. Hedging the beta halves the drawdown
and improves 2009 from -63.6% to -12.4%, but the resulting Sharpe of 0.196
still has a confidence interval containing zero.

**Refitting per fold removes a measurable share of apparent edge.** The
rolling walk-forward previously fit each strategy once on the full series.
Refitting on each fold's own training window lowers mean fold Sharpe by 0.42
to 0.49 for strategies whose behaviour depends on a Sharpe-selected choice,
by about 0.20 for fitted classifiers, and not at all for strategies with no
fit step.

Full detail, including the regime analysis, is in [FINDINGS.md](FINDINGS.md).
Generated tables are in [results/summary.md](results/summary.md).

## What property-based testing found

Two numeric bugs in the metrics code surfaced through the Hypothesis property
tests in `tests/test_properties.py`, and both are fixed in `src/qbt/analytics.py`:

- **A floating-point guard that missed zero.** A constant return series should
  have zero standard deviation, but mean-then-subtract arithmetic leaves about
  1e-17, so an exact `== 0` check let the Sharpe and Sortino ratios blow up.
  The guard is now `np.isclose(std, 0.0, atol=1e-12)`.
- **A float64 overflow in annualization.** Compounding a short or volatile
  window to a year (for example a 300x move over two days) overflowed
  `(1 + r) ** 252` to infinity. Annualized return is now computed in log space
  with the exponent clipped to +/-700, in `cagr()` and in the per-regime
  annualization.

## What's in the engine

- Backtest and portfolio backtest with explicit transaction costs and a
  shift-by-1 execution convention
- Risk metrics with stationary block bootstrap confidence intervals, deflated
  Sharpe, and Newey-West standard errors; the bootstrap has bit-identical
  Python, NumPy and C++ (pybind11) implementations
- Regime detection via k-means, Gaussian mixture, a hand-written Gaussian HMM
  with Baum-Welch, and rule-based thresholds, with causal fitting by default
- Walk-forward validation with per-fold refitting
- Cross-sectional momentum on an S&P 100 price panel
- A Streamlit dashboard over all of it

## Limitations

Single asset for the timing strategies, single 17-year window, long/flat
positions only. Costs are a flat 5bps one-way spread proxy with no market
impact, borrow, or financing.

The S&P 100 universe is current membership, so companies that left the index
between 2008 and 2025 are absent. Those are disproportionately the sustained
underperformers a short leg would have profited from, so the bias runs in the
strategy's favor and the negative result is if anything understated.

Headline metrics for fitted strategies come from full-period backtests in
which the model predicts across its own training period — 70% of the equity
curve for `ml_direction` and the adaptive wrappers. The walk-forward results
in `results/summary.md` refit each fold and are the honest read for those
strategies. Since every paired difference against buy-and-hold already
contains zero, correcting this would strengthen rather than change the
conclusion.

With 17 years of daily data the standard error on a Sharpe estimate is
roughly 0.26. Establishing a 0.32 improvement at 90% confidence would require
about 45 years — more history than SPY has, and long enough that market
structure would have changed repeatedly. The question is close to
unanswerable at this frequency and sample size.

## Setup

Requires Python 3.12+. A C++17 compiler (GCC, Clang, or MSVC Build Tools on
Windows) is needed for the fast bootstrap. CMake, Ninja and pybind11 are
fetched automatically during the install. Without a compiler, the install still
succeeds as pure Python, and the bootstrap falls back to the NumPy
implementation with a warning.

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -e ".[dev]"         # builds the C++ extension
python -c "import qbt.bootstrap as b; print('C++ bootstrap:', b.cpp_available())"
```

After changing `src/cpp/` or upgrading NumPy, re-run `pip install -e .` to
rebuild the extension. NumPy is pinned exactly: the extension is only used
with the NumPy version it was built against.

Run the dashboard:

```bash
streamlit run app/app.py
```

Run the CLI:

```bash
python main.py --ticker SPY --strategy momentum --rolling
```

Run the tests (184 tests, 82.9% line coverage):

```bash
pytest --cov=src/qbt
```

Run the benchmarks (not part of the test run; see
[benchmarks/README.md](benchmarks/README.md)):

```bash
python benchmarks/micro_interleaved.py
python benchmarks/full_run.py --runs 3
```

## References

Bailey DH, López de Prado M. 2014. The deflated Sharpe ratio. Journal of
Portfolio Management. 40:94-107.

Daniel K, Moskowitz TJ. 2016. Momentum crashes. Journal of Financial
Economics. 122:221-247.

Jegadeesh N, Titman S. 1993. Returns to buying winners and selling losers.
Journal of Finance. 48:65-91.

Politis DN, Romano JP. 1994. The stationary bootstrap. Journal of the
American Statistical Association. 89:1303-1313.