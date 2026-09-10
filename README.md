# momentum-strategy

![CI](https://github.com/nmhetre-unc/momentum-strategy/actions/workflows/ci.yml/badge.svg?branch=restructure)

A backtesting engine for single-asset and cross-sectional equity strategies,
built to test one question: does any of them beat buy-and-hold? None of them
do by a detectable margin. The findings below are the point of the project.

All results use SPY or the S&P 100 over 2008-2025, 5bps one-way costs,
90% confidence intervals from a stationary block bootstrap, and deflated
Sharpe ratios following Bailey & López de Prado (2014).

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

Full detail, including the regime analysis and the numeric bugs the test
suite surfaced, is in [FINDINGS.md](FINDINGS.md). Generated tables are in
[results/summary.md](results/summary.md).

## What's in the engine

- Backtest and portfolio backtest with explicit transaction costs and a
  shift-by-1 execution convention
- Risk metrics with stationary block bootstrap confidence intervals, deflated
  Sharpe, and Newey-West standard errors
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

Requires Python 3.12+.

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Run the dashboard:

```bash
streamlit run app/app.py
```

Run the CLI:

```bash
python main.py --ticker SPY --strategy momentum --rolling
```

Run the tests:

```bash
pytest --cov=src/qbt
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