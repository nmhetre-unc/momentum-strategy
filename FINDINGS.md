# Findings

SPY, 2008-01-01 to 2025-01-01, 4,278 trading days. Costs 5bps one-way on
position change unless stated. Bootstrap confidence intervals are 90%,
stationary block bootstrap, n_boot=2000, mean block 20. Deflated Sharpe
follows Bailey & López de Prado (2014).

## 1. No strategy beats buy-and-hold by a detectable margin

Eleven strategies were tested. Every one produces a 90% confidence interval on
the Sharpe difference versus buy-and-hold that contains zero. The two closest
are `adaptive_ensemble` at +0.324 [-0.045, +0.686] and `ml_direction` at +0.10
[-0.01, +0.20]. Buy-and-hold's own Sharpe over the period is 0.607.

`ml_direction` has by far the narrowest difference interval — width 0.21
against `adaptive_ensemble`'s 0.74 — because it is invested most of the time
and moves closely with the benchmark, so the paired difference has low
variance. Small effect, precisely measured, against a large effect measured
imprecisely.

`mean_reversion` is the only strategy that plausibly underperforms: -0.14
difference, deflated Sharpe against the benchmark of 0.038. Its kurtosis of
34.8 and -47.7% drawdown match the short-volatility payoff profile its own
documentation predicts — many small wins, occasional large loss.

## 2. Comparing against zero instead of the benchmark inverts the conclusion

Each strategy's own Sharpe confidence interval excludes zero: `sma_crossover`
[0.35, 1.17], `adaptive_ensemble` [0.59, 1.32]. Read alone, both look like
clear wins. But 0.607 — buy-and-hold's Sharpe — falls inside both intervals,
so neither is distinguishable from simply holding the index.

The same inversion appears in deflated Sharpe. Against zero, both strategies
stay above 0.90 at any trial count. Against the benchmark, `adaptive_ensemble`
falls to 0.549 at 5 trials and 0.380 at 11; `sma_crossover` falls to 0.236 and
0.126. Citing only the against-zero number is what a backtest implicitly does
when it reports a Sharpe ratio and calls it significant.

The benchmark's own against-benchmark cell is suppressed rather than computed.
The deflated Sharpe numerator is `sharpe − sr_benchmark − sr0`, so for the
benchmark row it collapses to `−sr0`, mechanically falling below 0.5 for any
trial count above one. Applying a multiple-testing penalty to the fixed
reference point is a category error.

## 3. Volatility targeting is the one mechanism that survives

`volatility_targeted` cut maximum drawdown from -51.9% to -18.7% at a Sharpe of
0.832 and annualized volatility of 11.0%. It averaged 71.7% exposure, so the
obvious objection is that it simply held less.

A constant 71.7% position over the same period, same costs, achieves a Sharpe
of exactly 0.607 — identical to buy-and-hold, because Sharpe is invariant under
constant rescaling: scaling position by k scales mean and volatility equally.
That 0.607 is the floor any exposure-reduction story must clear. Vol targeting's
0.832 clears it.

The drawdown and volatility comparisons are starker. The matched passive
position realizes 14.3% annualized volatility and still draws down -39.7%,
because a constant reduction leaves you at 72% of full size precisely when
volatility spikes. Vol targeting reaches 11.0% and -18.7% at the same average
exposure. The improvement comes from *when* it sizes down, not how much.

This is the only result here with a mechanism that does not require forecasting
returns. Vol targeting forecasts volatility, which clusters and is genuinely
predictable, and holds risk roughly constant rather than holding capital
constant. Its Sharpe difference versus the benchmark still does not clear
significance (+0.22, [-0.08, +0.54]); the drawdown result does.

It trades 922 times with turnover 2.1. At 5bps on SPY the cost is negligible,
but that is the figure that would grow on a wider spread or a less liquid
instrument.

## 4. Regime filtering never binds

Across four base strategies on this data, the learned allow-list is always
[0, 1, 2] — no regime is ever excluded. The reason is structural: a long/flat
overlay on an asset that rose in every regime has no losing regime to exclude.
SPY was up even in the turbulent regime, which contains the 2009 and 2020
recoveries alongside the crashes.

Where `regime_filtered` differs from its base strategy, the difference is
entirely the UNKNOWN warm-up override — 97 days for `momentum`, 229 for
`mean_reversion`, 266 for `ml_direction`, and 0 for `sma_crossover`, whose own
200-day warm-up already covers the regime detector's. That confounds any
comparison with a start-date effect rather than a regime effect.

## 5. The regime allow-list rests on an undetectable sign

The turbulent regime's learning-window Sharpe for `sma_crossover` is 0.33 over
380 days. The standard error on a Sharpe from 1.5 years of data is roughly
0.87, so the confidence interval spans approximately [-1.4, +2.1]. The filter's
include/exclude decision turns on the sign of a number that cannot be
distinguished from zero. A different HMM seed or a six-month shift in the
learning window could flip it, producing a materially different strategy from
identical code and data.

## 6. The data cannot answer the question

With 17 years of daily returns, the standard error on a Sharpe estimate is
roughly 0.26. Establishing a 0.32 Sharpe improvement at 90% confidence would
require the effect to sit about 1.65 standard errors from zero, which implies
roughly 45 years of daily data — more history than SPY has, and long enough
that market structure would have changed repeatedly.

The strategies are not obviously bad. The question is close to unanswerable at
this frequency and sample size, which is why professional work concentrates on
higher-frequency or cross-sectional problems where the signal-to-noise ratio is
tractable.

## 7. Fat tails limit what the sample size buys

Strategy return kurtosis ranges from 7.0 (`volatility_targeted`) to 34.8
(`mean_reversion`), against 3.0 for a normal distribution. `sma_crossover`
deflates harder than `adaptive_ensemble` — 0.236 versus 0.549 at 5 trials —
despite having no search behind it, because its skew of -0.86 and kurtosis of
24.3 are worse than `adaptive_ensemble`'s -0.22 and 14.8. Higher moments
inflate the standard error of the Sharpe estimate independently of any
selection effect.

## 8. The conclusion is insensitive to the trial count

The deflated Sharpe penalty grows as sqrt(2 ln N), so moving from 11 trials to
500 costs less than moving from 1 to 11. Even at n_trials=1 — no search at all
— neither `sma_crossover` (0.683) nor `adaptive_ensemble` (0.906) exceeds 0.91
against the benchmark. The result does not depend on getting N exactly right.

The eleven named strategies are a floor, not a ceiling: each has tunable
parameters, so the true trial count is higher.

## 9. Block bootstrap and IID bootstrap agree here, and would not elsewhere

On real daily returns the two methods give nearly identical Sharpe intervals
(`sma_crossover`: 0.820 versus 0.816 width). Sharpe depends on the mean and
standard deviation, both largely insensitive to ordering, so preserving
volatility clustering changes little.

On a synthetic AR(1) series with phi=0.6 the methods diverge sharply — widths
of 3.690 versus 1.609, with the naive interval failing to cross zero where the
honest one does. The block bootstrap would matter far more for path-dependent
statistics such as maximum drawdown or time-to-recovery, which depend heavily
on ordering.

Confidence intervals move by at most 0.04 across mean block lengths of 10, 20,
and 40, so the results do not depend on that choice.

## 10. Confidence intervals had to be made opt-in

Adding a 2,000-replicate bootstrap to `full_report` unconditionally made it
5,000x slower — 0.94ms to 5,270ms. `compare_strategies` calls it 22 times per
invocation, and Streamlit reruns the page script on every widget change, so the
Validation page took roughly 20 seconds per interaction. The test suite went
from a few seconds to 3m12s.

`full_report` now takes `n_boot=0` by default and skips the bootstrap entirely;
only the two call sites that render a confidence interval pass a nonzero value,
at 200 replicates. The default path is back to 0.94ms and the test suite to
9.4s.

## 11. Unpinned tooling makes CI nondeterministic

CI failed twice on mypy errors that did not reproduce locally, and in opposite
directions: first that `np.log(Series)` returns an ndarray so `.rolling()` is
invalid, then that the same expression returns an ndarray so it cannot be
assigned to a `pd.Series` variable. Both runs agreed with each other and
disagreed with local mypy 1.10.0, which infers a Series. At runtime it is a
Series, since pandas implements `__array_ufunc__`.

The code was never wrong. Local and runner had resolved different mypy and
pandas-stubs versions, which disagree about numpy ufunc return types on pandas
objects. Ruff, mypy, and pandas-stubs are now pinned to exact versions, since
an unpinned linter or type checker turns CI into a nondeterministic test that
can fail on a commit which changed nothing relevant.

## 12. Cross-sectional momentum: the signal works long, the construction fails

12-1 momentum on the S&P 100, 2008-2025, decile long-short, equal weight,
monthly rebalance, 5bps. Combined Sharpe 0.001 [-0.398, 0.430]. Max drawdown
-77.0%, annualized volatility 26.0%, turnover 12.95x.

The long leg alone has a Sharpe of 1.071 [0.727, 1.441] — the only strategy in
this project whose confidence interval clearly excludes zero. The short leg
alone is -0.882 [-1.281, -0.483], significantly negative. The short leg exactly
cancels a working long leg.

Equal dollar weights did not produce equal market exposure. The winner decile
skewed defensive (AMGN, BMY, MCD, SO, WMT recur), giving the long leg a beta of
+0.789, while shorting the loser decile gave a normal -1.003. Net beta -0.214,
CI [-0.384, -0.072], against a universe that returned +1059% over the period.

A log-space decomposition attributes 88.7% of the -45% total loss to that beta
mismatch and 11.3% to signal and volatility drag combined. The first attempt
summed simple returns and failed to reconstruct the actual loss, because
returns compound multiplicatively; at 26% annualized volatility the drag term
is roughly 3.4% per year.

The 2009 short leg held AIG, BAC, C, MS, AXP, COF, GE, and F going into March —
the beaten-down financials — and lost 73.4% that year while the long leg gained
23.7%. That is the Daniel & Moskowitz momentum crash reproduced directly. It is
not, however, what killed the strategy: March-May 2009 and February-April 2020
each account for about 1% of accumulated drawdown. The short leg lost money in
15 of 17 years.

## 13. Hedging the beta recovers about half of what it should

A beta-neutral variant rescales the short leg by trailing 252-day causal leg
betas, recomputed monthly. Net beta halves to -0.131, though the CI
[-0.218, -0.050] still excludes zero — trailing beta is a noisy forecast of
next month's realized beta, with scale factors ranging 0.26 to 2.87.

Max drawdown halves to -42.5%, kurtosis falls from 14.8 to 7.7, skew from -1.03
to -0.42, and 2009 improves from -63.6% to -12.4%. The hedge targets exactly the
mechanism it was designed for.

Sharpe rises to 0.196, but the CI [-0.170, 0.569] still contains zero and
overlaps the unhedged interval, so the improvement is directional rather than
established. Non-crisis years get worse — 2019 falls from -8.3% to -26.7%, and
2014 and 2017 also deteriorate — because in years where the problem was not
beta, the hedge adds sizing noise without addressing the cause. Turnover rises
from 12.95x to 14.60x.

Beta was the majority of the problem but not all of it. The short leg's
standalone Sharpe of -0.882 excludes zero on its own terms: past losers on this
universe did not keep underperforming enough to make shorting them profitable.

Note on benchmarking: a beta-neutral book is not designed to capture the
market's return, so its -0.602 difference versus equal-weight buy-and-hold
conflates stock selection with the intentional absence of market exposure. The
absolute Sharpe is the relevant read for this construction.

The universe is current S&P 100 membership, so companies that left the index
between 2008 and 2025 are absent. Those are disproportionately the sustained
underperformers a short leg would have profited from, so the survivorship bias
runs in the strategy's favor and this negative result is if anything
understated.

## 14. Refitting per fold removes a measurable share of apparent edge

The rolling walk-forward previously generated each strategy's signal once on
the full series and sliced it into folds, so a fitted strategy's model had seen
every fold's data before being evaluated on any of it. Refitting on each fold's
own training window changes mean fold Sharpe in a pattern that tracks how much
selection each strategy performs.

Strategies whose behaviour depends on a learn_frac-based, Sharpe-selected
choice lose the most: regime_switch 1.35 to 0.86, adaptive_ensemble 1.26 to
0.79, regime_filtered 1.08 to 0.66. Under the fixed-model approach that
selection was made once on the whole 17-year series, so even early folds were
judged on a choice informed by later data.

Fitted classifiers lose less: ml_regime_conditional 1.19 to 0.99, ml_direction
1.37 to 1.18.

Strategies with no fit step — sma_crossover, momentum, mean_reversion,
volatility_targeted — are unchanged to two decimals, which is the control that
confirms the mechanism. regime_sized and regime_parameters rise slightly
(+0.04, +0.06); neither has a Sharpe-driven selection step, so their only
per-fold variation is estimation noise in the unsupervised clustering.

adaptive_ensemble, the best performer in the main results table, loses 0.48 of
its mean fold Sharpe. A meaningful share of its apparent edge was a fitting
artifact.

Passing a precomputed regime object into a per-fold refit is incompatible by
construction: an early fold's refit would be judged against a regime model
fitted on the entire series. The function raises rather than silently producing
a number that looks honest and is not.

Warm-up costs nothing at the default train_days of 756, since the longest
lookback in the codebase (sma_crossover's 200 days) sits entirely inside each
fold's training slice. It would cost max(0, warmup_days - train_days) test days
per fold if train_days were reduced below a strategy's own lookback.

These deltas apply to the walk-forward numbers only. The main results table's
Sharpe ratios, confidence intervals, paired differences, and deflated Sharpes
come from full-period backtests and a single split, which never call
rolling_walk_forward and did not move.

## Limitations

Single asset, single 17-year window, long/flat positions only. No shorting, no
leverage, no cross-sectional universe. Costs are a flat 5bps one-way spread
proxy with no market impact, borrow, or financing. Regime labels come from a
3-state HMM fit on the first 70% of history; the walk-forward variant is
available but the reported numbers use the fixed fit. Walk-forward folds are sliced from a full-length backtest, so each fold
inherits the position going into it rather than starting flat. Fold-level
costs therefore exclude the entry cost of initiating a position at the fold's
start. Headline metrics for fitted strategies come from full-period backtests in
which the model predicts across its own training period — 70% of the equity
curve for ml_direction and the learn_frac-based adaptive wrappers. The
rolling walk-forward results in results/summary.md refit each fold and are
the honest read for these strategies; the gap is 0.19 to 0.49 Sharpe
depending on how much selection the strategy performs. Since every paired
difference against buy-and-hold already contains zero, correcting this would
strengthen rather than change the conclusion.