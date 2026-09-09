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

## Limitations

Single asset, single 17-year window, long/flat positions only. No shorting, no
leverage, no cross-sectional universe. Costs are a flat 5bps one-way spread
proxy with no market impact, borrow, or financing. Regime labels come from a
3-state HMM fit on the first 70% of history; the walk-forward variant is
available but the reported numbers use the fixed fit. Fitted strategies are not
refit per fold in the rolling walk-forward.