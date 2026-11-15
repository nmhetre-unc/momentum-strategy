# Findings

SPY, 2008-01-01 to 2025-01-01, 4,278 trading days. Costs 5bps one-way on
position change unless stated. Bootstrap confidence intervals are 90%,
stationary block bootstrap, n_boot=2000, mean block 20. Deflated Sharpe
follows Bailey & López de Prado (2014).

## 1. No strategy beats buy-and-hold by a detectable margin

Eleven strategies were tested. No strategy is scored on data it learned from.
The four that learn nothing (`sma_crossover`, `momentum`, `mean_reversion`,
`volatility_targeted`) are scored over the full period. The seven that learn
something from history (a fitted model, a regime model, or a choice selected
on past performance) are scored only on their out-of-sample window, against
buy-and-hold over the identical window. That window starts 2019-11-22 for the
two ML strategies and 2020-04-08 for the regime wrappers.

Every one of the eleven produces a 90% confidence interval on the Sharpe
difference versus buy-and-hold that contains zero. Over the full period the
closest is `volatility_targeted` at +0.22 [-0.08, +0.54], against
buy-and-hold's 0.607. Out of sample the closest are `regime_filtered` at +0.18
[-0.31, +0.72] and `regime_sized` at +0.11 [-0.42, +0.64], against
buy-and-hold's 1.15 over 2020-04-08 to 2024-12-31. Those windows are 1,191 to
1,284 days, under a third of the full period, so their intervals are wide.

Earlier versions of this section named `adaptive_ensemble` closest, at +0.324
[-0.045, +0.686], and before that `ml_direction` second, at +0.10 [-0.01,
+0.20]. Both were scored over the full period, days they had learned from
included. Before version 2, the regime wrappers held positions on 846 to
2,335 of their 3,088 in-sample days, and `regime_sized` sized every position
using the volatility of later days (`results/ml/learning_strategies_audit.json`;
for `ml_direction`, `results/ml/daily_ml_audit.json`). Out of sample,
`adaptive_ensemble`'s difference is -0.25 [-0.87, +0.38].

Of the full-period strategies, `mean_reversion` is the only one that plausibly underperforms: -0.14
difference, deflated Sharpe against the benchmark of 0.038. Its kurtosis of
34.8 and -47.7% drawdown match the short-volatility payoff profile its own
documentation predicts — many small wins, occasional large loss. Out of
sample, `regime_parameters` (-0.35 [-0.77, +0.05], deflated Sharpe against the
benchmark 0.027) and `regime_switch` (-0.31 [-0.90, +0.33], 0.031) have lower
point estimates, on a window a third as long.

## 2. Comparing against zero instead of the benchmark inverts the conclusion

Each strategy's own Sharpe confidence interval excludes zero: `sma_crossover`
[0.35, 1.17] over the full period, `adaptive_ensemble` [0.19, 1.71] out of
sample. Read alone, both look like clear wins. But buy-and-hold's Sharpe over
the same window (0.607 full period, 1.15 out of sample) falls inside both
intervals, so neither is distinguishable from simply holding the index.

The same inversion appears in deflated Sharpe. Against zero, `sma_crossover`
stays above 0.90 at any trial count (0.998, 0.957, 0.902 at 1, 5, 11), and
`adaptive_ensemble` out of sample reaches 0.974 at one trial. Against the
benchmark, `sma_crossover` falls to 0.682, 0.236 and 0.126, and
`adaptive_ensemble` to 0.295, 0.042 and 0.015. Citing only the against-zero number is what a backtest implicitly does
when it reports a Sharpe ratio and calls it significant. (The figures are in
`results/summary.md`, "Deflated Sharpe by trial count".)

`adaptive_ensemble` used to be the example that held up best: 0.906, 0.549
and 0.380 against the benchmark at 1, 5 and 11 trials. Those came from its
full-period returns, which included the days it had learned from, and they
don't survive scoring it out of sample.

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

For `sma_crossover`, `momentum` and `mean_reversion` on this data, the learned
allow-list is [0, 1, 2] — no regime is ever excluded. The reason is structural: a long/flat
overlay on an asset that rose in every regime has no losing regime to exclude.
SPY was up even in the turbulent regime, which contains the 2009 and 2020
recoveries alongside the crashes.

Out of sample (2020-04-08 on, 1,191 days), `regime_filtered` differs from each
of those base strategies on 0 days. Before version 2 it was scored over the
full period, and there every difference was the UNKNOWN warm-up override: 97
days for `momentum`, 229 for `mean_reversion`, and 0 for `sma_crossover`, whose
own 200-day warm-up already covers the regime detector's. Those days fall in
what is now the in-sample period, where the wrapper holds no position.

With `ml_direction` as its base, `regime_filtered` now raises an error.
`ml_direction` holds no position in the allow-list's learning window, which
ends 2018-03-13, before the model's out-of-sample split. Its Sharpe would be 0
in every regime, and the positive-Sharpe rule would exclude them all and leave
the filter flat every day. It used to do exactly that, silently. Counts,
allow-lists and the error are in `results/regime_filter_override_days.json`
(`scripts/regime_filter_override_days.py`).

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

Full-period strategy return kurtosis ranges from 7.0 (`volatility_targeted`)
to 34.8 (`mean_reversion`), against 3.0 for a normal distribution. Higher
moments inflate the standard error of the Sharpe estimate independently of
any selection effect: `sma_crossover`'s skew of -0.86 and kurtosis of 24.3
cost it more deflation than its Sharpe alone would suggest.

This section used to illustrate that with `sma_crossover` deflating harder
than `adaptive_ensemble` (0.236 versus 0.549 at 5 trials), on account of
`adaptive_ensemble`'s milder -0.22 and 14.8. Those were `adaptive_ensemble`'s
full-period figures, learning period included. Out of sample it deflates to
0.042, so the comparison no longer isolates the higher moments.

## 8. The conclusion is insensitive to the trial count

The deflated Sharpe penalty grows as sqrt(2 ln N), so moving from 11 trials to
500 costs less than moving from 1 to 11. Even at n_trials=1 — no search at all
— neither `sma_crossover` (0.682) nor `adaptive_ensemble` (0.295, out of sample)
exceeds 0.91 against the benchmark. The result does not depend on getting N
exactly right. (This section used to quote 0.683 and 0.906: the first was a
rounding slip for 0.6824, the second `adaptive_ensemble`'s full-period figure.)

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

## 14. Refitting per fold: what is left once no fold is in sample

The rolling walk-forward used to generate each strategy's signal once on the
full series and slice it into folds, so a fitted strategy was scored on folds
it had learned from. Refitting on each fold's own training window took 0.42
to 0.49 off the mean fold Sharpe of the three Sharpe-selected wrappers
(regime_switch 1.35 to 0.86, adaptive_ensemble 1.26 to 0.79, regime_filtered
1.08 to 0.66), and those numbers were read as the size of the fitting
artifact.

Now a learning strategy fitted once on the whole series holds no position
before its first out-of-sample day, and only folds where it can hold a
position are averaged. For the fixed model that is the last 8 of 27 folds for
the regime wrappers (9 for the ML strategies), and over the same folds the
comparison changes character. regime_switch still loses 0.40 (1.29 to 0.89)
and adaptive_ensemble 0.36 (1.21 to 0.84) when refit per fold, but
regime_filtered loses only 0.03 (1.44 to 1.41). regime_sized, ml_direction and
ml_regime_conditional gain 0.05 to 0.09, and regime_parameters gains 0.56
(0.73 to 1.29). On these folds the fixed model learned nothing from the data
it is scored on, so a remaining gap is not leakage. It is the difference
between a choice learned from ten years and one learned from each fold's
three, measured on 8 folds, and it runs in both directions.

Refit per fold, every strategy can hold a position in all 27 folds. The means
are regime_switch 0.86, adaptive_ensemble 0.79, regime_filtered 0.68 (0.66
before version 2), regime_sized 0.99, regime_parameters 0.96, ml_direction
1.19 and ml_regime_conditional 0.99. regime_filtered moved because a wrapper
whose learning window ends at the fold boundary now starts each fold flat and
pays its entry cost, instead of inheriting a position from its training
window. Its test-window positions are otherwise identical. regime_sized's are
not: version 1 sized each fold from volatility that included that fold's own
test window, so even its refit-per-fold number used the data it was scored on.
Its mean is 0.99 either way.

Strategies with no fit step — sma_crossover, momentum, mean_reversion,
volatility_targeted — are scored on all 27 folds either way and are
unchanged, which is the control. The counts and means are in
`results/summary.md`.

Passing a precomputed regime object into a per-fold refit is incompatible by
construction: an early fold's refit would be judged against a regime model
fitted on the entire series. The function raises rather than silently producing
a number that looks honest and is not.

Warm-up costs nothing at the default train_days of 756, since the longest
lookback in the codebase (sma_crossover's 200 days) sits entirely inside each
fold's training slice. It would cost max(0, warmup_days - train_days) test days
per fold if train_days were reduced below a strategy's own lookback.

These deltas apply to the walk-forward numbers only. The comparison tables'
Sharpe ratios, confidence intervals, paired differences and deflated Sharpes
come from single backtests and never call rolling_walk_forward.

## Limitations

Single asset, single 17-year window, long/flat positions only. No shorting, no
leverage, no cross-sectional universe. Costs are a flat 5bps one-way spread
proxy with no market impact, borrow, or financing. Regime labels come from a
3-state HMM fit on the first 70% of history; the walk-forward variant is
available but the reported numbers use the fixed fit. For a strategy that
learns nothing, walk-forward folds are sliced from a full-length backtest, so
each fold inherits the position going into it rather than starting flat, and
fold-level costs exclude the entry cost of initiating it. A strategy whose
learning window ends at the fold boundary starts each fold flat instead.

The seven learning strategies are scored out of sample only, over 1,191 to
1,284 days (2019-11-22 or 2020-04-08 to 2024-12-31). That is under a third
of the full period and a single, mostly rising market, so their intervals are
wide and say little about other conditions. The regime wrappers learn their
choice from data to 2018-03-13 but their regime model from data to 2020-04-07,
so the two years between are neither used for the choice nor scored. The
seven are compared with buy-and-hold over their own window, so their rows
can't be ranked directly against the full-period rows.