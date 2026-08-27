"""
The teaching content for the dashboard, kept as plain data.

Separated from app.py deliberately: the explanations are the actual
product of a training platform, they're long, and they want to be
editable without touching layout code. Nothing here imports streamlit,
so this module is also usable from a notebook or the CLI.

METRIC_DOCS feeds the tooltips next to every number. QUANT_NOTES feeds
the collapsible "Quant note" sections. PITFALLS and LEARNING_PATH feed
the Learn tab and the README.
"""

# --------------------------------------------------------------------------
# Tooltips: one line each, attached to the metric they explain
# --------------------------------------------------------------------------
METRIC_DOCS = {
    "total_return": "Cumulative growth over the whole period. Says nothing about how much risk was taken to get it, or whether it arrived smoothly.",
    "cagr": "The constant annual growth rate that would produce the same ending value. Comparable across periods of different length, unlike total return.",
    "annualized_volatility": "Standard deviation of daily returns, scaled to a year. The denominator of the Sharpe ratio, and the thing position sizing controls.",
    "sharpe_ratio": "Excess return per unit of total volatility. Above ~1.0 is good for a single strategy; above 2 on a daily backtest usually means a bug or lookahead bias. Assumes returns are roughly normal, which market returns are not — it understates tail risk.",
    "sortino_ratio": "Like Sharpe, but only downside deviation is in the denominator. Higher than Sharpe means the volatility was mostly upside, which is the good kind.",
    "max_drawdown": "Worst peak-to-trough loss. The number that actually decides whether a strategy is survivable — this is what you would have had to sit through, and what a risk manager would have cut you at.",
    "num_trades": "Count of position changes. Very low counts mean your metrics rest on a handful of independent bets, no matter how many days the backtest covers.",
    "win_rate": "Share of non-flat days that were profitable. Deliberately weak on its own: a strategy can win 70% of days and still lose money if the 30% are much larger.",
    "exposure": "Average absolute position. A Sharpe of 1.0 at 20% exposure and at 100% exposure are very different results — the first used a fifth of the capital and a fifth of the observations.",
    "turnover": "Position change per year, in full-position units. Turnover of 20 means you turned the book over 20 times — at 5bps a side that is roughly 1% a year of pure cost.",
    "benchmark_return": "Buy-and-hold over the same window. The bar every strategy has to clear, and the one most backtests quietly omit.",
    "test_base_rate": "Accuracy you'd get by always predicting the majority class. On daily equity data this is about 53% — any model at or below it has learned nothing.",
    "sharpe_decay": "In-sample Sharpe minus out-of-sample Sharpe. Large positive values mean the in-sample number was substantially fitted to that period.",
}

# --------------------------------------------------------------------------
# Collapsible notes, keyed by the section they belong to
# --------------------------------------------------------------------------
QUANT_NOTES = {
    "equity_curve": {
        "title": "How to read an equity curve",
        "body": """
An equity curve is the growth of $1 in the strategy. Read it in this order,
and resist reading the endpoint first — the endpoint is the least
informative part of the chart.

**1. The shape, not the endpoint.** Two strategies can end at 2.0x with
completely different characters: one rising steadily, one flat for four
years and then doubling in six months. The second one's return belongs to
a specific market episode. If that episode doesn't repeat, neither does
the return.

**2. The flat stretches.** Flat means out of the market or churning.
Ask how long the longest one is. A strategy with an 18-month flat period
is one you would have abandoned in month nine, which means its full-period
return was never actually available to you.

**3. The drawdowns underneath.** Always read the drawdown chart with the
equity curve, never alone. The equity curve shows what you earned; the
drawdown shows what you had to endure to earn it. -50% means you needed
+100% just to get back to even.

**4. The benchmark line.** A strategy that made 80% while buy-and-hold
made 120% did not make money. It lost 40% of what doing nothing would
have paid, plus the effort.

**5. The log-scale question.** On a linear axis, later years always look
more dramatic because the same percentage move is a bigger absolute one.
A curve that looks like a hockey stick often looks like a straight line
in logs — which is what steady compounding actually is.
""",
    },
    "walk_forward": {
        "title": "Why walk-forward validation matters",
        "body": """
Any strategy can be made to look good on a fixed history. You have
parameters; the history is finite; some setting of the parameters was
best on it. Finding that setting and reporting its performance is not
research, it's curve-fitting with extra steps.

Walk-forward is the cheapest honest defense. Fit or choose on one slice,
evaluate on a slice you never looked at, and report only the second
number.

**What the split protects against**
- *Parameter mining* — trying 50 window lengths and keeping the winner.
- *Strategy mining* — trying 20 strategies and keeping the winner.
- *Silent fitting* — the kind you do by choosing which chart looked good.

**What it does NOT protect against**
- Looking at the out-of-sample result, going back, changing something,
  and looking again. Do that three times and your out-of-sample period
  is in-sample. There is no technical fix; only discipline.
- Survivorship in the ticker itself. Backtesting on SPY over 2010-2025
  is backtesting on an asset you already know went up.

**Reading the result.** A modest drop from in- to out-of-sample is
normal and expected. A collapse — Sharpe 1.8 to 0.1, or a sign flip — means
the in-sample number was mostly fitting. What you want is not a high
out-of-sample number but a *small gap*, because the gap is what tells you
whether the process generalizes.

**One split is not enough.** A single 30% holdout is one draw from a
distribution. `rolling_walk_forward()` gives you ten or fifteen
consecutive holdouts instead. The share of positive folds is more
informative than the average Sharpe across them.
""",
    },
    "ml_overfitting": {
        "title": "Why the ML model overfits (and what that teaches)",
        "body": """
Daily equity direction is close to a coin flip with a slight upward
tilt — roughly 53% of days are up. That leaves very little signal, and
the noise is enormous.

Now give a random forest 12 features and 1,700 training rows and ask it
to predict that. It will find patterns. There are patterns in any 1,700
rows of noise; a flexible model's job is to find them, and it is good at
its job. The result is the classic signature:

```
random_forest   train_acc = 86%   test_acc = 42%
logistic        train_acc = 58%   test_acc = 49%
```

**Test accuracy below 50% is not a broken model.** It means the patterns
it memorized were not merely useless but actively misleading on new data.
That is what memorizing noise produces.

**Why logistic regression looks worse and is better.** It can only fit a
linear combination of features. That constraint is a feature: with so
little signal, low capacity means less room to fit noise. The general
rule — the less signal there is, the simpler your model should be — is
backwards from most people's intuition, which is to reach for a bigger
model when results disappoint.

**Compare against the base rate, always.** A model at 53% test accuracy
has matched "always predict up". It has added nothing. The dashboard
shows the base rate next to test accuracy for exactly this reason.

**Accuracy is not P&L.** A model can be 55% accurate and lose money, if
the 45% it gets wrong are the large-move days. Read the strategy's
out-of-sample Sharpe, not its accuracy.

**How to say this in an interview.** "I built an ML layer, my own
walk-forward validation caught it overfitting, and here's what that
told me about capacity versus signal." That is a stronger answer than
any good backtest, because a good backtest invites the question of what
you did wrong to get it.
""",
    },
    "regimes": {
        "title": "What a market regime is (and isn't)",
        "body": """
A regime is a persistent state of the market with its own statistical
character: its own volatility level, its own tendency to trend or revert,
its own correlation structure. Regimes matter because the strategies in
this project are all implicitly conditional on one — a trend follower is
a bet that you are in a trending market, whether or not you said so.

**Three things a real regime has**
1. *Persistence.* Regimes last weeks to months. If your detected regimes
   average four days, you have detected noise. Check the diagonal of the
   transition matrix: it should be 0.95+ on daily data.
2. *Statistical distinctness.* The regimes must actually differ in return
   or volatility. Read the summary table. If all three rows look alike,
   there is nothing there to condition on and no downstream cleverness
   will create some.
3. *Recurrence.* A regime that occurred once is an event, not a regime.
   You cannot build a rule from a single instance.

**What regimes are not.** They are not predictions. Detecting that you
are currently in a high-volatility regime says nothing about when it will
end. The value is in *conditioning* — behaving differently given where
you are — not in *forecasting* the switch.

**The labels are estimates.** Every regime label carries uncertainty, and
that uncertainty is largest exactly at the transitions, which is exactly
when acting on the label matters most. This is the central difficulty of
the whole field, and no smoothing parameter makes it go away.
""",
    },
    "regime_lookahead": {
        "title": "The regime lookahead trap",
        "body": """
This is the single most common way to produce a spectacular and
completely fake regime-strategy backtest.

You fit a clustering model or an HMM on your full price history. It finds
three regimes. You label every day, run a strategy that trades
differently in each, and the equity curve is beautiful.

**It is beautiful because the labels contain the future.** The cluster
centers were computed from all the data, 2020 included. When the model
labels a day in 2015 as "pre-crisis", it is using information that did
not exist in 2015. You have built a strategy that knows what happens
next and is unsurprised that it performs well.

**Two honest constructions, both in regime.py**

- `detect_regimes(fit_frac=0.6)` — fit on the first 60% only, apply the
  frozen model afterwards. Simple, and the out-of-sample portion is clean.
- `detect_regimes_walk_forward()` — refit on an expanding window every
  quarter, label only forward. Slower, and much closer to what you could
  genuinely have run. The first two years get no labels at all, because
  you genuinely had no model then.

**Do the comparison yourself.** Run the same adaptive strategy on
full-sample labels and on walk-forward labels. The gap between the two
equity curves is the lookahead bias, drawn to scale, on your data. It is
usually much larger than people expect, and seeing it once is worth
more than reading about it ten times.

**Smoothing is part of this too.** A centered rolling filter on the
labels looks tidier and uses future days. Every smoother in this project
is backward-looking only, and pays for it in lag.
""",
    },
    "adaptive": {
        "title": "How adaptive strategies behave",
        "body": """
There are four distinct ways to be regime-aware, and they behave very
differently. Know which one you're running.

**Filtering** — sit out bad regimes. Only removes exposure, never invents
any, so its downside is bounded: at worst you remove the wrong days. This
is usually the mechanism that survives out-of-sample, and it's the one to
try first.

**Switching** — different strategy per regime. The most intuitive and
the most disappointing in practice. Two costs eat it: smoothed labels
arrive several days late, and the first days of a new regime are where
the move is largest; and switching flips the whole position, which costs
real money. Enable `cost_bps` before believing any switching result.

**Re-parameterizing** — same strategy, regime-specific settings. Easiest
to overfit, because every regime hands you a fresh parameter set. Three
regimes times two parameters is six numbers fitted to one history.

**Position sizing** — scale the size by volatility or regime. Usually the
best return on complexity in the whole project. Volatility is genuinely
persistent, so trailing volatility forecasts tomorrow's reasonably well,
and sizing down going into turbulence cuts drawdown far more than it cuts
return.

**The result you should expect, and report honestly.** Very often plain
volatility targeting — which uses no regime model at all — beats every
regime-based mechanism here. If that's what your data says, that is the
finding. "The simple version captured most of the benefit" is a real
result, and preferring it is what good quant judgment looks like.

**Always check exposure.** An adaptive strategy that is flat 70% of the
time has a Sharpe ratio computed on 30% of the days. It may be excellent;
it is definitely being measured on a small sample.
""",
    },
    "risk_by_regime": {
        "title": "Why risk metrics change across regimes",
        "body": """
A full-period Sharpe ratio is an average over every market condition in
the sample. Averages hide structure, and here the structure is the point.

Take a trend-following strategy with a full-period Sharpe of 0.6. Split
it by regime and you typically find something like +1.4 in the trending
regime and -0.5 in the choppy one. Neither of those is 0.6. The blended
number describes a market that never existed — one that was permanently
half-trending.

**What this buys you**
- If the strategy loses in a regime you can identify in real time, you
  can stop trading it there. That's `regime_filtered`, and it's the most
  direct payoff from the whole regime layer.
- If the strategy's edge is concentrated in a rare regime, your effective
  sample is much smaller than the day count suggests, and your confidence
  should shrink accordingly.

**Read the `days` column before anything else in the table.** A Sharpe
ratio over 40 days has an error bar wide enough to contain almost any
conclusion. Standard error on an annualized Sharpe is roughly
sqrt(252/N) — so 40 days gives you about ±2.5. Any per-regime number
built on fewer than a couple hundred days is a hint, not a finding.

**Regime mix explains most performance changes.** When a strategy decays
out-of-sample, compare the regime mix between the two periods before
concluding the strategy broke. Often it didn't: the market simply served
up more of the regime it dislikes.
""",
    },
    "fair_comparison": {
        "title": "How to compare strategies fairly",
        "body": """
Most strategy comparisons are unfair in at least one of these ways.

**Different exposure.** A strategy invested 30% of the time and one
invested 100% of the time are not comparable on total return. Compare
risk-adjusted metrics, and read the exposure column.

**Ignored costs.** Turnover varies wildly across these strategies — the
SMA crossover trades twice a decade; the ML model trades most days. At
zero cost the comparison flatters the high-turnover strategy enormously.
Set `cost_bps` to something realistic (5-10 for liquid ETFs) and watch
the ranking change.

**In-sample vs out-of-sample mixing.** The ML strategy's full-period
number includes its own training data. Comparing that against a
rule-based strategy's full-period number compares an in-sample result
against an honest one.

**No benchmark.** Every one of these should be read against buy-and-hold
on the same asset over the same dates. Many "good" strategies are worse
than doing nothing.

**Selection bias — the subtle one.** If you compare twenty strategies and
report the best one's out-of-sample Sharpe, that number is no longer
out-of-sample: you used the holdout period to make the choice. With
twenty candidates on random data, the best out-of-sample Sharpe will look
respectable purely by chance. The honest report is the whole table, which
is why `compare_strategies()` returns all rows and never picks a winner.
""",
    },
    "position_sizing": {
        "title": "Position sizing: the underrated half",
        "body": """
Most effort in retail quant work goes into the entry signal. Most of the
improvement available is in sizing.

The reason is that **volatility is far more predictable than direction**.
Tomorrow's direction is close to a coin flip. Tomorrow's volatility is
strongly predicted by today's — volatility clusters, which is why the
`vol_20d` regime feature does so much of the work in every clustering
model here.

So: you can't reliably know which way it goes, but you can reliably know
how much it will move. Size accordingly.

`volatility_targeted()` implements the standard version:

```
position = signal x (target_vol / trailing_realized_vol)
```

capped at 1.0. When realized volatility doubles, the position halves.
The typical effect on equity data is that returns are roughly preserved
while maximum drawdown falls noticeably, because the position was already
small when the crash arrived rather than being cut after it.

**What to check.** Turnover — continuous resizing trades every day, and
`regime_sized()` is the cheaper discrete cousin that only resizes at
regime boundaries. And exposure: a target of 15% against an asset
realizing 20% means you are usually well under a full position, so your
absolute return will be lower even if your Sharpe is better.
""",
    },
    "costs": {
        "title": "Transaction costs change conclusions, not just numbers",
        "body": """
`run_backtest(..., cost_bps=...)` charges a cost proportional to position
change. Default is 0, which is convenient and wrong.

At 5bps per unit of turnover, a strategy turning over 20x a year pays
about 1% annually. On a strategy with a 4% expected return, that is a
quarter of the edge. At 10bps it is half.

**This does not shift all strategies equally.** It penalizes exactly the
strategies that look best in a frictionless backtest — high-turnover ML
signals and regime-switching systems that flip the entire position. It
barely touches a 200-day crossover. Turning costs on frequently reorders
the ranking table, and the post-cost ranking is the real one.

**Realistic figures.** Liquid ETFs: 1-5bps. Large-cap single stocks:
5-10bps. Small caps or size: much more, and it grows with your position.
This project models cost as a constant, which understates the real thing
for anything but small trades in liquid names.
""",
    },
    "trend_in_chop": {
        "title": "Why trend-following bleeds in sideways markets",
        "body": """
This is the most important structural failure in the whole project, and it
is not a bug or a tuning problem. It is what the rule does by construction.

**The mechanism.** An SMA crossover has no concept of "range". It only
knows whether the fast average sits above the slow one. In a sideways
market price oscillates across both averages, and each oscillation fires
the same sequence:

1. Price rises enough to pull the fast average above the slow one — but
   by the time a *lagging average* has crossed, price is already near the
   local top. You buy high.
2. Price falls back through the middle of the range. The averages cross
   back down, again late. You sell low.
3. Repeat.

The strategy systematically buys near local highs and sells near local
lows, and it does so *every time the market goes nowhere*. The lag that
makes moving averages useful in a trend is exactly what makes them
poisonous in a range.

**What the equity curve looks like.** A staircase: long flat-to-declining
stretches punctuated by a few large gains. Trend-following is often
described as paying an insurance premium in quiet markets to collect a
large payout in trending ones. The premium is those flat stretches.

**Why win rate is useless here.** The strategy loses on most trades and
makes its money on a handful of large ones. A 35% win rate is normal and
says nothing about whether it works. Read the payoff profile instead.

**Why this is hard to hold.** The flat stretches can run for a year or
more. Most people abandon the strategy during one, which means the
full-period return in the backtest was never actually available to them.

**The practical response.** Don't trade the strategy in the regime where
it structurally loses — that's `regime_filtered` on the Adaptive page.
Whether you can identify that regime *in real time*, rather than in
hindsight, is a separate and much harder question, and it's the one the
Regimes page exists to make you confront.
""",
    },
    "drawdown_vs_return": {
        "title": "Why drawdown matters more than return",
        "body": """
Return is what a strategy earned. Drawdown is what you had to survive to
collect it. Only one of those decides whether you were still holding at
the end.

**The arithmetic is asymmetric, and brutally so.**

| Drawdown | Gain needed to recover |
|---|---|
| -10% | +11% |
| -25% | +33% |
| -50% | +100% |
| -75% | +300% |

A 50% loss does not need a 50% gain to undo. It needs a double. This is
why capital preservation is not timidity — it's arithmetic.

**Three reasons the drawdown number binds harder than the return number:**

1. **You get stopped out.** At a fund, a drawdown limit is a hard rule; a
   -20% drawdown often means your book is cut regardless of what the
   backtest says happens next. The backtest keeps holding. You don't.
2. **You stop yourself.** The psychological version of the same thing,
   and more common. A backtest never panics in month nine of an 18-month
   underwater stretch.
3. **It bounds your leverage.** How large you can run a strategy is set
   by its worst case, not its average. Halving drawdown at constant
   return means you can run twice the size.

**How to read the drawdown chart.** Depth is only half of it. Also read
**duration** — how long the curve stayed underwater. A -15% drawdown that
recovers in a month is an inconvenience; a -15% drawdown that takes two
years to recover is a strategy you would have abandoned.

**The corollary for this dashboard.** When a change (volatility targeting,
regime filtering) leaves return roughly flat and cuts max drawdown
materially, that is a *large* improvement even though the headline return
barely moved. Most of the value in position sizing shows up here and
nowhere else.
""",
    },
    "sharpe_can_mislead": {
        "title": "Why the Sharpe ratio can mislead you",
        "body": """
Sharpe is the default risk-adjusted metric because it is simple and
comparable. It is also wrong in specific, knowable ways, and quoting it
without the caveats is how people get caught out.

**1. It assumes returns are roughly normal. They are not.** Market
returns have fat tails and negative skew — crashes are far larger and
more frequent than a normal distribution predicts. Sharpe uses standard
deviation, which treats a -8% day as merely unusual rather than as the
thing that ends your fund. It systematically understates tail risk.

**2. It punishes upside volatility.** A strategy that jumps +10% is
penalized exactly as hard as one that drops -10%. That's why Sortino
exists — same formula, downside deviation only. When Sortino is much
higher than Sharpe, your volatility was mostly the good kind.

**3. It says nothing about path.** Two strategies with identical Sharpe
can have completely different drawdown profiles. Sharpe is computed from
the distribution of daily returns and is blind to their *ordering* — and
ordering is exactly what a drawdown is.

**4. It has a large error bar that nobody quotes.** The standard error of
an annualized Sharpe is roughly `sqrt(252/N)` for N days. Over one year
that's about ±1.0. Two strategies at 0.8 and 1.2 over a single year are
statistically indistinguishable. Most published Sharpe comparisons are
noise.

**5. It can be gamed by construction.** Strategies that sell options, or
that hold through small losses and rarely realize them, post excellent
Sharpe ratios right up until the one event that defines them. High Sharpe
with rare enormous losses is a recognizable and dangerous profile.

**6. Exposure hides inside it.** A Sharpe of 1.0 earned at 15% average
exposure and one earned at 100% exposure are very different results — the
first is computed on far fewer invested days, so its error bar is wider.

**How to use it anyway.** Read Sharpe *with* max drawdown, exposure, and
trade count. Above ~2 on a daily equity strategy, don't celebrate — go
looking for the lookahead bug, because that is much more often the
explanation than genuine edge.
""",
    },
}

# --------------------------------------------------------------------------
# The three pitfalls, in the order interns tend to hit them
# --------------------------------------------------------------------------
PITFALLS = {
    "Lookahead bias": {
        "summary": "Using information in a decision that wasn't available when the decision was made.",
        "where_it_hides": [
            "Acting on a signal on the same day's close it was computed from — the reason backtest.py shifts every signal forward one day.",
            "Standardizing or ranking features against full-sample statistics instead of expanding ones.",
            "Fitting a regime model on all of history and then backtesting on that history.",
            "Centered smoothing filters on labels or signals — they use future bars by construction.",
            "Using an adjusted-close series whose adjustments reflect later corporate actions.",
        ],
        "tell": "Implausibly high Sharpe. Above 2 on a daily equity strategy, assume a leak until you've found otherwise.",
        "fix": "For every input to a decision, ask what date it was knowable. If the answer is after the decision date, it's a leak.",
    },
    "Overfitting": {
        "summary": "Fitting the noise in your sample and mistaking it for signal.",
        "where_it_hides": [
            "Parameter search — trying many windows and keeping the best.",
            "Strategy search — trying many strategies and reporting the winner.",
            "High-capacity models on low-signal data (the random forest here).",
            "Regime-conditional models, which split your data while keeping the parameter count.",
            "Re-running after seeing the out-of-sample result, which converts it into in-sample data.",
        ],
        "tell": "A large gap between in-sample and out-of-sample performance, and results that change character with small parameter changes.",
        "fix": "Hold out data and only look once. Prefer fewer parameters. Check whether the result survives on a different ticker and a different period.",
    },
    "Regime drift": {
        "summary": "The market changes, and a relationship that was real stops being real.",
        "where_it_hides": [
            "A strategy fit on 2010-2019 — a single long low-volatility bull market — evaluated on it.",
            "Models trained on one volatility environment and deployed in another.",
            "Fixed parameters (a 200-day window) applied across decades with very different speeds.",
            "Regime models themselves: the regimes present in your training window may not be the ones you meet later.",
        ],
        "tell": "Rolling walk-forward folds that are consistently positive early and consistently negative later.",
        "fix": "Use rolling_walk_forward() and look at the fold sequence, not the average. Prefer strategies whose logic you can state a reason for, since those degrade more gracefully than ones you found by search.",
    },
}

# --------------------------------------------------------------------------
# Suggested order of study
# --------------------------------------------------------------------------
LEARNING_PATH = [
    {
        "stage": "1. Baselines",
        "goal": "Know what doing nothing looks like.",
        "do": "Run buy-and-hold on SPY, 2010-2025. Note the CAGR, Sharpe and max drawdown. Every strategy you build is competing with these three numbers.",
        "done_when": "You can state the benchmark's numbers from memory and you compare against them without being reminded.",
    },
    {
        "stage": "2. Rule-based strategies",
        "goal": "Understand what each strategy is actually betting on.",
        "do": "Run all three rule-based strategies on the same ticker and dates. For each, write one sentence on what market behaviour it needs in order to make money.",
        "done_when": "You can predict, before running it, which strategy will do better on a given chart.",
    },
    {
        "stage": "3. Validation",
        "goal": "Distrust your own backtests.",
        "do": "Walk-forward every strategy. Then run rolling_walk_forward() and look at the fold-by-fold results.",
        "done_when": "You instinctively ask 'in-sample or out-of-sample?' about any performance number, including your own.",
    },
    {
        "stage": "4. The ML layer",
        "goal": "See overfitting happen rather than read about it.",
        "do": "Train logistic and random forest. Compare train and test accuracy, and compare both against the base rate.",
        "done_when": "You can explain why the more accurate model is the worse model.",
    },
    {
        "stage": "5. Regimes",
        "goal": "Stop treating the market as one thing.",
        "do": "Detect regimes with rules, then k-means, then HMM. Compare transition matrices and average durations. Look at the per-regime performance table for a trend strategy.",
        "done_when": "You can point at a strategy's per-regime table and say where its return came from.",
    },
    {
        "stage": "6. The lookahead demonstration",
        "goal": "Internalize the most expensive mistake in the field.",
        "do": "Run the same adaptive strategy on full-sample regime labels and on walk-forward labels. Measure the gap.",
        "done_when": "You can quote the size of the gap you measured, on your data.",
    },
    {
        "stage": "7. Adaptive strategies",
        "goal": "Learn which mechanisms are worth their complexity.",
        "do": "Run filtering, switching, re-parameterizing and volatility targeting separately, with costs enabled. Attribute the result to a mechanism.",
        "done_when": "You reach for the simplest mechanism that explains the improvement, and you notice when the simple one wins.",
    },
    {
        "stage": "8. Write it up",
        "goal": "Communicate like a researcher, not a salesperson.",
        "do": "Write a one-page summary of one strategy: hypothesis, method, out-of-sample result, per-regime breakdown, what would falsify it, what you would do next.",
        "done_when": "The write-up leads with a limitation and is more convincing for it.",
    },
]
