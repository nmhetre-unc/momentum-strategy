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
    "regime_volatility_clusters": {
        "title": "Why volatility is the primary axis of every regime model",
        "body": """
Open any clustering model in this project, look at what actually separates
the clusters, and it will be volatility. That isn't a modelling accident —
it's the one property of markets that is reliably predictable.

**Volatility clusters. Direction does not.** Tomorrow's direction is close
to a coin flip: daily returns have almost no autocorrelation. Tomorrow's
*volatility* is strongly predicted by today's. Quiet days follow quiet
days; violent days follow violent days, for weeks at a time. This is one
of the oldest and most robust findings in empirical finance — it's the
observation the entire GARCH literature was built to model.

**So volatility is what a regime model can actually find.** A model
looking for persistent states in market data will land on volatility
states, because those are the states that persist. If you build a
three-regime model and the clusters differ mainly in volatility, the model
is working correctly, not failing to find something more interesting.

**Why this ordering is baked in here.** `regime.py` always renumbers
regimes by realized volatility, so regime 0 is the calmest and the highest
ID the most violent. That's what makes the IDs stable across refits,
methods and tickers — and it's why the color ramp treats regimes as an
*ordered* variable rather than as unrelated categories.

**Direction is the secondary axis, and it's much weaker.** Trend
separates regimes far less cleanly than volatility does, which is why the
naming scheme combines both ("Calm Uptrend", "Crisis / Selloff") but the
ordering uses only volatility. Be suspicious of any regime model that
claims to cleanly separate future direction — that's the thing nobody can
predict, and finding it usually means you leaked.

**The practical consequence.** If volatility is what you can forecast,
then sizing — not timing — is where the edge is. See
[[position_sizing]]: plain volatility targeting frequently beats every
regime-switching mechanism in this project, using no regime model at all.
""",
    },
    "regime_transition_persistence": {
        "title": "Persistence and transitions: reading the matrix",
        "body": """
The transition matrix answers one question per cell: given that today is
regime *i*, what is the probability tomorrow is regime *j*?

**Read the diagonal first, and read it before anything else on the page.**
Those cells are the persistence probabilities — P(tomorrow is the same as
today). On daily data they should be **0.95 or higher**. A regime that
persists 97% of days lasts about a month; one that persists 99% lasts a
quarter.

**Expected duration falls straight out of it.** If a regime stays with
probability `p`, its expected length is `1 / (1 - p)` days:

| p_stay | Expected duration |
|---|---|
| 0.90 | 10 days |
| 0.95 | 20 days |
| 0.98 | 50 days |
| 0.99 | 100 days |

**What a bad matrix looks like.** A diagonal near `1/k` — 0.33 for three
regimes — means tomorrow's regime is independent of today's. That is not a
regime model. It is a noisy day-classifier, and every strategy built on it
will do nothing but pay transaction costs. This is the single fastest
check on whether regime detection worked.

**The off-diagonal cells carry real information too.** Markets usually
don't jump from calmest to most violent in one step — they escalate
through the middle regime. If your matrix shows a large direct
calm→crisis probability, either the model is mislabelling, or the asset
genuinely gaps (which is itself worth knowing before you size a position
in it).

**Why persistence is the whole basis for acting on regimes.** Detecting
that today is turbulent is only useful if tomorrow is likely turbulent
too. If regimes didn't persist, the label would be a description of the
past with no predictive content, and conditioning on it would be pointless.
Persistence is what converts a *description* into something tradeable —
and it is exactly what an HMM models and what k-means and GMM ignore.
""",
    },
    "trend_vs_chop": {
        "title": "The second axis: trending vs. choppy markets",
        "body": """
Volatility tells you *how much* the market is moving. It says nothing about
whether that movement gets anywhere. Two markets can realize identical
volatility while one grinds steadily upward and the other thrashes and ends
where it started — and those two markets are opposite environments for
almost every strategy here.

**The efficiency ratio measures exactly this.** Kaufman's ratio is net
movement divided by total distance travelled over a window:

```
|price_today - price_60d_ago|  /  sum of |daily changes| over 60 days
```

Near **1.0**: the market walked in a straight line. Near **0.0**: it
travelled a long way and went nowhere. That second case is chop, and it is
where trend-following goes to die.

**The two families sit on opposite ends of this axis.**

- *Trend-following* (SMA crossover, momentum) needs high efficiency. It
  buys strength and needs that strength to continue. In chop it buys every
  local top and sells every local bottom — see [[trend_in_chop]] for the
  mechanism.
- *Mean-reversion* (RSI) needs low efficiency. It bets moves overshoot and
  get given back, which is precisely what chop does. In a strong trend it
  fights the move the whole way down.

**Which is why the pair is the motivating example for regime switching.**
If you could reliably tell which environment you were in, you could run
trend in one and reversion in the other and harvest both. The Adaptive page
tests whether you actually can — and the usual answer is "less well than
you'd hope", because labels arrive late and switching costs money.

**The related feature to watch.** `autocorr_60` — rolling lag-1
autocorrelation of returns — measures the same idea from the other
direction. Positive means moves follow through (momentum-friendly);
negative means they get reversed (reversion-friendly).

**One caution.** Efficiency ratio is backward-looking, like everything
else here. It tells you the last 60 days were choppy, not that the next 60
will be. It is useful because chop, like volatility, is somewhat
persistent — not because it forecasts.
""",
    },
    "regime_drift": {
        "title": "Regime drift: when the regimes themselves change",
        "body": """
There is a failure mode one level above "my strategy stopped working":
**the regime structure itself changes**, so a model that correctly learned
three regimes from 2008–2016 is describing a market that no longer exists.

**How it happens.** A regime model learns cluster centres — what "calm"
and "turbulent" *look like* — from its training window. But the meaning of
those words drifts:

- The 2017 volatility environment was so quiet that its "high volatility"
  regime would be classified as calm by a model trained through 2008.
- A model trained entirely on 2010–2019 never saw a genuine crisis, so it
  has no cluster for one. In March 2020 it must assign those days to
  *something*, and whatever it picks will be wrong.
- Market microstructure changes over decades — decimalization, the growth
  of ETFs, the rise of systematic flow. Relationships that held in 2005
  need not hold now.

**Why a single walk-forward split hides it.** One 70/30 split gives you one
out-of-sample number, and it can't distinguish "works generally" from
"worked until 2018 and never again". `rolling_walk_forward()` gives ten or
fifteen consecutive holdouts; read the folds **in sequence**. A strategy
that is positive early and negative late has drifted, and its average
across folds is a meaningless blend of two different worlds.

**Why walk-forward regime detection matters here specifically.**
`detect_regimes_walk_forward()` refits on an expanding window, so the
regime definitions themselves update as the market changes — which is both
more honest and more robust than freezing 2008's idea of "calm" forever.
The cost is noisier labels and no labels at all for the first two years.

**Two ways to tell drift from ordinary decay.** The Validation page's
regime-mix table is the diagnostic. If per-regime performance held up but
the *mix* of regimes shifted, the strategy is intact and your expectations
were built on a biased sample of history. If per-regime performance itself
deteriorated, that's real decay and no amount of regime timing fixes it.

**The uncomfortable implication.** Every backtest in this project is
conditioned on the specific history it ran over. A fifteen-year sample of
daily data contains perhaps three or four genuinely independent market
environments — which is a much smaller effective sample than 3,700 rows
suggests, and a much weaker basis for confidence than it feels like.
""",
    },
    "lookahead_bias": {
        "title": "Lookahead bias: the general case",
        "body": """
Lookahead bias is using information in a decision that was not available
when the decision was made. It is the most expensive mistake in
quantitative research, because it does not announce itself — it produces a
*beautiful* backtest and a strategy that loses money.

**The one question that catches almost all of it.** For every input to a
decision, ask: **what date was this knowable?** If the answer is after the
decision date, you have a leak. Applied honestly, that single question
finds more bugs than any amount of code review.

**Where it hides in a project like this one.**

1. **Same-bar execution.** Computing a signal from today's close and
   trading at today's close. This is why `backtest.py` shifts every signal
   forward one day, and it is the single most common beginner error.
2. **Full-sample preprocessing.** Any scaler, PCA, percentile rank, outlier
   clip or feature selection fitted on the whole dataset. The sample mean
   of 2010–2025 was not knowable in 2010. This project uses *expanding*
   z-scores and *expanding* percentile ranks for exactly this reason.
3. **Full-sample model fitting**, then backtesting over the same span — the
   regime-specific version of which is [[regime_lookahead]].
4. **Centered filters.** Any smoother that uses bars on both sides of the
   current one. A centered rolling median looks tidier on the chart and is
   pure leakage. Every smoother in `regime.py` is backward-looking only.
5. **Survivorship in the universe.** Backtesting on today's index members
   over ten years quietly excludes everything that went to zero.
6. **Restated data.** Fundamentals get revised; the value you can download
   today is not what was published then.
7. **The human kind.** You already know 2020 crashed and 2021 rallied.
   Choosing to test a crash filter is itself a decision informed by the
   future, and no code change fixes it.

**The tell.** Implausibly high Sharpe. Above ~2 on a daily equity
strategy, assume a leak and go looking, because that is far more often the
explanation than genuine edge. A suspiciously smooth equity curve is the
same signal.

**The habit worth building.** Ask of any result: *could I have produced
this number on that date, with only what existed then?* If you cannot
answer yes without qualification, the number is not evidence.
""",
    },
    "adaptive_filtering": {
        "title": "Filtering: the mechanism most likely to survive",
        "body": """
Filtering keeps the strategy exactly as it is and simply refuses to trade
it in regimes where it historically lost money. Of the four adaptive
mechanisms, it is the one that most reliably holds up out-of-sample, and
it should be the first thing you try.

**Why it is the safest of the four.** Filtering only ever *removes*
exposure. It never invents a new position, never picks a different rule,
never introduces a parameter. Its downside is therefore bounded: at worst
you remove the wrong days and give up some return. Compare that with
switching, which can put you in an actively wrong position, or
re-parameterizing, which hands you a fresh set of numbers to overfit.

**What it is really doing.** You are not predicting anything new. You are
acting on something you already knew — that this strategy is structurally
broken in this environment — and declining to pay for that knowledge
twice. Not trading is a legitimate position, and an underused one.

**The three things to check before believing it.**

1. **Exposure.** If filtering cuts you to 30% invested, every metric is now
   computed on 30% of the days. It may be excellent; it is definitely being
   measured on a smaller sample. See [[exposure_caveat]].
2. **Was the allow-list learned honestly?** "Sit out the regimes where it
   lost" is trivially profitable if you looked at the whole history to
   decide which those were. That is a description of the past, not a
   strategy. Everything here learns from the first 60% only, and
   `describe_filter()` shows you the evidence.
3. **Is the regime gap real?** If the best and worst regime Sharpes differ
   by less than their combined standard error, you are filtering on noise.
   The Regimes page computes that comparison for you.

**The failure mode that looks like a bug and isn't.** If the base strategy
had a negative Sharpe in *every* regime during the learning window, the
allow-list comes back empty and the strategy stays flat forever, returning
exactly 0%. That is the correct output: it means "on this data, there was
no market condition in which this strategy worked." A framework that
quietly traded anyway would be the broken one.
""",
    },
    "adaptive_switching": {
        "title": "Switching: the intuitive one that usually disappoints",
        "body": """
Switching runs a different strategy in each regime — trend-following when
the market trends, mean-reversion when it ranges. It is the most
appealing idea in the whole adaptive section and the one that most often
fails to deliver.

**The theory is genuinely sound.** Trend and mean-reversion profit from
opposite market behaviours (see [[trend_vs_chop]]). If you could reliably
tell which environment you were in, you could harvest both instead of
picking one and suffering through the other half of the time.

**Three costs eat the gains, and they compound.**

1. **Labels arrive late.** Smoothing deliberately delays every regime
   change by several days — that is the price of not being whipsawed. But
   the first days of a new regime are frequently where the largest move
   happens. You systematically miss the best part and arrive for the rest.
2. **Switching flips the entire position.** Going from a long trend signal
   to a flat reversion signal is a full round trip, not a marginal
   adjustment. Do that at every regime change and turnover climbs sharply.
   Run it with `cost_bps` at 5 or 10 before believing any result.
3. **You now need two things right, not one.** The regime label must be
   correct *and* the per-regime strategy choice must be stable. Each is
   uncertain; the errors multiply rather than cancel.

**The selection problem underneath it.** Choosing "the best strategy for
this regime" from three candidates, in each of three regimes, is nine
comparisons on a single price history. With that many comparisons some
regime will show a winner by chance. Read the evidence table: if the
winner beat the runner-up by 0.05 of Sharpe over 80 days, that is a coin
flip wearing a decision's clothes.

**How to evaluate it fairly.** Compare it against filtering alone, and
against plain volatility targeting. If the simpler mechanism captured most
of the benefit, say so and prefer it — that judgement is worth more than
the extra complexity.
""",
    },
    "adaptive_overfitting": {
        "title": "How adaptive logic overfits (especially re-parameterizing)",
        "body": """
Every adaptive mechanism buys flexibility, and flexibility is exactly what
lets a model fit noise. The order below is roughly the order of danger.

**Re-parameterizing is the worst offender.** One strategy with
regime-specific settings sounds modest. Count the degrees of freedom:
three regimes times two window parameters is **six numbers fitted to one
price history**, up from two. The in-sample equity curve will improve
almost by construction. That improvement is not evidence of anything.

**Regime-conditional models split your data while keeping the parameter
count.** With 2,500 rows, a 70% train split and three regimes, each
per-regime model sees roughly 580 rows to fit a dozen features on. You
have not made the model smarter; you have made each copy of it hungrier
and fed it a third as much. This is why `ml_regime_conditional` typically
shows the worst in-sample-to-out-of-sample decay of anything here.

**Auto-selection is a search, and searches find noise.** Picking the best
of three candidates in each of three regimes is nine comparisons. Even on
pure noise, the maximum of nine draws looks good. Reporting that maximum
without accounting for the search is the strategy-mining version of
p-hacking.

**The compounding problem.** Stack switching on top of sizing on top of
regime detection and you have stacked their assumptions too. Each layer
was fitted, each has its own error, and the composite is far less robust
than the sum of its parts appears.

**Four defences, all cheap.**

- **Attribute before you combine.** Run each mechanism alone first. If
  volatility targeting explains the whole improvement, the switching layer
  is adding complexity and turnover for nothing.
- **Prefer mechanisms with fewer knobs.** Filtering adds an allow-list.
  Re-parameterizing adds a parameter per regime. That difference is the
  whole story.
- **Read the evidence table, not just the choice.** A decision made on a
  0.05 Sharpe difference over 80 days will not repeat.
- **Count every comparison you ran**, including the ones you discarded.
  They all spent statistical power whether or not you report them.
""",
    },
    "volatility_targeting": {
        "title": "Volatility targeting: the mechanism to beat",
        "body": """
Volatility targeting keeps the entry signal exactly as it is and scales
the *size* of the position so that expected volatility stays near a target:

```
position = signal x clip(target_vol / trailing_realized_vol, 0, max_leverage)
```

When realized volatility doubles, the position halves. That is the whole
idea, and it uses **no regime model at all** — which is precisely why it
is the benchmark every regime-based mechanism on this page has to beat.

**Why it works when regime switching often doesn't.** It leans on the one
market property that is genuinely forecastable. Volatility clusters:
today's realized volatility is a good predictor of tomorrow's. Direction
is not (see [[regime_volatility_clusters]]). Targeting converts the
forecastable quantity directly into a position size, with no intermediate
classification step that can be wrong.

**What it does to the return profile.** On most equity data, returns are
roughly preserved while maximum drawdown falls noticeably. The reason is
timing: the position was *already small* when the crash arrived, rather
than being cut afterwards at the worst prices. Compare that with a
stop-loss, which acts after the damage.

**Four things to check.**

1. **Exposure.** Targeting 15% on an asset that realizes 20% leaves you
   persistently under a full position, so absolute return falls even as
   Sharpe improves. Set the target near the asset's own realized
   volatility to see the mechanism rather than the cap.
2. **Turnover.** Continuous resizing trades every single day.
   `regime_sized` is the discrete cousin that only resizes at regime
   boundaries — far cheaper, and worth comparing directly.
3. **The `max_leverage` cap.** Default 1.0 keeps output in [0, 1], so the
   strategy is never more exposed than the unscaled version. Raising it
   above 1.0 means borrowing, with everything that implies.
4. **Causality.** The trailing volatility window ends at day *t* and the
   whole signal is shifted forward a day in the backtest, so nothing here
   sees the future.

**The finding to be ready for.** Plain volatility targeting frequently
beats every regime-based mechanism in this project. If that is what your
data says, that is the result — report it. Preferring the simpler
explanation of the same outcome is what good quant judgement looks like,
and [[position_sizing]] makes the broader version of the argument.
""",
    },
    "exposure_caveat": {
        "title": "Exposure: the number that reframes every other number",
        "body": """
Exposure is the average absolute position — the share of the period you
actually held risk. It is the most commonly ignored column on a results
table, and it silently changes the meaning of everything beside it.

**Two identical Sharpe ratios, two different results.** A Sharpe of 1.0 at
100% exposure and a Sharpe of 1.0 at 20% exposure are not the same
achievement. The second was earned on a fifth of the days, so it rests on
a fifth of the evidence and carries a much wider error bar. It may be the
better strategy; it is certainly the less well-established one.

**Why adaptive strategies make this urgent.** Filtering, regime gating and
volatility targeting all *reduce* exposure by design. That is the
mechanism working. But it means the improved drawdown you are admiring
may be partly — or entirely — the trivial consequence of holding less:

> Any strategy can halve its drawdown by halving its position. That is not
> skill, it is arithmetic.

**How to tell skill from arithmetic.** Compare risk-*adjusted* metrics,
not raw drawdown. If Sharpe improved, the strategy was out of the market
at the *right* times. If only drawdown improved while Sharpe fell, it was
simply out of the market — you could have achieved the same by trading the
original at half size, with far less machinery.

**The capital question nobody asks.** At 30% exposure, 70% of your capital
sat idle. Real money needs somewhere to be. Either the return on *total*
capital is much lower than the headline, or you need a second use for that
capital — and that second use has its own risk.

**Rules of thumb.** Below ~40% exposure, always report it alongside the
headline metrics. Below ~25%, treat the metrics as provisional: they rest
on a small sample of invested days. At exactly 0%, the strategy has
concluded there was no condition in which it worked, which is a real
answer rather than an error.
""",
    },
    "turnover_costs": {
        "title": "Turnover: what adaptation actually costs you",
        "body": """
Turnover is total position change per year in full-position units.
Turnover of 20 means you replaced the entire book twenty times. It is the
price tag attached to every adaptive mechanism, and it is usually left off
the comparison.

**The arithmetic.** Annual drag is roughly `turnover x cost_bps / 10,000`.

| Turnover | at 5bps | at 10bps |
|---|---|---|
| 1x | 0.05% | 0.10% |
| 10x | 0.50% | 1.00% |
| 50x | 2.50% | 5.00% |

Against a strategy with a 4% expected return, 50x turnover at 10bps
consumes more than the entire edge.

**Why this matters more on this page than anywhere else.** Adaptive
strategies buy their improved risk profile *with extra trading*, and the
mechanisms differ enormously in how much:

- *Filtering* — trades only at regime boundaries. Cheap.
- *Regime sizing* — resizes only at boundaries. Cheap.
- *Volatility targeting* — resizes every day, but in small increments.
  Moderate.
- *Switching* — flips the entire position at every regime change.
  Expensive.
- *Regime-conditional ML* — trades on most days. Very expensive.

A comparison run at 0bps systematically flatters the bottom of that list.
Turn costs on and the ranking frequently reverses — which means the
frictionless ranking was not merely optimistic, it was **wrong**.

**The question to ask of any adaptive result.** Did the extra turnover buy
anything? If the adaptation added 10x turnover and improved Sharpe by
0.03, it did not. Reporting that plainly — "the added complexity did not
pay for itself" — is a complete and useful finding.

**The break-even habit.** For any strategy, compute the cost level at
which it stops making money. If the answer is 3bps, it is a theoretical
object that will not survive contact with a broker. If it survives 20bps,
it deserves more of your time. See [[costs]] for the broader argument.
""",
    },
    "ml_base_rate": {
        "title": "The base rate: the bar every model has to clear",
        "body": """
Before you can say a classifier is any good, you need to know what "no
skill" scores. That number is the **base rate**: the accuracy you'd get by
always predicting the majority class, with no model at all.

**On daily equity data it is about 53%.** Roughly 53% of days are up,
because equity indices drift upward over time. So a model that predicts
"up" every single day — a constant, a single line of code, no features,
no training — scores about 53%.

**Which reframes almost every accuracy number you'll see.**

| Test accuracy | What it actually means |
|---|---|
| 48% | Worse than a coin flip. The model learned anti-signal. |
| 51% | Below the base rate. Worse than a constant. |
| 53% | Exactly the base rate. The model added **nothing**. |
| 55% | +2 points of real edge. Genuinely notable on daily data. |
| 60% | Extraordinary. Go looking for the lookahead bug first. |

"My model is 54% accurate!" sounds like a result and is, at best, one
percentage point of edge. Reported without the base rate beside it, it is
not a claim you can evaluate at all — which is exactly why this dashboard
shows them side by side and computes the difference for you.

**Why the base rate isn't a fixed 53%.** It depends on the window. A test
period that happened to contain a bull run has a higher base rate; one
containing a crash has a lower one, and may even flip so the majority class
is "down". Always compute it on the *same* period you're evaluating, never
from memory.

**The trap this creates.** A model can maximize accuracy by simply
learning to predict the majority class always. It will look mediocre-but-
respectable on the accuracy metric and be completely useless — see the
confusion matrix, where it shows up as an entire column of zeros. That
failure is common enough that this page checks for it explicitly.

**And the deeper point.** Even genuine edge over the base rate does not
mean profit. See [[ml_accuracy_vs_pnl]]: which days you get right matters
far more than how many.
""",
    },
    "ml_accuracy_vs_pnl": {
        "title": "Why accuracy is not profit",
        "body": """
Accuracy counts predictions. Markets pay in magnitude. Those are different
things, and conflating them is one of the most common errors in applied ML
for trading.

**The core asymmetry.** A model that is right on 60 small days and wrong
on 40 large ones has 60% accuracy and loses money. Accuracy weights every
day equally; your P&L weights each day by how far the market moved. Since
daily returns are fat-tailed — a handful of days carry most of the year's
move — being right on the *typical* day is close to irrelevant.

**A worked example.** 100 test days:

- 60 days right, average move +0.3% → **+18%**
- 40 days wrong, average move −0.7% → **−28%**
- Accuracy: **60%**. Return: **−10%**.

Nothing is wrong with the model's classification. It simply got the days
that mattered wrong, and no accuracy figure can reveal that.

**It runs the other way too.** A 48% accurate model can be profitable if
its wins are systematically larger than its losses. Trend-following lives
here: it is wrong most of the time and makes its money on a few large
moves. Judged on accuracy it looks broken; judged on P&L it works.

**Then costs finish the job.** A daily direction model trades constantly —
turnover of 30-80x a year is normal. At 5bps that is 1.5-4% of annual
drag. A model with one point of genuine edge over the base rate does not
generate enough to pay for its own trading.

**So what should you read?** Out-of-sample **Sharpe**, and turnover beside
it. Accuracy is a diagnostic for whether the model learned anything at
all; it is not a measure of whether the strategy makes money. This page
deliberately shows both, in that order, so the gap between them is visible.

**The habit.** When someone quotes a model's accuracy, ask two questions:
*what was the base rate* (see [[ml_base_rate]]) and *what was the Sharpe
after costs*. The second question ends most conversations.
""",
    },
    "ml_regime_conditional": {
        "title": "Conditioning a model on regime: two ways, both with a catch",
        "body": """
If the relationship between today's features and tomorrow's return genuinely
differs across market environments, a model that knows which environment it
is in should do better. That is a reasonable hypothesis. This page lets you
test it two ways.

**Mode "feature" — one model, regime as an input.** The regime label is
one-hot encoded into the feature matrix. The model *can* learn "in a crisis,
ignore momentum" if the data supports it. It costs a handful of extra
parameters and keeps every training row. This is the conservative option and
usually the right first try.

**Mode "conditional" — a separate model per regime.** Each model is fitted
only on days in its own regime. Maximum flexibility, and a direct route to
overfitting:

> With 2,500 rows, a 70% train split and three regimes, each per-regime model
> sees roughly **580 rows** to fit a dozen features on — while the parameter
> count per model stays exactly the same.

You have not made the model smarter. You have made three copies of it and
fed each a third of the data. This is why the regime-conditional variant
typically shows the worst in-sample-to-out-of-sample decay of anything in
this project — see [[adaptive_overfitting]].

**The guard rail, and what it tells you.** Regimes with fewer than 150
training rows fall back to a model fitted on everything. The `Own model?`
column says which regimes got their own. If most fell back, conditional
mode is barely doing anything — and that is worth knowing before you
attribute any difference to it.

**How to read the per-regime accuracy table.** The only column that matters
is **edge over base rate**, because each regime has its own base rate. A
crisis regime might be 45% up-days, so 48% accuracy there is *positive*
edge, while 52% in a calm regime with a 56% base rate is negative. Raw
accuracy across regimes is not comparable.

**What a real finding looks like.** Consistent positive edge in one regime
across enough test days to matter, ideally with a mechanism you can state.
What you usually get instead is a scatter of small positive and negative
edges with no pattern — which is the honest answer that there was nothing
to condition on, and more useful than a marginal improvement you couldn't
trust.
""",
    },
    "ml_feature_importance": {
        "title": "Feature importance tells you what, never whether",
        "body": """
Feature importance answers one question: **what did the model lean on?** It
does not answer whether leaning on it was correct, and reading it as
evidence of a real relationship is a standard mistake.

**An overfit model produces confident importances for pure noise.** If a
random forest memorized the training rows, it memorized them *using* some
features, and those features get high importance scores. The scores are an
accurate description of the model's internals and tell you nothing about the
market. Always read importance next to the train/test gap — high importances
from a model with a 20-point gap describe a fantasy.

**The two charts here mean different things.**

- *Logistic regression* shows **coefficients**: signed, so direction is
  meaningful. A positive coefficient on `rsi_14` means higher RSI pushed the
  prediction toward "up". Because the features are standardized first, the
  magnitudes are comparable to each other.
- *Random forest* shows **impurity-based importance**: unsigned, so you get
  "this feature mattered" with no direction. It is also biased toward
  high-cardinality continuous features — which is nearly all of them here, so
  treat small differences in rank as meaningless.

**Three things worth checking.**

1. **Concentration.** If one feature holds most of the importance, the model
   is close to a single-variable rule. That is not automatically bad — simple
   is good — but it means the other eleven features are decoration.
2. **Which horizon dominates.** `return_1d` and `return_2d` are the noisiest
   inputs available: single-day moves on a liquid index are close to random.
   A model leaning hardest on those is leaning on noise, whatever the score
   says.
3. **Stability.** Change the train fraction from 0.7 to 0.6 and look again.
   If the ranking reshuffles, the importances are describing sampling
   variation, not structure. Stable rankings across splits are the minimum
   bar before you take any of it seriously.

**What importance can't do.** It is not causal, it is not a hedge ratio, and
it does not transfer to a different model class. A feature with high
importance in the forest may have a near-zero coefficient in the logistic
model on identical data. Both are correct descriptions of different models.
""",
    },
    "walkforward_reason": {
        "title": "Why a backtest on fixed history is not evidence",
        "body": """
Before walk-forward makes sense as a *solution*, it's worth being precise
about the problem — because the problem is not "backtests can be buggy".
It is structural, and it applies to a perfectly correct backtest.

**The setup that guarantees a good result.** You have a strategy with
parameters. You have one finite price history. Some setting of those
parameters was, necessarily, the best one on that history. Find it and
report its performance, and you have reported the maximum of a search —
not the expected performance of a process.

This is true even if you never ran an explicit optimizer. Choosing 50 and
200 for your moving averages because "those are conventional" is still a
choice informed by decades of other people searching the same history.

**Why the number is biased upward, always.** Suppose you test 20 parameter
sets on pure noise. Their true edge is zero. Their measured Sharpes will
scatter around zero, and the best of the 20 will be meaningfully positive
by construction. Report that one and you have reported a selection
artifact. The more you tried, the more inflated it is — and the count
includes every variant you tried and discarded.

**The three searches that all count.**

1. *Parameter search* — trying many windows, keeping the best.
2. *Strategy search* — trying many strategies, keeping the best.
3. *Silent search* — looking at a chart, disliking it, changing something.
   This one leaves no record and is the hardest to account for. See
   [[silent_fitting]].

**Why "but it's ten years of data" doesn't rescue it.** Ten years of daily
data is 2,500 rows but nothing like 2,500 independent observations.
Returns are autocorrelated in volatility, regimes persist for months, and
the whole sample contains perhaps three or four genuinely distinct market
environments. Your effective sample size is far smaller than the row count
suggests, and your confidence should shrink accordingly.

**What walk-forward actually buys you.** It doesn't make a strategy good.
It gives you one number that the search process could not have touched —
and one honest number is worth more than a dozen flattering ones. See
[[is_vs_oos]] for how to read it, and [[fold_uncertainty]] for why one
such number still isn't enough.
""",
    },
    "is_vs_oos": {
        "title": "In-sample and out-of-sample: reading the two numbers",
        "body": """
**In-sample (IS)** is the period your strategy could have been influenced
by — where parameters were chosen, models fitted, rules learned, and where
you looked at charts while deciding what to build.

**Out-of-sample (OOS)** is data that played no part in any of that.

The split here is **chronological**, never random. Shuffling rows and
holding out 30% at random would leak badly: neighbouring days share
volatility regimes and overlapping rolling windows, so a "held-out"
Tuesday sitting between two training days is not held out in any
meaningful sense.

**The gap is the metric, not either level.**

| Pattern | Reading |
|---|---|
| IS 1.2 → OOS 1.0 | Small gap. The process generalizes. This is the good outcome. |
| IS 1.8 → OOS 0.2 | Collapse. The IS number was mostly fitting. |
| IS 1.8 → OOS −0.4 | Sign flip. Whatever was learned was actively wrong on new data. |
| IS 0.3 → OOS 0.4 | No gap, but nothing to generalize either. Honest and unremarkable. |

A **small gap with a modest level** beats a **large level with a large
gap** every time. The first describes a process you can repeat; the second
describes one lucky period.

**Four things that make the comparison lie.**

1. **Trade count.** Two trades OOS is two coin flips. Check `num_trades`
   before reading any OOS Sharpe — this is the caveat people skip most.
2. **Regime mix.** If the OOS period contained more of the environment
   your strategy dislikes, per-regime performance may be perfectly intact
   while the blended number falls. That's a changed market, not a broken
   strategy — the regime-mix table settles which.
3. **Drawdown.** A strategy that keeps its return but doubles its drawdown
   OOS has still degraded. Don't read Sharpe alone.
4. **Repeated looks.** See [[silent_fitting]]. Looking, adjusting, and
   looking again converts OOS into IS silently.

**One split is one draw.** Even a clean gap on a single split might be
luck about *where the split landed*. That is what rolling walk-forward
addresses — see [[fold_uncertainty]].
""",
    },
    "fold_uncertainty": {
        "title": "Why one split isn't enough, and what folds tell you",
        "body": """
A single 70/30 split produces exactly **one** out-of-sample number. One
number cannot distinguish "this works" from "this happened to be tested on
a friendly stretch of market".

Rolling walk-forward fixes that by producing many consecutive out-of-sample
blocks. Now you have a *distribution*, and a distribution answers questions
a point estimate cannot.

**Read the folds in three passes.**

1. **The share positive.** Ten of twelve folds positive is a far better bet
   than a high average driven by two enormous folds. Consistency is the
   property that repeats; magnitude often isn't.
2. **The spread.** The standard deviation across folds is your honest
   uncertainty about the strategy. If fold Sharpes range from −1.5 to +2.0,
   your "true" Sharpe is somewhere in a very wide band, and quoting a
   single figure to two decimal places is false precision.
3. **The sequence, not just the set.** This is the pass people skip. Folds
   arrive in time order. Consistently positive early and consistently
   negative later is not noise — it is [[regime_drift]], and averaging
   across it produces a number describing a market that no longer exists.

**Why fold Sharpes are so noisy.** Each fold is short. The standard error
of an annualized Sharpe is roughly `sqrt(252/N)`, so a 126-day fold carries
an error bar of about **±1.4**. Individual folds are almost uninformative
on their own; the value is entirely in the pattern across them.

**What a robust strategy looks like.** Most folds positive, the bad ones
survivable, no trend across the sequence, and a spread you could live
with. Notably, *not* a high average — a strategy averaging 0.4 across
twelve folds with ten positive is a better bet than one averaging 0.8 off
two spectacular folds and ten flat ones.

**The stitched equity curve** concatenates only the out-of-sample days into
one continuous record. It is the closest thing in this project to a paper
trading log, and its shape matters more than its endpoint: long flat
stretches are periods you would have had to sit through without knowing
they would end.

**One honest limitation here.** This implementation generates the signal
once and evaluates it in rolling blocks; it does not refit fitted
strategies per fold. That is stated in the returned `fitted_note`, and it
means the folds are slightly optimistic for `ml_direction` and the
auto-selecting wrappers.
""",
    },
    "silent_fitting": {
        "title": "Silent fitting: the leak with no code",
        "body": """
Every other bias in this project has a technical fix. This one doesn't,
and it is probably the most common way research goes wrong.

**The pattern.** You run walk-forward. The out-of-sample result is poor.
You go back, change a parameter — or the feature set, or the date range,
or the ticker — and run it again. The second result is better, so you keep
it and report that out-of-sample number.

**That number is no longer out-of-sample.** You used the holdout period to
make a decision. It became training data the moment it influenced you. Do
this three or four times and your carefully constructed holdout is
thoroughly in-sample, with none of the code showing it.

**Why it's so easy to do.** It doesn't feel like cheating. Each individual
step is reasonable — "that parameter was clearly badly chosen", "of course
I should exclude the COVID crash", "this ticker is more representative".
The bias comes from the *sequence* of decisions, each informed by the
result you were trying to validate, and no single step looks wrong.

**The same thing at team scale.** Twenty people each testing one strategy
on the same data is statistically identical to one person testing twenty.
The winner gets published; the nineteen failures leave no trace. This is
why published anomalies decay so reliably after publication.

**What actually helps — none of it technical.**

- **Write the decision rule down before you look.** "I will accept this if
  OOS Sharpe > 0.5 with at least 30 trades" is a commitment. Deciding what
  counts as success *after* seeing the number is not.
- **Budget your looks.** Treat the holdout as a scarce resource. Two or
  three evaluations, not thirty.
- **Keep a log of everything you tried**, including what you abandoned.
  Those attempts spent statistical power whether or not you report them.
- **Hold back a second period you have never touched** and check once, at
  the very end.
- **Prefer strategies you can state a reason for.** A hypothesis specified
  in advance costs far less statistical power than one found by search, and
  degrades more gracefully.

**The uncomfortable corollary.** If you have been iterating on this
dashboard for an afternoon, the out-of-sample numbers you are looking at
are already somewhat contaminated. That is not a reason to stop — it is a
reason to say so when you report them.
""",
    },
    "survivorship_bias": {
        "title": "Survivorship bias: the sample you never see",
        "body": """
Every dataset here is a record of things that still exist. What's missing
is everything that didn't make it, and its absence quietly inflates every
result computed on what remains.

**The classic form.** Backtest a strategy on today's S&P 500 members over
the last twenty years and you have selected companies that survived and
grew enough to still be in the index. Enron, Lehman, Wachovia, Bear
Stearns are not in your sample. You have unintentionally built a strategy
tested on the winners of a race whose result you already know.

**Where it hides in this project specifically.**

1. **The ticker.** Backtesting on SPY, 2010-2025, is backtesting on an
   asset you already know roughly tripled. Any long-biased strategy
   inherits that. Run the same strategy on something that went sideways or
   down before believing it — the choice of ticker is itself a decision
   informed by the future.
2. **The date range.** The default here starts in 2008, which includes a
   crisis. Start in 2010 instead and you exclude it, testing exclusively on
   one long low-volatility bull market. Both are defensible; only one is
   what you thought you were measuring.
3. **Fund and index data generally.** Fund databases drop closed funds,
   which is why the average reported fund return beats the average actual
   investor experience by a wide margin.
4. **Strategies you read about.** The moving-average crossover is famous
   because it worked on some history someone looked at. The thousands of
   rules that didn't were never written up. Every "classic" strategy you
   test here reached you through that filter.

**Why it interacts badly with everything else on this page.** Walk-forward
protects you from fitting *within* a sample. It does nothing about the
sample itself being selected. A strategy can pass every fold cleanly and
still be an artifact of having chosen a survivor to test it on.

**Partial defences.**

- Test on several tickers, including ones that performed badly.
- Test on several date ranges, including ones containing crises.
- Always compare against buy-and-hold on the same asset — if the asset
  tripled, your strategy needs to beat *that*, not zero.
- State the universe and period you selected, and why, as part of any
  result. Making the choice explicit is most of the remedy.
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
