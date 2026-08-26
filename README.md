# Quant Training Terminal

A multi-strategy backtesting engine — rule-based, ML-driven, and
regime-adaptive — wrapped in a training platform for incoming quant
traders and interns. Risk-adjusted metrics, walk-forward validation,
market regime detection, adaptive strategies, guided exercises, and a
dashboard that explains itself.

**Live demo:** _add your Streamlit Community Cloud URL here_

The goal here is **exposure to the work, not production trading**. Every
number in this repo is computed on free daily data with a simple cost
model. What transfers is the reasoning: how to build a hypothesis, how to
test it honestly, and how to recognize when you have fooled yourself.

---

## Quant training overview

Most people learn backtesting by building something that makes money on
historical data and then wondering why it doesn't work. This platform is
built around the opposite habit — **assume your result is wrong until you
have tried to break it** — and gives you the tools to try.

It teaches eight things, each with a place in the dashboard where you can
see it happen rather than read about it:

| What | Where |
|---|---|
| Strategy design — what each rule bets on | Backtest page |
| ML modeling, and why it overfits here | ML lab |
| Walk-forward validation, single and rolling | Validation page |
| Market regime detection | Regimes page |
| Adaptive strategies and their four mechanisms | Adaptive page |
| Risk analysis, and how it changes by regime | Regimes → performance by regime |
| Interpreting equity curves and drawdowns | Backtest page, quant notes |
| Comparing strategies fairly | Validation → compare |

---

## How to think like a quant

Six habits, in rough order of how much trouble each one saves you.

**1. Always name the bet.** Every strategy is a claim about market
behaviour. SMA crossover claims trends persist. RSI mean-reversion claims
short moves overshoot. If you can't say the claim in one sentence, you
are not researching, you are searching — and search finds noise.

**2. Ask what date each input was knowable.** This single question
catches almost every lookahead bug. Feature computed at today's close?
Fine. Ranked against the full sample's distribution? Not fine — that
distribution didn't exist yet.

**3. Separate the process from the result.** A good process can produce
a bad result and a bad process a good one. Walk-forward evaluates the
process. The gap between in-sample and out-of-sample tells you more than
either number.

**4. Count your degrees of freedom.** Every parameter you tuned, every
strategy you tried and discarded, every regime you conditioned on —
they all spend statistical power, whether or not you report them. A
strategy chosen from twenty candidates needs to clear a much higher bar
than one you specified in advance.

**5. Prefer the boring explanation.** If volatility targeting explains
your improvement, say so, even though the regime-switching HMM is more
interesting to talk about. Simpler explanations of the same result are
worth more, and they survive longer.

**6. Report the limitation first.** "I built an ML layer and my own
walk-forward caught it overfitting" is a stronger claim than any good
backtest, because a good backtest invites the question of what you did
wrong to get it.

---

## Common pitfalls

### Lookahead bias

Using information in a decision that wasn't available when the decision
was made. Where it hides in a project like this:

- Acting on a signal on the same close it was computed from — the reason
  `backtest.py` shifts every signal forward one day.
- Standardizing or ranking features against full-sample statistics
  instead of expanding ones.
- **Fitting a regime model on all of history and then backtesting on it.**
  The cluster centers encode the future; your 2015 "low volatility"
  label was computed partly from 2020.
- Centered smoothing filters on labels or signals.

**The tell:** implausibly high Sharpe. Above 2 on a daily equity
strategy, assume a leak until you've found otherwise.

`regime.py` defends against this two ways — `fit_frac < 1.0` and
`detect_regimes_walk_forward()` — and the Regimes page will run both side
by side so you can measure the gap on your own data.

### Overfitting

Fitting the noise in your sample and calling it signal. It arrives
through parameter search, strategy search, model capacity, and
regime-conditional models that split your data while keeping the
parameter count.

**The tell:** a large in-sample/out-of-sample gap, and results that
change character under small parameter changes.

The random forest in this repo exists to demonstrate this. On synthetic
random-walk data:

```
                train_acc   test_acc   base rate
logistic          57.7%       48.0%      54.9%
random_forest     86.2%       45.1%      54.9%
```

86% collapsing to 45% — worse than a coin flip, and ten points below
the base rate — is a textbook memorization signature. **With this little signal, the less flexible
model is the better one.** That runs against most people's instinct,
which is to reach for a bigger model when results disappoint.

### Regime drift

The market changes and a real relationship stops being real. A strategy
fit on 2010–2019 was fit on one long low-volatility bull market.

**The tell:** rolling walk-forward folds that are consistently positive
early and consistently negative later — which a single 70/30 split cannot
show you.

**The fix:** `rolling_walk_forward()`, and reading the fold *sequence*
rather than its average.

---

## How to use this platform

### Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### The dashboard

```bash
streamlit run app.py
```

Seven pages, sharing one data selection and one regime model configured
in the sidebar:

- **Backtest** — one strategy, its equity curve, drawdown, position, and
  an unavoidable comparison against buy-and-hold. Walk-forward is always
  shown, not hidden behind a toggle.
- **Regimes** — detect regimes five ways, check whether they're actually
  regimes (persistence, duration, distinctness), see what the asset and
  your strategy did in each, and run the lookahead demonstration.
- **Adaptive** — the four adaptation mechanisms, each with its learned
  rule and the evidence behind it exposed.
- **ML lab** — train the classifier, watch the train/test gap, compare
  against the base rate, and break accuracy down by regime.
- **Validation** — rolling walk-forward, regime-attributed decay, and a
  fair side-by-side comparison of all eleven strategies.
- **Exercises** — ten guided exercises with automated checks against
  whatever data you have loaded.
- **Learn** — the learning path, the pitfalls, every quant note, and a
  glossary.

### The CLI

```bash
# the original commands all still work
python main.py --ticker SPY --strategy sma_crossover
python main.py --ticker AAPL --strategy momentum --walk-forward
python main.py --ticker SPY --strategy ml_direction --walk-forward --model-report

# regimes and adaptive strategies
python main.py --ticker SPY --start 2008-01-01 --strategy sma_crossover --regimes
python main.py --ticker SPY --strategy adaptive_ensemble --regimes --cost-bps 5
python main.py --ticker SPY --strategy regime_filtered --regimes --regime-walk-forward

# rolling validation and the exercise checks
python main.py --ticker SPY --strategy sma_crossover --rolling
python main.py --ticker SPY --exercises
```

### Tests

```bash
python test_logic.py     # base engine, synthetic data
python test_regime.py    # regimes, adaptive strategies, validation layer
```

Both run offline on synthetic data. `test_regime.py` builds a series with
*known* regime structure, which is the only place the detection machinery
can be checked against ground truth — on real data you never have it.

---

## Suggested learning path

Each stage's "done when" is a habit, not a completed task. The habit is
the part that transfers.

**1. Baselines.** Run buy-and-hold on SPY, 2010–2025. Memorize its CAGR,
Sharpe and max drawdown. *Done when* you compare against them without
being reminded.

**2. Rule-based strategies.** Run all three on the same data. Write one
sentence per strategy on what behaviour it needs. *Done when* you can
predict which will win on a given chart before running it.

**3. Validation.** Walk-forward everything, then run
`rolling_walk_forward()`. *Done when* you instinctively ask "in-sample or
out-of-sample?" about any number, including your own.

**4. The ML layer.** Train both models. Compare train, test, and base
rate. *Done when* you can explain why the more accurate model is the
worse one.

**5. Regimes.** Detect with `rules`, then `kmeans`, then `hmm`. Compare
transition matrices and durations. *Done when* you can point at a
per-regime table and say where a strategy's return came from.

**6. The lookahead demonstration.** Run the same adaptive strategy on
full-sample and walk-forward regime labels. *Done when* you can quote the
size of the gap you measured, on your data.

**7. Adaptive strategies.** Run each mechanism separately with costs on.
*Done when* you reach for the simplest mechanism that explains the
improvement — and notice when the simple one wins.

**8. Write it up.** One page on one strategy: hypothesis, method,
out-of-sample result, per-regime breakdown, what would falsify it, what
you'd do next. *Done when* the write-up leads with a limitation and is
more convincing for it.

---

## Strategies

**Base strategies** — binary long/flat, `strategies.STRATEGIES`:

- **`sma_crossover`** — long when the short MA is above the long MA (trend-following)
- **`momentum`** — long when the trailing N-day return clears a threshold (trend-following)
- **`mean_reversion`** — RSI-based; long when oversold, flat when overbought (mean-reversion)
- **`ml_direction`** — logistic regression or random forest predicting next-day direction

**Adaptive strategies** — fractional positions in [0, 1],
`adaptive.ADAPTIVE_STRATEGIES`. Four distinct mechanisms, which fail
differently:

| Strategy | Mechanism | The catch |
|---|---|---|
| `regime_filtered` | Filtering — sit out bad regimes | Cuts your exposure; check the day count |
| `regime_switch` | Switching — different strategy per regime | Label lag plus full position flips; needs costs on |
| `regime_parameters` | Re-parameterizing — faster settings in high vol | Every regime is a fresh set of parameters to overfit |
| `volatility_targeted` | Position sizing — scale to a volatility target | Uses **no regime model at all**, and often wins anyway |
| `regime_sized` | Position sizing — size per regime | Compare against the continuous version |
| `adaptive_ensemble` | All of the above | Attribution: run the pieces separately first |
| `ml_regime_conditional` | Regime-conditioned ML | Conditional mode splits your training data by regime |

Every `auto` choice — which regimes to allow, which strategy to run
where — is learned **only** from the first 60% of history, and
`describe_choices()` / `describe_filter()` expose both the decision and
the evidence behind it.

---

## Regime detection

`regime.py` labels each day with a market regime using one of five methods:

| Method | What it is |
|---|---|
| `rules` | Explicit volatility/trend thresholds. Fits nothing, so nothing can leak. The baseline every fitted method must beat. |
| `kmeans` | Hard clustering. Fast, but assumes round equal clusters and has no notion of persistence. |
| `gmm` | Soft clustering with per-day probabilities. Handles elongated clusters; still no persistence. |
| `hmm` | Gaussian HMM (implemented in-repo, no extra dependency). Models persistence via a transition matrix — the standard tool. |
| `supervised` | Regimes defined from *forward* 21-day return and vol on the training window, then learned as a classifier. |

Regime IDs are always renumbered so **0 is the calmest**, ordered by
realized volatility, which keeps the IDs comparable across refits,
methods and tickers. Names come from the cluster centroid ("Calm
Uptrend", "Turbulent", "Crisis / Selloff").

Smoothing is **causal only** — `min_duration` (confirmation filter),
`ema_prob`, and a trailing `median`. A centered filter would look tidier
on the chart and would be lookahead bias.

**Three questions to ask of any regime labelling**, all answered at the
top of the Regimes page: do episodes last weeks (not days)? Is the
transition matrix diagonal above 0.95? Do the regimes actually differ in
return or volatility? A no to any of them means there is nothing there to
condition on.

---

## Structure

**Core engine** (unchanged interfaces):

- `data_loader.py` — fetches and locally caches OHLCV data via yfinance
- `strategies.py` — the four base strategies, plus `STRATEGY_DOCS` and `PARAM_SPECS` metadata for the dashboard
- `backtest.py` — signal → equity curve; forward-shifts signals; optional `cost_bps` and regime passthrough
- `analytics.py` — CAGR, volatility, Sharpe, Sortino, max drawdown, win rate, exposure, turnover, and `performance_by_regime`
- `walk_forward.py` — single split, `rolling_walk_forward`, `evaluate_with_regimes`, `compare_strategies`
- `features.py` — ML features (lagged returns, MA ratios, RSI, volatility, volume)
- `ml_strategy.py` — the classifier, now with regime-as-feature and regime-conditional modes
- `visualize.py` — static PNG charts for the CLI's `--plot`
- `main.py` — CLI

**Regime layer** (new):

- `regime_features.py` — point-in-time environment features: volatility level/expansion/percentile, trend, efficiency ratio, autocorrelation, drawdown, downside share, Parkinson vol, liquidity
- `regime.py` — the five detection methods, a self-contained Gaussian HMM, causal smoothing, transition/episode analysis
- `adaptive.py` — the regime-aware strategy wrappers and `ALL_STRATEGIES`

**Training layer** (new):

- `quant_notes.py` — every tooltip, quant note, pitfall and learning-path stage, as plain data
- `exercises.py` — ten exercises with automated checks; also runnable headless via `run_all(df)`
- `regime_dashboard.py` — Altair chart builders, table configs, teaching widgets, cached loaders
- `app.py` + `app_pages/` — the seven-page dashboard

**Tests:**

- `test_logic.py` — the base engine on synthetic data
- `test_regime.py` — regimes, adaptive strategies and validation, including explicit point-in-time and causality checks

---

## On the ML strategy specifically — read this before trusting its numbers

Adding a trained model introduces a sharper version of the overfitting
risk the walk-forward split already exists to catch. On synthetic
random-walk data (no real signal to find):

```
                train_acc   test_acc   base rate
logistic          57.7%       48.0%      54.9%
random_forest     86.2%       45.1%      54.9%
```

The random forest's 86% train accuracy collapsing to 45% out-of-sample is
a textbook overfitting signature — it memorized noise in the training
period rather than learning anything that generalizes. This is genuinely
useful, not an embarrassing result: **it's exactly the kind of honest
finding worth describing directly in an interview**.

Two things the dashboard now adds to that story:

- **The base rate.** On daily equity data roughly 53% of days are up, so
  "always predict up" scores 53%. A model at 53% test accuracy has added
  nothing. Test accuracy is always shown against this bar.
- **Accuracy is not P&L.** A 55%-accurate model loses money if the 45% it
  misses land on the big-move days. Read the out-of-sample Sharpe.

Only the out-of-sample numbers should ever be quoted as this strategy's
performance — the full-period and in-sample numbers are optimistic by
construction.

---

## Notes and limitations

- `test_logic.py` and `test_regime.py` use synthetic data so the logic can
  be verified without network access. Run `main.py` or `app.py` for real data.
- The ML strategy trains once on the first `train_frac` of history — it does
  not retrain incrementally.
- `rolling_walk_forward()` evaluates in rolling out-of-sample blocks but does
  **not** refit fitted strategies per fold. Its returned `fitted_note` says so.
  A production framework would refit every fold.
- Transaction costs are modeled as a constant basis-point charge on position
  change. Real costs grow with size and vary with liquidity.
- Regime labels are estimates, and their uncertainty is largest exactly at
  transitions — which is when acting on them matters most. No smoothing
  parameter makes that go away.
- The dashboard caps regimes at 4. More over-segments a decade of daily data,
  and the ordinal color ramp stops being distinguishable past four steps.
