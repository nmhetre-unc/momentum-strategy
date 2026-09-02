# Quant Training Terminal

A multi-strategy backtesting engine — rule-based, ML-driven, and regime-adaptive — wrapped
inside a training platform for incoming quant traders and interns. It includes risk-adjusted
metrics, walk-forward validation, market regime detection, adaptive strategies, guided
exercises, and a dashboard that explains itself as you use it.

**Live demo:** https://momentum-strategy-zpgmmb89pfwjghigjk8soc.streamlit.app/

The goal is **exposure to the work, not production trading**. All results use free daily data
and a simple cost model. What transfers is the reasoning: how to form a hypothesis, how to
test it honestly, and how to recognize when you've fooled yourself.

Every metric has a tooltip explaining what it means *and what it hides*. Every chart has a
plain-language reading guide. Around ninety contextual warnings fire automatically when your
own numbers cross a threshold worth knowing about — too few trades, non-causal regime labels,
accuracy below the base rate, a Sharpe ratio high enough to suspect a leak.

---

## Quant training overview

Most people learn backtesting by building something that makes money on historical data and
then wondering why it doesn't work. This platform teaches the opposite habit:

> **Assume your result is wrong until you've tried to break it.**

The dashboard teaches eight core skills, each demonstrated directly on your own data:

| Concept | Where to learn it |
|---|---|
| Strategy design — what each rule bets on | Backtest |
| ML modeling — and why it overfits | ML lab |
| Walk-forward validation — single and rolling | Validation |
| Market regime detection | Regimes |
| Adaptive strategies — four mechanisms | Adaptive |
| Risk analysis by regime | Regimes → performance by regime |
| Reading equity curves and drawdowns | Backtest + quant notes |
| Fair strategy comparison | Validation → compare |

---

## How to think like a quant

Six habits that prevent most mistakes:

1. **Always name the bet.** If you can't summarize the strategy's claim in one sentence,
   you're searching, not researching — and search finds noise.
2. **Ask what date each input was knowable.** This single question catches almost every
   lookahead bug.
3. **Separate process from result.** Walk-forward evaluates the process; the in-sample to
   out-of-sample gap tells the truth.
4. **Count your degrees of freedom.** Every parameter, every discarded idea, every regime
   split spends statistical power — whether or not you report it.
5. **Prefer the boring explanation.** If volatility targeting explains the improvement,
   say so, even when the regime-switching model is more interesting to talk about.
6. **Report the limitation first.** Honest limitations make your work stronger, not weaker.

---

## Common pitfalls

### Lookahead bias

Using information that wasn't available at the time. Examples:

- Acting on a signal on the same close it was computed from
- Ranking features against full-sample statistics instead of expanding ones
- Fitting a regime model on all history and then backtesting on it
- Centered smoothing filters on labels or signals

**Tell:** Sharpe above 2 on daily data — assume a leak until you've found otherwise.

The Regimes page shows full-sample and walk-forward labels side by side, so you can measure
the gap on your own data.

### Overfitting

Fitting noise and calling it signal.

**Tell:** a large in-sample to out-of-sample gap, and results that change character under
small parameter changes.

On synthetic random-walk data:

```
                train_acc   test_acc   base rate
logistic          57.7%       48.0%      54.9%
random_forest     86.2%       45.1%      54.9%
```

The random forest memorizes noise. The simpler model generalizes better — which runs against
most people's instinct to reach for more capacity when results disappoint.

### Regime drift

Markets change; relationships stop being real.

**Tell:** early walk-forward folds positive, later folds negative.

**Fix:** read the fold *sequence*, not just the average.

---

## How to use this platform

### Setup

Requires **Python 3.12+** (numpy 2.5 needs ≥3.12; pandas 3 and scikit-learn 1.9 need ≥3.11).
Developed and tested on 3.14.

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run app.py
```

Ten pages share one data selection and one regime model:

- **Start here** — roadmap, beginner traps, first-session checklist
- **Backtest** — equity curve, drawdown, walk-forward, reading guide
- **Regimes** — five detection methods, persistence checks, lookahead demo
- **Adaptive** — four mechanisms, learned rules, per-regime attribution
- **ML lab** — train/test gap, base rate, regime breakdown
- **Validation** — rolling walk-forward, regime-attributed decay, fair comparison
- **Exercises** — ten guided exercises with automated checks
- **Learn** — learning path, pitfalls, 35 quant notes, glossary
- **Graduation checklist** — 29 abilities, grouped by page
- **Next steps** — first research project, extensions, specialization paths

### Deploy it (Streamlit Community Cloud)

1. Push to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → **Create app** → deploy from GitHub,
   main file path `app.py`
3. **Open "Advanced settings" and set the Python version to 3.12 or newer.** The default is
   3.9, which cannot install `streamlit>=1.62` at all — pip errors out and the build dies
   before your code runs. The version can only be set at creation, so an existing app must be
   deleted and recreated to change it.

If the build succeeds but the app shows a data error, that is Yahoo Finance blocking cloud IP
ranges rather than a bug. The repo ships cached CSVs for SPY and AAPL.

---

## CLI

```bash
python main.py --ticker SPY --strategy sma_crossover
python main.py --ticker AAPL --strategy momentum --walk-forward
python main.py --ticker SPY --strategy ml_direction --walk-forward --model-report
python main.py --ticker SPY --strategy adaptive_ensemble --regimes --cost-bps 5
python main.py --ticker SPY --strategy regime_filtered --regimes --regime-walk-forward
python main.py --ticker SPY --strategy sma_crossover --rolling
python main.py --ticker SPY --exercises
```

## Tests

```bash
python test_logic.py     # base engine
python test_regime.py    # regimes, adaptive strategies, validation layer
```

Both run offline on synthetic data. `test_regime.py` builds a series with *known* regime
structure — the only place the detection machinery can be checked against ground truth, since
on real data you never have it.

---

## The teaching layer

Four devices make the platform self-explanatory:

| Device | Purpose |
|---|---|
| **Tooltips** | Explain every metric and what it conceals |
| **Contextual caveats** | ~90 warnings computed from your own numbers |
| **Reading guides** | "How to interpret this" beside every chart and table |
| **Quant notes** | 35 collapsible explanations, surfaced in context |

Consistency is structural rather than a matter of discipline: every page draws its
breadcrumb, "why this matters", common mistakes and closing links from one registry
(`page_guide.py`) through one set of helpers (`regime_dashboard.py`), so no page can drift
into a different voice or shape.

**Principles:**

- Negative results are results
- Sample size before conclusions
- Prefer the boring explanation

---

## Suggested learning path

Each stage's "done when" is a habit, not a completed task:

1. **Baselines** — memorize SPY's long-term stats
2. **Rule-based strategies** — predict the winner before running it
3. **Validation** — instinctively ask "in-sample or out-of-sample?"
4. **The ML layer** — explain why the simpler model wins
5. **Regimes** — attribute returns by environment
6. **Lookahead demo** — measure the gap yourself
7. **Adaptive strategies** — prefer the simplest mechanism that explains the result
8. **Write-up** — lead with the limitation

---

## Strategies

### Base strategies

Binary long/flat, in `strategies.STRATEGIES`:

- **`sma_crossover`** — trend-following
- **`momentum`** — trend-following
- **`mean_reversion`** — RSI-based
- **`ml_direction`** — logistic regression or random forest

### Adaptive strategies

Fractional positions in [0, 1], in `adaptive.ADAPTIVE_STRATEGIES`:

| Strategy | Mechanism | The catch |
|---|---|---|
| `regime_filtered` | Filtering | Cuts exposure |
| `regime_switch` | Switching | Label lag plus costs |
| `regime_parameters` | Re-parameterizing | Overfits per regime |
| `volatility_targeted` | Sizing | Often wins without regimes at all |
| `regime_sized` | Sizing per regime | Compare against the continuous version |
| `adaptive_ensemble` | Ensemble | Attribute the components first |
| `ml_regime_conditional` | Conditional ML | Splits your training data |

Every automatic choice — which regimes to allow, which strategy to run where — is learned
**only** from the first 60% of history, and `describe_choices()` / `describe_filter()` expose
both the decision and the evidence behind it.

---

## Regime detection

Five methods:

| Method | Description |
|---|---|
| `rules` | Baseline. No fitting, so nothing can leak. |
| `kmeans` | Hard clustering. No notion of persistence. |
| `gmm` | Soft clustering with per-day probabilities. |
| `hmm` | Gaussian HMM with persistence, implemented in-repo — no extra dependency. |
| `supervised` | Forward-defined regimes learned as a classifier. |

Regimes are always renumbered so **0 = calmest**, ordered by realized volatility, which keeps
the IDs comparable across refits, methods and tickers. Smoothing is **causal only** — a
centered filter would look tidier and be lookahead bias.

**Three questions to ask of any labelling:**

1. Do episodes last weeks, not days?
2. Is the transition matrix diagonal above 0.95?
3. Do the regimes actually differ in return or volatility?

A no to any of them means there is nothing there to condition on.

---

## Structure

**Core engine:**

- `data_loader.py` — fetches and locally caches OHLCV data via yfinance
- `strategies.py` — the four base strategies, plus docs and parameter specs
- `backtest.py` — signal → equity curve; forward-shifts signals; optional `cost_bps`
- `analytics.py` — CAGR, volatility, Sharpe, Sortino, drawdown, exposure, turnover, per-regime
- `walk_forward.py` — single split, rolling folds, regime-attributed decay, fair comparison
- `features.py` / `ml_strategy.py` — ML features and the classifier
- `visualize.py` — static PNG charts for the CLI's `--plot`
- `main.py` — CLI

**Regime layer:**

- `regime_features.py` — point-in-time environment features
- `regime.py` — five detection methods, a self-contained Gaussian HMM, causal smoothing
- `adaptive.py` — the regime-aware wrappers and `ALL_STRATEGIES`

**Training layer:**

- `quant_notes.py` — 35 quant notes, metric tooltips, pitfalls, learning path (plain data)
- `page_guide.py` — per-page orientation content, so pages are consistent by construction
- `exercises.py` — ten exercises with automated checks; runnable headless via `run_all(df)`
- `regime_dashboard.py` — Altair charts, the validated palette, and the teaching widgets
- `app.py` + `app_pages/` — the ten-page dashboard

**Tests:** `test_logic.py`, `test_regime.py`

---

## Notes and limitations

- The ML strategy trains once on the first `train_frac` — it does not retrain incrementally
- `rolling_walk_forward()` evaluates in rolling folds but does **not** refit fitted strategies
  per fold; the returned `fitted_note` says so
- Transaction costs are modeled as a constant basis-point charge on position change; real
  costs grow with size and vary with liquidity
- Regime labels are estimates, and their uncertainty is largest exactly at transitions — which
  is when acting on them matters most
- The dashboard caps regimes at 4: more over-segments a decade of daily data, and the ordinal
  color ramp stops being distinguishable past four steps
