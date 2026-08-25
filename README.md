# Quant Strategy Dashboard

A multi-strategy backtesting engine — rule-based and ML-driven — with
risk-adjusted performance metrics, walk-forward out-of-sample validation,
and a live interactive dashboard.

**Live demo:** _add your Streamlit Community Cloud URL here_

## Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage — CLI

```bash
python main.py --ticker SPY --strategy sma_crossover
python main.py --ticker AAPL --strategy momentum --walk-forward
python main.py --ticker MSFT --strategy mean_reversion --plot
python main.py --ticker SPY --strategy ml_direction --walk-forward --model-report
```

## Usage — Interactive dashboard

```bash
streamlit run app.py
```

Pick a ticker, date range, and strategy in the sidebar (including the ML
strategy, with a model-type toggle and train/test split slider), then
click **Run Backtest**. Toggle walk-forward validation to see in-sample
vs. out-of-sample performance side by side.

## Deploying it live (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. **New app** → select this repo → main file path: `app.py` → Deploy
4. Put the resulting URL at the top of this README and on your resume

## Strategies

- **`sma_crossover`** — long when the short-term moving average is above the long-term average (trend-following)
- **`momentum`** — long when the trailing N-day return is positive (trend-following)
- **`mean_reversion`** — RSI-based; long when oversold, flat when overbought, holds position between thresholds (mean-reversion)
- **`ml_direction`** — a classifier (logistic regression or random forest) trained on lagged returns, moving-average ratios, RSI, volatility, and volume features to predict next-day direction

## Structure

- `data_loader.py` — fetches and locally caches OHLCV data via yfinance
- `strategies.py` — all four strategy functions, registered in one `STRATEGIES` dict
- `backtest.py` — turns a signal into an equity curve; shifts signals forward one day to avoid lookahead bias
- `analytics.py` — CAGR, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, trade count, win rate
- `walk_forward.py` — in-sample/out-of-sample split to check for overfitting
- `features.py` — feature engineering for the ML strategy (lagged returns, moving-average ratios, RSI, volatility, volume)
- `ml_strategy.py` — trains the classifier, converts predictions to the same long/flat signal interface, and reports model diagnostics (accuracy, feature importance)
- `visualize.py` — static PNG charts (used by the CLI's `--plot` flag)
- `app.py` — the interactive Streamlit dashboard
- `main.py` — CLI entry point
- `test_logic.py` — sanity-checks everything against synthetic data

## On the ML strategy specifically — read this before trusting its numbers

Adding a trained model introduces a sharper version of the overfitting
risk the walk-forward split already exists to catch. Concretely, in
testing on synthetic random-walk data (no real signal to find):

```
logistic        train_acc=58.0%  test_acc=48.6%
random_forest   train_acc=86.2%  test_acc=42.3%
```

The random forest's 86% train accuracy collapsing to 42% out-of-sample
(worse than a coin flip) is a textbook overfitting signature — it
memorized noise in the training period rather than learning anything
that generalizes. This is genuinely useful, not an embarrassing result:
**it's exactly the kind of honest finding worth describing directly in
an interview** — "I built an ML layer, and my own walk-forward validation
caught it overfitting, so here's what that tells me about model
complexity vs. generalization." That's a stronger signal of quant/ML
maturity than a suspiciously good backtest would have been.

Only the out-of-sample numbers (from `--walk-forward` or the dashboard's
walk-forward section) should ever be quoted as this strategy's
performance — the full-period and in-sample numbers are optimistic by
construction.

## Notes

- `test_logic.py` uses synthetic random-walk data so the logic can be verified without
  network access to Yahoo Finance. Run `main.py` or `app.py` directly on your own machine for real data.
- The ML strategy trains once on the first `train_frac` (default 70%) of history and predicts
  on the rest — it does not retrain incrementally as new data arrives.
