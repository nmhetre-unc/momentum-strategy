# Quant Strategy Dashboard (Week 3: Interactive UI + Deployment)

A multi-strategy backtesting engine with risk-adjusted performance metrics,
walk-forward validation, and a live interactive dashboard.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage — CLI (unchanged from Week 2)

```bash
python main.py --ticker SPY --strategy sma_crossover
python main.py --ticker AAPL --strategy momentum --walk-forward
python main.py --ticker MSFT --strategy mean_reversion --plot
```

## Usage — Interactive dashboard (new)

```bash
streamlit run app.py
```

This opens a local dashboard in your browser. From there:
- Pick a ticker, date range, and strategy in the sidebar
- Adjust strategy parameters with sliders (window sizes, RSI thresholds, etc.)
- Click **Run Backtest** to see the equity curve, drawdown chart, and full metrics table
- Toggle walk-forward validation to see in-sample vs. out-of-sample performance side by side

## Deploying it live (free)

1. Push this repo to GitHub (see below if you haven't already)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select this repo, and set the main file path to `app.py`
4. Deploy — you'll get a public URL like `https://your-app-name.streamlit.app`

Put that live link at the top of your GitHub README and on your resume — this
is what turns the project from "code someone would have to run themselves"
into "something a recruiter can actually click and use."

## Structure

- `data_loader.py` — fetches and locally caches OHLCV data via yfinance
- `strategies.py` — strategy functions (`sma_crossover`, `momentum`, `mean_reversion_rsi`)
- `backtest.py` — turns a signal into an equity curve; shifts signals forward one day to avoid lookahead bias
- `analytics.py` — CAGR, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, trade count, win rate
- `walk_forward.py` — in-sample/out-of-sample split to check for overfitting
- `visualize.py` — static PNG charts (used by the CLI's `--plot` flag)
- `app.py` — the interactive Streamlit dashboard (new this week)
- `main.py` — CLI entry point
- `test_logic.py` — sanity-checks everything against synthetic data

## Notes on app.py

- `@st.cache_data` on `load_data()` means switching strategy or parameters
  without changing the ticker/dates re-uses cached price data instead of
  re-hitting the Yahoo Finance API.
- Streamlit reruns the entire script top-to-bottom on every widget interaction
  (every slider drag, every button click). Results are stashed in
  `st.session_state` specifically so that adjusting an unrelated widget
  (like toggling walk-forward) doesn't wipe the chart you just generated.
- The walk-forward section automatically flags a warning if the Sharpe ratio
  drops by more than 0.5 out-of-sample — a simple, honest heuristic for
  "this might be overfit," worth explaining in an interview if asked.

## Coming in Week 4

An ML-driven strategy (logistic regression or random forest predicting
next-day direction) added as a fourth option in the dashboard, tying back
to the crypto ML research project.
