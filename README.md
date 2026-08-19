# Quant Strategy Dashboard (Week 2: Analytics & Rigor)

A multi-strategy backtesting engine with risk-adjusted performance metrics
and walk-forward out-of-sample validation.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
python main.py --ticker SPY --strategy sma_crossover
python main.py --ticker AAPL --strategy momentum --walk-forward
python main.py --ticker MSFT --strategy mean_reversion --plot
python main.py --ticker SPY --strategy sma_crossover --walk-forward --plot
```

## Structure

- `data_loader.py` — fetches and locally caches OHLCV data via yfinance
- `strategies.py` — strategy functions (`sma_crossover`, `momentum`, `mean_reversion_rsi`), all sharing the interface `fn(df, **params) -> pd.Series` (1 = long, 0 = flat)
- `backtest.py` — turns a signal into an equity curve; shifts signals forward one day to avoid lookahead bias
- `analytics.py` — CAGR, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, trade count, win rate
- `walk_forward.py` — splits a backtest chronologically into in-sample/out-of-sample periods and reports both, to check whether performance is real or just overfit to one lucky stretch
- `visualize.py` — saves equity curve and drawdown charts as PNGs
- `main.py` — CLI entry point
- `test_logic.py` — sanity-checks everything against synthetic data (useful in any environment without live market data access)

## Reading the walk-forward output

If a strategy's Sharpe ratio is strong in-sample but collapses (or goes
negative) out-of-sample, that's a sign the strategy was fit to noise in
the earlier period rather than capturing something that generalizes.
This is worth mentioning explicitly in an interview — it shows you're
checking for overfitting, not just reporting the best-looking number.

## Fixed since Week 1

`mean_reversion_rsi` previously used `avg_loss.replace(0, np.nan)` to
avoid a divide-by-zero warning when computing RSI. That silently turned
a legitimate case (zero average loss over the window = maximally
overbought) into `NaN` instead of the mathematically correct `RSI = 100`.
Now handled explicitly instead of papering over it.

## Notes

- `test_logic.py` uses synthetic random-walk data so the logic can be verified without
  network access to Yahoo Finance. Run `main.py` directly on your own machine for real data.
- `--plot` requires matplotlib (already in requirements.txt) and saves PNGs to the current directory.

## Coming in Week 3

An interactive Streamlit UI on top of this — deployed live, so the
dashboard is something a recruiter can actually click and use, not just
code they'd have to run themselves.
