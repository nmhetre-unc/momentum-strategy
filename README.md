# Quant Strategy Dashboard (Week 1: Foundation)

Backtesting engine supporting multiple trading strategies through a shared interface.
This is Week 1 of a 4-week build — see the full project plan for what's coming
(performance analytics, an interactive Streamlit UI, and an ML-driven strategy).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

```bash
python main.py --ticker SPY --strategy sma_crossover
python main.py --ticker AAPL --strategy momentum --start 2018-01-01 --end 2024-01-01
python main.py --ticker MSFT --strategy mean_reversion
```

## Structure

- `data_loader.py` — fetches and locally caches OHLCV data via yfinance
- `strategies.py` — strategy functions, all sharing the interface `fn(df, **params) -> pd.Series` (1 = long, 0 = flat)
- `backtest.py` — turns any strategy's signal into an equity curve and summary stats (returns, trade count, win rate)
- `main.py` — CLI entry point
- `test_logic.py` — sanity-checks all strategies against synthetic data (useful in any environment without live market data access)

## Adding a new strategy

Write a function with the signature `(df: pd.DataFrame, **params) -> pd.Series` returning
1/0 values, then add it to the `STRATEGIES` dict in `strategies.py`. The backtest
engine and CLI pick it up automatically — no other changes needed.

## Notes

- The backtest shifts signals forward by one day to avoid lookahead bias — you can only
  act on a signal the day after it fires.
- `test_logic.py` uses synthetic random-walk data so the logic can be verified without
  network access to Yahoo Finance. Run `main.py` directly on your own machine for real data.
# SPY Momentum Trading Strategy

A simple moving average crossover strategy backtested on SPY (S&P 500 ETF) from 2020-2024.

## Strategy Logic

This project implements a classic momentum trading strategy using two moving averages:
- **50-day SMA** — captures short-term price momentum
- **200-day SMA** — captures long-term trend

**Buy signal:** 50-day SMA crosses above the 200-day SMA (short-term momentum stronger than long-term trend)
**Sell signal:** 50-day SMA crosses below the 200-day SMA (short-term momentum weaker than long-term trend)

## Results (2020-2024)

| Metric | Buy & Hold SPY | Momentum Strategy |
|--------|---------------|-------------------|
| Total Return | 55.81% | 53.10% |
| Sharpe Ratio | 0.61 | 0.76 |

The strategy generated slightly lower total returns but a meaningfully higher Sharpe Ratio, meaning it delivered more return per unit of risk taken. It also avoided much of the volatility during the COVID-19 crash in early 2020.

## Key Implementation Detail

Signals are shifted forward by one day (`signal.shift(1)`) before calculating strategy returns to avoid look-ahead bias — ensuring the backtest only acts on information that would have actually been available at the time.

## Tech Stack

- Python
- pandas
- NumPy
- Matplotlib
- yfinance

## What I'd Improve Next

- Test the strategy on multiple assets, not just SPY
- Add transaction costs to make the backtest more realistic
- Try different moving average windows to see if performance improves
- Add a stop-loss mechanism for risk management
