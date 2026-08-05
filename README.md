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
