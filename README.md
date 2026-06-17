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