"""
Usage:
    python main.py --ticker SPY --strategy sma_crossover
    python main.py --ticker AAPL --strategy momentum --start 2018-01-01 --end 2024-01-01
"""

import argparse

from data_loader import fetch_ohlcv
from strategies import STRATEGIES
from backtest import run_backtest, summary_stats


def main():
    parser = argparse.ArgumentParser(description="Run a backtest for a given ticker and strategy.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. SPY")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES.keys())
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    args = parser.parse_args()

    df = fetch_ohlcv(args.ticker, args.start, args.end)
    signal = STRATEGIES[args.strategy](df)
    result = run_backtest(df, signal)
    stats = summary_stats(result)

    print(f"\nStrategy: {args.strategy} | Ticker: {args.ticker} | {args.start} to {args.end}")
    print(f"Total return:      {stats['total_return']:.2%}")
    print(f"Benchmark (B&H):   {stats['benchmark_return']:.2%}")
    print(f"Number of trades:  {stats['num_trades']}")
    print(f"Win rate:          {stats['win_rate']:.2%}")


if __name__ == "__main__":
    main()
