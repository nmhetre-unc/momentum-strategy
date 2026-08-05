"""
Sanity-check strategies.py and backtest.py against synthetic price data,
since this sandbox can't reach the Yahoo Finance API. 
"""
import numpy as np
import pandas as pd

from strategies import STRATEGIES
from backtest import run_backtest, summary_stats

np.random.seed(42)
n = 600
dates = pd.date_range("2022-01-01", periods=n, freq="B")
returns = np.random.normal(0.0004, 0.012, n)
prices = 100 * (1 + pd.Series(returns)).cumprod()
df = pd.DataFrame({"Close": prices.values}, index=dates)

for name, fn in STRATEGIES.items():
    signal = fn(df)
    assert signal.isin([0, 1]).all(), f"{name} produced a non-binary signal"
    result = run_backtest(df, signal)
    stats = summary_stats(result)
    print(f"{name:15s} total_return={stats['total_return']:.2%}  "
          f"benchmark={stats['benchmark_return']:.2%}  "
          f"trades={stats['num_trades']:4d}  win_rate={stats['win_rate']:.2%}")

print("\nAll strategies ran without errors on synthetic data.")
