"""
Sanity-checks strategies.py, backtest.py, analytics.py, and walk_forward.py
against synthetic price data, since this sandbox can't reach the Yahoo
Finance API. Run main.py directly on your own machine for real data.
"""
import numpy as np
import pandas as pd

from strategies import STRATEGIES
from backtest import run_backtest
from analytics import full_report
from walk_forward import evaluate_out_of_sample

np.random.seed(42)
n = 600
dates = pd.date_range("2022-01-01", periods=n, freq="B")
returns = np.random.normal(0.0004, 0.012, n)
prices = 100 * (1 + pd.Series(returns)).cumprod()
df = pd.DataFrame({"Close": prices.values}, index=dates)

print("=== Full-period backtests ===")
for name, fn in STRATEGIES.items():
    signal = fn(df)
    assert signal.isin([0, 1]).all(), f"{name} produced a non-binary signal"
    assert not signal.isna().any(), f"{name} produced a NaN in the signal"

    result = run_backtest(df, signal)
    stats = full_report(result)

    assert -1.0 <= stats["max_drawdown"] <= 0.0, f"{name} max_drawdown out of range: {stats['max_drawdown']}"
    assert 0.0 <= stats["win_rate"] <= 1.0, f"{name} win_rate out of range: {stats['win_rate']}"

    print(f"{name:15s} return={stats['total_return']:7.2%}  cagr={stats['cagr']:7.2%}  "
          f"sharpe={stats['sharpe_ratio']:6.2f}  sortino={stats['sortino_ratio']:6.2f}  "
          f"max_dd={stats['max_drawdown']:7.2%}  trades={stats['num_trades']:4d}")

print("\n=== Walk-forward validation (70/30 split) ===")
for name, fn in STRATEGIES.items():
    wf = evaluate_out_of_sample(df, fn, split_frac=0.7)
    in_s, out_s = wf["in_sample"], wf["out_sample"]
    print(f"{name:15s} split={wf['split_date']}  "
          f"in-sample sharpe={in_s['sharpe_ratio']:6.2f}  "
          f"out-of-sample sharpe={out_s['sharpe_ratio']:6.2f}")

# Explicitly test the RSI zero-average-loss edge case that was fixed:
# force a monotonically increasing price series so avg_loss == 0 for a
# whole stretch, and confirm the strategy doesn't crash or produce NaN.
straight_up = pd.DataFrame({"Close": np.linspace(100, 200, 60)}, index=pd.date_range("2023-01-01", periods=60, freq="B"))
rsi_signal = STRATEGIES["mean_reversion"](straight_up)
assert not rsi_signal.isna().any(), "RSI signal produced NaN on the zero-avg-loss edge case"
print("\nRSI zero-average-loss edge case handled without NaN or crash.")

print("\nAll checks passed on synthetic data.")
