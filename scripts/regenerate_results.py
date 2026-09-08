"""
Runs every strategy in STRATEGIES and ADAPTIVE_STRATEGIES on SPY, once at
cost_bps=0 (gross) and once at cost_bps=5 (net), and prints a single
comparison table: gross Sharpe, net Sharpe, net total return, max
drawdown, turnover, and trade count.

Uses the 17-year cached SPY file (2008-01-01 to 2025-01-01) rather than
the 10-year one: several adaptive wrappers only learn their rules from
the first 60-70% of history, and the longer window leaves them more to
learn from and a longer out-of-sample tail to be judged on.

Adaptive strategies are called with no regime arguments, so each detects
its own regimes with every default (hmm, 3 regimes, fit_frac=0.7) rather
than sharing one precomputed result -- this is a standalone report, not
part of the dashboard's shared-state flow.

Run with: python scripts/regenerate_results.py
"""

import pandas as pd

from qbt.adaptive import ADAPTIVE_STRATEGIES
from qbt.analytics import full_report
from qbt.backtest import run_backtest
from qbt.data import fetch_ohlcv
from qbt.strategies import STRATEGIES

TICKER = "SPY"
START, END = "2008-01-01", "2025-01-01"


def main():
    df = fetch_ohlcv(TICKER, START, END)

    # Buy-and-hold: full position throughout, no costs -- the bar every
    # strategy above it in the table has to clear. Gross and net columns
    # are identical here since there are no costs to charge.
    benchmark = full_report(run_backtest(df, pd.Series(1.0, index=df.index), cost_bps=0))
    rows = [{
        "strategy": "buy_and_hold",
        "gross_sharpe": benchmark["sharpe_ratio"],
        "net_sharpe": benchmark["sharpe_ratio"],
        "net_total_return": benchmark["total_return"],
        "max_drawdown": benchmark["max_drawdown"],
        "turnover": benchmark["turnover"],
        "num_trades": benchmark["num_trades"],
    }]
    for name, fn in {**STRATEGIES, **ADAPTIVE_STRATEGIES}.items():
        signal = fn(df)
        gross = full_report(run_backtest(df, signal, cost_bps=0))
        net = full_report(run_backtest(df, signal, cost_bps=5))
        rows.append({
            "strategy": name,
            "gross_sharpe": gross["sharpe_ratio"],
            "net_sharpe": net["sharpe_ratio"],
            "net_total_return": net["total_return"],
            "max_drawdown": net["max_drawdown"],
            "turnover": net["turnover"],
            "num_trades": net["num_trades"],
        })

    header = (
        f"{'strategy':22s} {'gross_sharpe':>12s} {'net_sharpe':>10s} "
        f"{'net_return':>10s} {'max_dd':>8s} {'turnover':>8s} {'trades':>6s}"
    )
    print(f"{TICKER} {START} to {END} (data_cache/{TICKER}_{START}_{END}.csv)")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['strategy']:22s} {row['gross_sharpe']:12.2f} {row['net_sharpe']:10.2f} "
            f"{row['net_total_return']:10.1%} {row['max_drawdown']:8.1%} "
            f"{row['turnover']:8.1f} {row['num_trades']:6d}"
        )


if __name__ == "__main__":
    main()
