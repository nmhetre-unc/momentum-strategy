"""
Audit: adaptive_ensemble's deflated Sharpe against buy-and-hold, before
and after it stopped being scored on the days it learned from.

Writes results/ml/deflated_sharpe_audit.json, one section per run:

    python scripts/audit_deflated_sharpe.py --section after
    python scripts/audit_deflated_sharpe.py --section before --source "<what code was measured>"

The "before" section was produced by running this same script against
the pre-fix code (an export of commit 7b5bb84, the last commit before
the regime wrappers traded only out of sample) instead of the editable
install (the export's src/ on PYTHONPATH, with the editable install's
import finder removed from sys.meta_path, since it takes precedence over
PYTHONPATH), and with the repo's data_cache copied next to the export so
both runs read the same prices. The other section in the file is left as
it is.

Both runs measure the same way, through functions that exist in both
versions: the strategy's net returns at 5bps, scored from its first
out-of-sample day when the code marks one (qbt.timesplit; the pre-fix
adaptive_ensemble marks none, so it is scored over the full period, as it
was then), and the deflated Sharpe (qbt.stats.deflated_sharpe_ratio)
with buy-and-hold's Sharpe over the same window as the threshold, at the
trial counts scripts/regenerate_results.py reports. The "after" numbers
equal results/summary.md's "Deflated Sharpe by trial count" row; the
"before" number at 5 trials equals the DSR vs bench column of
results/summary.md at 7b5bb84 (0.549).
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from qbt.adaptive import ADAPTIVE_STRATEGIES
from qbt.analytics import TRADING_DAYS_PER_YEAR, sharpe_ratio
from qbt.backtest import run_backtest
from qbt.data import fetch_ohlcv
from qbt.stats import deflated_sharpe_ratio
from qbt.timesplit import out_of_sample_start

TICKER, START, END = "SPY", "2008-01-01", "2025-01-01"
COST_BPS = 5.0
STRATEGY = "adaptive_ensemble"
TRIAL_SWEEP = (1, 5, 11)
OUT = Path(__file__).resolve().parents[1] / "results" / "ml" / "deflated_sharpe_audit.json"


def measure(source: str) -> dict:
    df = fetch_ohlcv(TICKER, START, END)
    signal = ADAPTIVE_STRATEGIES[STRATEGY](df)
    start = out_of_sample_start(signal)
    net = run_backtest(df, signal, cost_bps=COST_BPS)
    bench = run_backtest(df, pd.Series(1.0, index=df.index), cost_bps=0)
    if start is not None:
        net, bench = net.loc[start:], bench.loc[start:]
    returns = net["strategy_return"].dropna()
    bench_returns = bench["strategy_return"].dropna()
    root = np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe, bench_sharpe = sharpe_ratio(returns), sharpe_ratio(bench_returns)
    skew_, kurt_ = float(skew(returns)), float(kurtosis(returns, fisher=False))
    return {
        "source": source,
        "strategy": STRATEGY,
        "window": {"first": returns.index[0].date().isoformat(),
                   "last": returns.index[-1].date().isoformat(), "days": len(returns),
                   "scored_from_first_out_of_sample_day": start is not None},
        "sharpe": float(sharpe), "buy_and_hold_sharpe_same_window": float(bench_sharpe),
        "dsr_vs_buy_and_hold": {
            str(n): deflated_sharpe_ratio(sharpe / root, n, skew_, kurt_, len(returns),
                                          sharpe_benchmark=bench_sharpe / root)
            for n in TRIAL_SWEEP},
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=OUT.parents[2]).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("before", "after"), required=True)
    parser.add_argument("--source", help="what code is being measured (default: git HEAD)")
    args = parser.parse_args()
    source = args.source or f"git {_git('rev-parse', '--short', 'HEAD')} + working tree"
    measured = measure(source)
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    report = existing | {"audit": __doc__.strip().splitlines()[0], args.section: measured}
    OUT.write_text(json.dumps({k: report[k] for k in ("audit", "before", "after")
                               if k in report}, indent=2) + "\n")
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()
