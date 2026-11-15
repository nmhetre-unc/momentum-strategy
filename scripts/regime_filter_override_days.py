"""
FINDINGS.md section 4's numbers: on how many days does regime_filtered
differ from its base strategy, and why?

regime_filtered forces the base strategy flat (a) in regimes its learned
allow-list excludes and (b) on UNKNOWN days, the regime detector's
warm-up before it has enough history to label anything. On SPY the
learned allow-list never excludes a regime, so every difference is (b):
the base strategy wanted a position before the detector produced its
first label. That count is a start-date effect, not a regime effect.

The exception is ml_direction (version 2 on): it holds no position before
its out-of-sample split (70% of history), so over the allow-list's
learning window (the first 60%) it has a Sharpe of 0 in every regime and
the "positive Sharpe" rule excludes them all. The filter is then flat
every day, which is a degenerate pairing, not a regime effect either.

Writes results/regime_filter_override_days.json.

Run with: python scripts/regime_filter_override_days.py
"""

import json
from pathlib import Path

import pandas as pd

from qbt.adaptive import (
    _base_signal,
    _learn_cutoff,
    _regime_sharpe_table,
    _resolve_regimes,
    regime_filtered,
)
from qbt.data import fetch_ohlcv
from qbt.regime import UNKNOWN

TICKER, START, END = "SPY", "2008-01-01", "2025-01-01"
BASES = ("sma_crossover", "momentum", "mean_reversion", "ml_direction")
OUT = Path(__file__).resolve().parents[1] / "results" / "regime_filter_override_days.json"


def main() -> None:
    df = fetch_ohlcv(TICKER, START, END)
    labels = _resolve_regimes(df).labels.reindex(df.index).fillna(UNKNOWN)
    learn_end = _learn_cutoff(df, 0.6)  # regime_filtered's default learn_frac
    rows = {}
    for base in BASES:
        signal = _base_signal(df, base, None).reindex(df.index).fillna(0.0)
        filtered = regime_filtered(df, base=base).reindex(df.index).fillna(0.0)
        differs = filtered != signal
        unknown = labels == UNKNOWN
        table = _regime_sharpe_table(df, labels, [base], learn_end)
        n_unknown, n_excluded = int((differs & unknown).sum()), int((differs & ~unknown).sum())
        assert int(differs.sum()) == n_unknown + n_excluded
        rows[base] = {
            # The rule regime_filtered applies (allowed_regimes=None).
            "learned_allow_list": [int(r["regime"]) for _, r in table.iterrows()
                                   if pd.isna(r["sharpe"]) or r["sharpe"] > 0],
            "base_days_in_position_in_learning_window":
                int((signal[signal.index <= learn_end] != 0).sum()),
            "days_differing_from_base": int(differs.sum()),
            "of_which_unknown_warm_up": n_unknown,
            "of_which_excluded_regime": n_excluded,
        }
    out = {
        "data": {"ticker": TICKER, "start": START, "end": END, "n_trading_days": len(df)},
        "learn_end": learn_end.date().isoformat(),
        "unknown_days": int((labels == UNKNOWN).sum()),
        "bases": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
