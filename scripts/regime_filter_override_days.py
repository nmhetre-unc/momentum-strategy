"""
FINDINGS.md section 4's numbers: on how many days does regime_filtered
differ from its base strategy, and why?

regime_filtered forces the base strategy flat (a) in regimes its learned
allow-list excludes and (b) on UNKNOWN days, the regime detector's
warm-up before it has enough history to label anything. Since version 2
it is also flat (c) before its first out-of-sample day, so it is only
scored from that day on, and only those days are counted here as a
regime effect. Before version 2, (b) was the whole difference on SPY:
the base strategy wanted a position before the detector produced its
first label, a start-date effect inside what is now the in-sample
period.

A base that holds no position in the allow-list's learning window
(ml_direction, which trades only after its own out-of-sample split)
makes regime_filtered raise; the error is recorded instead of counts.

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
from qbt.timesplit import out_of_sample_start

TICKER, START, END = "SPY", "2008-01-01", "2025-01-01"
BASES = ("sma_crossover", "momentum", "mean_reversion", "ml_direction")
OUT = Path(__file__).resolve().parents[1] / "results" / "regime_filter_override_days.json"


def main() -> None:
    df = fetch_ohlcv(TICKER, START, END)
    labels = _resolve_regimes(df).labels.reindex(df.index).fillna(UNKNOWN)
    learn_end = _learn_cutoff(df, 0.6)  # regime_filtered's default learn_frac
    unknown = labels == UNKNOWN
    rows: dict[str, dict] = {}
    for base in BASES:
        signal = _base_signal(df, base, None).reindex(df.index).fillna(0.0)
        try:
            raw = regime_filtered(df, base=base)
        except ValueError as exc:
            rows[base] = {"raises": str(exc)}
            continue
        start = out_of_sample_start(raw)
        assert start is not None
        filtered = raw.reindex(df.index).fillna(0.0)
        scored = df.index >= start
        differs = (filtered != signal) & scored
        n_unknown, n_excluded = int((differs & unknown).sum()), int((differs & ~unknown).sum())
        table = _regime_sharpe_table(df, labels, [base], learn_end)
        rows[base] = {
            # The rule regime_filtered applies (allowed_regimes=None).
            "learned_allow_list": [int(r["regime"]) for _, r in table.iterrows()
                                   if pd.isna(r["sharpe"]) or r["sharpe"] > 0],
            "first_out_of_sample_day": start.date().isoformat(),
            "out_of_sample_days": int(scored.sum()),
            "out_of_sample_days_differing_from_base": int(differs.sum()),
            "of_which_unknown_warm_up": n_unknown,
            "of_which_excluded_regime": n_excluded,
            # Before version 2 these were the whole difference (FINDINGS section 4).
            "in_sample_unknown_days_base_held_a_position":
                int(((signal != 0) & unknown & ~scored).sum()),
        }
    out = {
        "data": {"ticker": TICKER, "start": START, "end": END, "n_trading_days": len(df)},
        "learn_end": learn_end.date().isoformat(),
        "unknown_days": int(unknown.sum()),
        "bases": rows,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
