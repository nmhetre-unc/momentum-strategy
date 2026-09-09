"""
Two canaries for lookahead bias: one on the backtest engine, one on the
regime detector. Both use signals/fits deliberately built from information
that would not have been knowable at the time, then check the harness
either exposes that (Sharpe blows up) or refuses to use it (labels don't
move when future data is appended).
"""

import numpy as np
import pandas as pd

from qbt.analytics import full_report
from qbt.backtest import run_backtest
from qbt.regime import detect_regimes


def test_lookahead_canary():
    # If this harness had a lookahead bug -- e.g. run_backtest forgot to
    # shift the signal forward before applying it -- both halves of this
    # test would read the same, because both signals would then be acting
    # on same-day information. The gap between them is the proof the shift
    # is doing its job.
    rng = np.random.default_rng(0)
    n = 1500
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    returns = rng.normal(0.0003, 0.011, n)
    close = pd.Series(100 * np.cumprod(1 + returns), index=dates)
    df = pd.DataFrame({"Close": close})

    # (a) Oracle signal: literally cannot be computed at time t without
    # knowing close[t+1]. run_backtest() shifts every signal forward by one
    # day before applying it, so this signal ends up perfectly timed to
    # capture tomorrow's return -- exactly what a real lookahead bug would
    # look like. A Sharpe this high on daily data is the canary: if the
    # measured strategies ever produced a number in this range, that would
    # be the signal to go looking for a leak, not a reason to celebrate.
    oracle_signal = (close.shift(-1) > close).astype(int)
    oracle_stats = full_report(run_backtest(df, oracle_signal, cost_bps=0))
    assert oracle_stats["sharpe_ratio"] > 8, (
        f"oracle signal Sharpe {oracle_stats['sharpe_ratio']:.2f} is too low to prove the "
        "harness would expose leakage if it existed"
    )

    # (b) Same-day signal: knowable at the close of day t (it only compares
    # close[t] to close[t-1]), so this is what a strategy computed "live" at
    # today's close would look like. run_backtest()'s shift(1) means this
    # signal only gets acted on tomorrow, one day after the move it's named
    # after -- so it should carry no real edge. This is the honest control:
    # if this came back high, shift(1) would not be doing what it claims.
    sameday_signal = (close > close.shift(1)).astype(int)
    sameday_stats = full_report(run_backtest(df, sameday_signal, cost_bps=0))
    assert abs(sameday_stats["sharpe_ratio"]) < 2, (
        f"same-day signal Sharpe {sameday_stats['sharpe_ratio']:.2f} is too high -- "
        "the backtest may be acting on a signal before it was knowable"
    )


def test_regime_fit_end_ignores_future_data():
    # detect_regimes(..., fit_end=...) is supposed to fit only on data up
    # to that date. If it silently used later rows too -- e.g. because
    # standardization or decoding ran over the whole series instead of
    # stopping at fit_end -- then appending more history after the original
    # end date would change the labels on days that already existed, even
    # though nothing about their own past changed. That's the leak this
    # test is built to catch.
    rng = np.random.default_rng(1)
    dates_through_2022 = pd.bdate_range("2018-01-01", "2022-01-01")
    returns = rng.normal(0.0003, 0.012, len(dates_through_2022))
    close_through_2022 = 100 * np.cumprod(1 + returns)
    df_through_2022 = pd.DataFrame({"Close": close_through_2022}, index=dates_through_2022)

    extra_dates = pd.bdate_range(dates_through_2022[-1] + pd.Timedelta(days=1), "2024-01-01")
    extra_returns = rng.normal(0.0003, 0.012, len(extra_dates))
    extra_close = close_through_2022[-1] * np.cumprod(1 + extra_returns)
    df_through_2024 = pd.concat([
        df_through_2022,
        pd.DataFrame({"Close": extra_close}, index=extra_dates),
    ])

    result_through_2022 = detect_regimes(df_through_2022, method="hmm", n_regimes=3, fit_end="2020-01-01")
    result_through_2024 = detect_regimes(df_through_2024, method="hmm", n_regimes=3, fit_end="2020-01-01")

    labels_on_original_index = result_through_2024.labels.loc[df_through_2022.index]
    assert result_through_2022.labels.equals(labels_on_original_index), (
        "labels on the original date range changed after appending future rows -- "
        "fit_end is not being honored"
    )
