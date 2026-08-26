"""Detect market regimes, check whether they're real, and see where a strategy's return came from."""

import pandas as pd
import streamlit as st

from analytics import benchmark_by_regime, performance_by_regime
from backtest import run_backtest
from regime import (
    REGIME_METHOD_DOCS, SMOOTHING_DOCS, detect_regimes, regime_episodes,
    regime_stability, regime_summary,
)
from regime_dashboard import (
    PERFORMANCE_CONFIG, REGIME_SUMMARY_CONFIG, caveat, duration_histogram,
    performance_by_regime_chart, quant_note, regime_feature_chart,
    regime_probability_chart, regime_ribbon_chart, require_regimes,
    show_regime_health, transition_heatmap,
)
from regime_features import FEATURE_DOCS
from strategies import STRATEGIES, STRATEGY_DOCS

df, regimes = require_regimes()
settings = st.session_state["regime_settings"]

st.caption(REGIME_METHOD_DOCS[settings["method"]])
st.caption(f"**Smoothing — {settings['smooth']}:** {SMOOTHING_DOCS[settings['smooth']]}")

stability = regime_stability(regimes.labels)
show_regime_health(regimes, stability)

# ---------- The ribbon ----------
st.subheader("Regimes over time", divider="gray")
st.altair_chart(regime_ribbon_chart(df, regimes))
st.caption(
    "Check the labels against your own reading of the chart. If the model calls a crash calm, "
    "you have learned something about the model rather than about the crash."
)
quant_note("regimes")

probability_chart = regime_probability_chart(regimes)
if probability_chart is not None:
    with st.expander("Model confidence over time", icon=":material/percent:"):
        st.altair_chart(probability_chart)
        st.markdown(
            "Where the bands are cleanly separated the model is confident. Where they interleave "
            "it is guessing — and those days are transitions, which is exactly when a "
            "regime-switching strategy acts."
        )

# ---------- Is there anything to condition on? ----------
st.subheader("What the asset did in each regime", divider="gray")
summary = regime_summary(regimes, df)
st.dataframe(
    summary.drop(columns=["regime"]), hide_index=True,
    column_config=REGIME_SUMMARY_CONFIG, key="regime_summary",
)
if not summary.empty:
    spread = summary["ann_volatility"].max() - summary["ann_volatility"].min()
    if spread < 0.05:
        caveat(
            f"Annualized volatility differs by only {spread:.1%} across these regimes. They are "
            "barely distinguishable, which means there is nothing here to condition a strategy on — "
            "no amount of downstream cleverness creates a difference that isn't in the data."
        )

# ---------- Transitions ----------
st.subheader("Transitions and persistence", divider="gray")
transition_left, transition_right = st.columns([1, 1])
with transition_left:
    matrix = regimes.transition_matrix()
    heatmap = transition_heatmap(matrix)
    if heatmap is not None:
        st.altair_chart(heatmap)
    st.caption(
        "Read the diagonal first: those are the persistence probabilities. On daily data they "
        "should be 0.95+. A diagonal near 1/k means the model found noise, not regimes."
    )
with transition_right:
    episodes = regime_episodes(regimes.labels, regimes.names)
    histogram = duration_histogram(episodes, regimes.names)
    if histogram is not None:
        st.altair_chart(histogram)
    if not matrix.empty:
        expected = pd.DataFrame({
            "Regime": matrix.index,
            "Expected duration (days)": [
                (1 / (1 - matrix.iloc[i, i])) if matrix.iloc[i, i] < 1 else float("inf")
                for i in range(len(matrix))
            ],
        })
        st.dataframe(
            expected, hide_index=True, key="expected_duration",
            column_config={"Expected duration (days)": st.column_config.NumberColumn(format="%.0f")},
        )
        st.caption("Expected duration is 1/(1 − p_stay), straight off the diagonal above.")

# ---------- Where did the return come from? ----------
st.subheader("Strategy performance by regime", divider="gray")
strategy_name = st.selectbox(
    "Strategy", list(STRATEGIES), key="regime_strategy",
    help="Runs the strategy unchanged, then splits its daily P&L by the regime in force that day.",
)
st.caption(f"**Hypothesis to test:** {STRATEGY_DOCS[strategy_name]['regime_hint']}")

signal = STRATEGIES[strategy_name](df)
result = run_backtest(df, signal, cost_bps=5, regimes=regimes.labels)
table = performance_by_regime(result, regimes.labels, regimes.names)

if table.empty:
    st.info("No labelled days to attribute performance to yet.", icon=":material/info:")
else:
    metric_choice = st.segmented_control(
        "Metric", ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"],
        default="sharpe_ratio", key="regime_metric",
    ) or "sharpe_ratio"
    st.altair_chart(performance_by_regime_chart(table, metric_choice, regimes.names))

    st.dataframe(
        table.drop(columns=["regime"]), hide_index=True,
        column_config=PERFORMANCE_CONFIG, key="regime_perf",
    )

    with st.expander("Compare against buy-and-hold in the same regimes", icon=":material/balance:"):
        st.markdown(
            "A long-only strategy looks good in any regime where the asset rose. This table "
            "separates *the strategy worked* from *the market went up*."
        )
        st.dataframe(
            benchmark_by_regime(result, regimes.labels, regimes.names).drop(columns=["regime"]),
            hide_index=True, column_config=PERFORMANCE_CONFIG, key="regime_bench",
        )

    thin = table[table["days"] < 200]
    if not thin.empty:
        caveat(
            "Regimes with under 200 days here: "
            + ", ".join(f"**{row['name']}** ({int(row['days'])}d)" for _, row in thin.iterrows())
            + ". Standard error on an annualized Sharpe is roughly sqrt(252/days), so those rows "
              "carry error bars wide enough to contain almost any conclusion.",
            level="info",
        )

quant_note("risk_by_regime")

# ---------- Features ----------
st.subheader("What the model is looking at", divider="gray")
available = [c for c in regimes.features.columns if c in FEATURE_DOCS]
feature = st.selectbox("Regime feature", available, key="regime_feature")
st.caption(FEATURE_DOCS[feature])
feature_chart = regime_feature_chart(regimes, feature)
if feature_chart is not None:
    st.altair_chart(feature_chart)

# ---------- The lookahead demonstration ----------
st.subheader("The lookahead demonstration", divider="gray")
quant_note("regime_lookahead", expanded=True)
st.markdown(
    "Below, the same detection method fitted on the **full sample** versus refitted on an "
    "**expanding window**. Both label the same days. Only one of them could have existed at the time."
)

if st.button("Compare honest and leaky labels", icon=":material/compare_arrows:", key="lookahead_run"):
    with st.spinner("Fitting both versions..."):
        leaky = detect_regimes(
            df, method=settings["method"], n_regimes=settings["n_regimes"], fit_frac=1.0,
            smooth=settings["smooth"], min_duration=settings["min_duration"],
        )
        try:
            from regime import detect_regimes_walk_forward

            honest = detect_regimes_walk_forward(
                df, method=settings["method"], n_regimes=settings["n_regimes"],
                smooth=settings["smooth"], min_duration=settings["min_duration"],
            )
        except ValueError as exc:
            honest = None
            st.warning(str(exc), icon=":material/warning:")

    leaky_column, honest_column = st.columns(2)
    with leaky_column:
        st.markdown("**Full-sample fit** — not tradeable")
        st.altair_chart(regime_ribbon_chart(df, leaky, height=240))
    if honest is not None:
        with honest_column:
            st.markdown("**Expanding-window refit** — tradeable")
            st.altair_chart(regime_ribbon_chart(df, honest, height=240))

        agreement = (leaky.labels == honest.labels)[honest.valid()].mean()
        st.metric(
            "Label agreement on days both versions labelled", f"{agreement:.1%}",
            help="Every disagreement is a day where knowing the future changed the answer.",
        )
        st.caption(
            f"The honest version produced no labels at all for its first "
            f"{int((~honest.valid()).sum()):,} days, and refit "
            f"{honest.meta.get('n_refits', 0)} times. Exercise 'Measure the regime lookahead bias' "
            f"turns this comparison into a number you can quote."
        )
