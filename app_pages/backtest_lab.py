"""Backtest one strategy and read the result properly."""

import streamlit as st

from analytics import full_report
from backtest import run_backtest
from regime_dashboard import (
    caveat, drawdown_chart, equity_chart, metric_row, position_chart,
    quant_note, require_data, show_metric_table,
)
from strategies import PARAM_SPECS, STRATEGIES, STRATEGY_DOCS, default_params
from walk_forward import evaluate_out_of_sample

df = require_data()
ticker = st.session_state["ticker"]

# ---------- Controls ----------
with st.container(border=True):
    controls = st.columns([2, 1, 1])
    strategy_name = controls[0].selectbox(
        "Strategy", list(STRATEGIES), key="bt_strategy",
        help="Each one bets on a different market behaviour. The note below says which.",
    )
    cost_bps = controls[1].number_input(
        "Cost (bps)", 0.0, 50.0, 5.0, step=1.0, key="bt_cost",
        help=(
            "Charged on every unit of position change. 1-5bps is realistic for liquid ETFs. "
            "Leave it at 0 only if you want to know what a frictionless world would have paid."
        ),
    )
    log_scale = controls[2].toggle(
        "Log scale", value=False, key="bt_log",
        help="Steady compounding is a straight line in logs. On a linear axis it always looks like a hockey stick.",
    )

    params = default_params(strategy_name)
    specs = PARAM_SPECS.get(strategy_name, [])
    if specs:
        param_cols = st.columns(len(specs))
        for column, spec in zip(param_cols, specs):
            params[spec["name"]] = column.slider(
                spec["label"], spec["min"], spec["max"], spec["default"],
                step=spec["step"], help=spec["help"], key=f"bt_{strategy_name}_{spec['name']}",
            )
    if strategy_name == "ml_direction":
        params["model_type"] = st.segmented_control(
            "Model type", ["logistic", "random_forest"], default="logistic", key="bt_model_type",
        ) or "logistic"

# ---------- What this strategy is ----------
doc = STRATEGY_DOCS[strategy_name]
with st.expander(f"What {strategy_name} does — and when it fails", icon=":material/help:"):
    st.markdown(
        f"**Family:** {doc['family']}\n\n{doc['what']}\n\n"
        f"**Works when:** {doc['works_when']}\n\n"
        f"**Fails when:** {doc['fails_when']}\n\n"
        f"**Watch for:** {doc['watch_for']}\n\n"
        f"**Regime hypothesis to test:** {doc['regime_hint']}"
    )

# ---------- Run ----------
signal = STRATEGIES[strategy_name](df, **params)
result = run_backtest(df, signal, cost_bps=cost_bps)
stats = full_report(result)

benchmark = result.copy()
benchmark["strategy_return"] = result["daily_return"]
benchmark["position"] = 1.0
benchmark_stats = full_report(benchmark)

st.subheader(f"{ticker} · {strategy_name}", divider="gray")
metric_row(stats, ["total_return", "sharpe_ratio", "max_drawdown", "exposure"])

# The benchmark comparison, made unavoidable rather than optional.
beat_return = stats["total_return"] - benchmark_stats["total_return"]
beat_sharpe = stats["sharpe_ratio"] - benchmark_stats["sharpe_ratio"]
if beat_sharpe < 0:
    caveat(
        f"Buy-and-hold beat this strategy on risk-adjusted terms over the same period "
        f"(Sharpe {benchmark_stats['sharpe_ratio']:.2f} vs {stats['sharpe_ratio']:.2f}, "
        f"return {benchmark_stats['total_return']:.1%} vs {stats['total_return']:.1%}). "
        f"That is the most common outcome and the most commonly omitted comparison. If the "
        f"strategy is still worth having, the argument has to be about drawdown or about the "
        f"{stats['exposure']:.0%} exposure, and you have to make it explicitly."
    )
else:
    st.success(
        f"Beats buy-and-hold on Sharpe by {beat_sharpe:+.2f} ({beat_return:+.1%} on total return), "
        f"at {stats['exposure']:.0%} average exposure. Check whether it survives the Validation page.",
        icon=":material/check_circle:",
    )

st.altair_chart(equity_chart(result, log_scale=log_scale))
st.altair_chart(drawdown_chart(result))
quant_note("equity_curve")

detail_left, detail_right = st.columns([1, 1])
with detail_left:
    st.markdown("**Full metrics**")
    show_metric_table(stats, key="bt_metrics")
with detail_right:
    st.markdown("**Position over time**")
    st.altair_chart(position_chart(result))
    st.caption(
        f"Turnover {stats['turnover']:.1f}x a year at {cost_bps:.0f}bps costs roughly "
        f"{stats['turnover'] * cost_bps / 10_000:.2%} annually — "
        f"{abs(stats['turnover'] * cost_bps / 10_000 / stats['cagr']):.0%} of this strategy's CAGR."
        if stats["cagr"] else "This strategy's CAGR is zero, so the cost ratio is undefined."
    )
    quant_note("costs")

# ---------- Walk-forward, on by default ----------
st.subheader("Walk-forward validation", divider="gray")
st.caption(
    "The numbers above cover the whole period, including data any fitted component saw. "
    "These don't."
)
wf = evaluate_out_of_sample(df, STRATEGIES[strategy_name], **params)
decay = wf["in_sample"]["sharpe_ratio"] - wf["out_sample"]["sharpe_ratio"]

wf_left, wf_right = st.columns(2)
with wf_left:
    st.markdown(f"**In-sample** (to {wf['split_date']})")
    show_metric_table(wf["in_sample"], key="bt_is")
with wf_right:
    st.markdown(f"**Out-of-sample** (from {wf['split_date']})")
    show_metric_table(wf["out_sample"], key="bt_oos")

if decay > 0.5:
    caveat(
        f"Sharpe fell {decay:.2f} out-of-sample ({wf['in_sample']['sharpe_ratio']:.2f} → "
        f"{wf['out_sample']['sharpe_ratio']:.2f}). The size of the gap is the finding, not the level "
        f"of either number."
    )
if wf["out_sample"]["num_trades"] < 10:
    caveat(
        f"Only {wf['out_sample']['num_trades']} trades out-of-sample. Whatever the Sharpe ratio says, "
        f"it is describing that many independent bets — state that caveat before quoting the number.",
        level="info",
    )
quant_note("walk_forward")
