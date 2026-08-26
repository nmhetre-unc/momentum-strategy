"""Run regime-aware strategies and attribute the result to a mechanism."""

import pandas as pd
import streamlit as st

from adaptive import ADAPTIVE_DOCS, ADAPTIVE_STRATEGIES, describe_choices, describe_filter
from analytics import full_report, performance_by_regime
from backtest import run_backtest
from regime_dashboard import (
    PERFORMANCE_CONFIG, caveat, drawdown_chart, equity_chart, metric_row,
    position_chart, quant_note, require_regimes, show_metric_table,
)
from strategies import STRATEGIES
from walk_forward import evaluate_out_of_sample

df, regimes = require_regimes()
ticker = st.session_state["ticker"]

quant_note("adaptive", expanded=True)

# ---------- Controls ----------
with st.container(border=True):
    controls = st.columns([2, 1, 1])
    adaptive_name = controls[0].selectbox(
        "Adaptive strategy", list(ADAPTIVE_STRATEGIES), key="ad_strategy",
    )
    base = controls[1].selectbox(
        "Base strategy", list(STRATEGIES)[:3], key="ad_base",
        help="The underlying rule the wrapper adapts. Ignored by regime_switch and adaptive_ensemble, which choose per regime.",
    )
    cost_bps = controls[2].number_input(
        "Cost (bps)", 0.0, 50.0, 5.0, step=1.0, key="ad_cost",
        help="Adaptive strategies trade more than the rules they wrap. Comparing them at zero cost flatters them.",
    )

    params = {"regimes": regimes}
    if adaptive_name in ("regime_filtered", "regime_parameters", "regime_sized", "volatility_targeted"):
        params["base"] = base
    if adaptive_name in ("volatility_targeted", "adaptive_ensemble"):
        params["target_vol"] = st.slider(
            "Target volatility (annualized)", 0.05, 0.40, 0.15, step=0.01, key="ad_target",
            help="Aim for roughly this much volatility. Setting it far above what the asset realizes makes the cap inactive.",
        )
    if adaptive_name == "ml_regime_conditional":
        params["regime_mode"] = st.segmented_control(
            "Regime mode", ["feature", "conditional"], default="feature", key="ad_mode",
        ) or "feature"

doc = ADAPTIVE_DOCS[adaptive_name]
with st.container(border=True):
    st.markdown(
        f"**Mechanism — {doc['mechanism']}.** {doc['what']}\n\n"
        f"**Why it might work:** {doc['why']}\n\n"
        f"**Watch for:** {doc['watch_for']}"
    )

# ---------- Run the adaptive strategy and its unadapted base ----------
signal = ADAPTIVE_STRATEGIES[adaptive_name](df, **params)
result = run_backtest(df, signal, cost_bps=cost_bps, regimes=regimes.labels)
stats = full_report(result)

base_for_comparison = base if adaptive_name != "ml_regime_conditional" else "ml_direction"
base_result = run_backtest(df, STRATEGIES[base_for_comparison](df), cost_bps=cost_bps)
base_stats = full_report(base_result)

st.subheader(f"{ticker} · {adaptive_name}", divider="gray")
metric_row(stats, ["total_return", "sharpe_ratio", "max_drawdown", "exposure"])

if stats["exposure"] == 0:
    caveat(
        "This strategy is flat for the entire period, so every metric above is zero. That is a "
        "real answer rather than a bug: on the learning window, the base strategy had a negative "
        "Sharpe in every regime, so the allow-list came back empty. Not trading is the correct "
        "response to 'there was no condition in which this worked'."
    )
elif stats["exposure"] < 0.25:
    caveat(
        f"Average exposure is only {stats['exposure']:.0%}. The metrics above are computed on a "
        f"small slice of the period — check the day count before comparing them with a "
        f"fully-invested strategy.",
        level="info",
    )

st.altair_chart(equity_chart(result))
st.altair_chart(drawdown_chart(result))
st.altair_chart(position_chart(result))
st.caption(
    "The position chart is the fastest way to see which mechanism you're running: a square wave "
    "means filtering or switching, a breathing line means volatility targeting."
)

# ---------- Did adapting help? ----------
st.subheader("Did adapting actually help?", divider="gray")
comparison = pd.DataFrame({
    f"{base_for_comparison} (unadapted)": base_stats,
    f"{adaptive_name}": stats,
}).loc[["total_return", "cagr", "annualized_volatility", "sharpe_ratio",
        "sortino_ratio", "max_drawdown", "exposure", "turnover", "num_trades"]]
st.dataframe(comparison, key="ad_compare")

sharpe_gain = stats["sharpe_ratio"] - base_stats["sharpe_ratio"]
drawdown_gain = stats["max_drawdown"] - base_stats["max_drawdown"]
extra_turnover = stats["turnover"] - base_stats["turnover"]

verdict = st.columns(3)
verdict[0].metric("Sharpe change", f"{sharpe_gain:+.2f}")
verdict[1].metric("Drawdown change", f"{drawdown_gain:+.1%}",
                  help="Positive is better here — both numbers are negative, so a positive change means a shallower drawdown.")
verdict[2].metric("Extra turnover", f"{extra_turnover:+.1f}x",
                  help="What the adaptation cost you in trading, per year.")

if sharpe_gain <= 0 and extra_turnover > 0:
    caveat(
        f"The adaptation added {extra_turnover:.1f}x of annual turnover and did not improve "
        f"risk-adjusted return. That is a complete result and worth reporting as one — the simpler "
        f"strategy wins here."
    )

# ---------- What did `auto` decide? ----------
if adaptive_name in ("regime_switch", "adaptive_ensemble", "regime_filtered"):
    st.subheader("What the automatic rule decided, and on what evidence", divider="gray")
    st.caption(
        "Every automatic choice is learned from the first 60% of history only, then applied "
        "unchanged. Read the evidence table before trusting the choice."
    )
    if adaptive_name == "regime_filtered":
        described = describe_filter(df, base=base, regimes=regimes)
        allowed_names = [regimes.names.get(r, str(r)) for r in described["allowed"]]
        blocked_names = [regimes.names.get(r, str(r)) for r in described["blocked"]]
        st.markdown(
            f"**Learning window ends:** {described['learn_end'].date()}  \n"
            f"**Traded in:** {', '.join(allowed_names) or '_nothing_'}  \n"
            f"**Sat out:** {', '.join(blocked_names) or '_nothing_'}"
        )
        evidence = described["table"]
    else:
        described = describe_choices(df, regimes=regimes)
        choices = pd.DataFrame([
            {"Regime": regimes.names.get(regime_id, str(regime_id)),
             "Strategy chosen": name or "(insufficient evidence — traded normally)"}
            for regime_id, name in sorted(described["choices"].items())
        ])
        st.dataframe(choices, hide_index=True, key="ad_choices")
        evidence = described["table"]

    evidence = evidence.copy()
    evidence["regime"] = evidence["regime"].map(lambda r: regimes.names.get(r, str(r)))
    st.dataframe(
        evidence, hide_index=True, key="ad_evidence",
        column_config={
            "regime": st.column_config.TextColumn("Regime"),
            "strategy": st.column_config.TextColumn("Candidate"),
            "days": st.column_config.NumberColumn("Days in learning window"),
            "sharpe": st.column_config.NumberColumn("Sharpe (learning window)", format="%.2f"),
            "mean_return": st.column_config.NumberColumn("Mean daily return", format="%.5f"),
        },
    )
    st.caption(
        "If the winning candidate in a regime beat the runner-up by 0.05 of Sharpe over 80 days, "
        "that is a coin flip dressed as a decision, and it will not repeat."
    )

# ---------- Per-regime attribution ----------
st.subheader("Where the P&L came from", divider="gray")
table = performance_by_regime(result, regimes.labels, regimes.names)
if table.empty:
    st.info("No labelled days to attribute yet.", icon=":material/info:")
else:
    st.dataframe(table.drop(columns=["regime"]), hide_index=True,
                 column_config=PERFORMANCE_CONFIG, key="ad_by_regime")

# ---------- Honest evaluation ----------
st.subheader("Out-of-sample", divider="gray")
st.caption(
    "The adaptive wrappers learn their rules from the first 60% of history; this split holds out "
    "the last 30%, so the held-out period is genuinely unseen by the rule as well as by the strategy."
)
wf = evaluate_out_of_sample(df, ADAPTIVE_STRATEGIES[adaptive_name], **params)
wf_left, wf_right = st.columns(2)
with wf_left:
    st.markdown(f"**In-sample** (to {wf['split_date']})")
    show_metric_table(wf["in_sample"], key="ad_is")
with wf_right:
    st.markdown(f"**Out-of-sample** (from {wf['split_date']})")
    show_metric_table(wf["out_sample"], key="ad_oos")

quant_note("position_sizing")
