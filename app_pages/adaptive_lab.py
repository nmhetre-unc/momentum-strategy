"""
Run regime-aware strategies and attribute the result to a mechanism.

This page carries the teaching layer for adaptation. The question it is
built to make unavoidable is not "did the number go up" but **"which
mechanism moved it, and did it pay for its own turnover"** -- because the
usual honest answer is that the simplest mechanism did most of the work.
Every control and chart from the plain version is unchanged.
"""

import numpy as np
import pandas as pd
import streamlit as st

from adaptive import ADAPTIVE_DOCS, ADAPTIVE_STRATEGIES, describe_choices, describe_filter
from analytics import full_report, performance_by_regime
from backtest import run_backtest
from regime_dashboard import (
    PERFORMANCE_CONFIG, caveat, drawdown_chart, equity_chart, explainer, how_to_read,
    metric_row, position_chart, quant_note, require_regimes, show_metric_table,
)
from strategies import STRATEGIES
from walk_forward import evaluate_out_of_sample

df, regimes = require_regimes()
ticker = st.session_state["ticker"]

# --------------------------------------------------------------------------
# Onboarding: what adaptation is, before any of it runs
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        "#### New here? Adaptive strategies in ninety seconds\n"
        "An **adaptive strategy** wraps one of the fixed rules from the Backtest page and "
        "changes its behaviour depending on what kind of market you are in. The rule stays; "
        "*how you apply it* becomes conditional."
    )

    why_left, why_right = st.columns(2)
    with why_left:
        st.markdown(
            "**Why adapt at all**\n\n"
            "Every fixed strategy is a bet that the market is one thing. It isn't. A trend "
            "strategy typically earns its whole return in trending regimes and gives a chunk "
            "back in choppy ones — so its blended Sharpe describes a market that never "
            "existed.\n\n"
            "Adaptation acts on that: trade the rule where it works, and stop, resize or "
            "replace it where it doesn't."
        )
    with why_right:
        st.markdown(
            "**Why the simplest one often wins**\n\n"
            "Volatility is genuinely forecastable — it clusters. Direction is not. So "
            "**volatility targeting**, which needs no regime model at all, converts the "
            "one predictable quantity straight into a position size.\n\n"
            "Regime switching has to classify the environment correctly *and* pick the right "
            "strategy for it, and it pays turnover for both. Very often the plain sizing "
            "mechanism captures most of the benefit. When it does, that is the finding."
        )

    st.markdown("**The four mechanisms — they fail in different ways, so know which you're running**")
    st.markdown(
        "| Mechanism | What it does | Strategies here | Main risk |\n"
        "|---|---|---|---|\n"
        "| **Filtering** | Keeps the rule, sits out bad regimes | `regime_filtered` | Cuts exposure — metrics rest on fewer days |\n"
        "| **Switching** | Different rule per regime | `regime_switch` | Late labels + full position flips = turnover |\n"
        "| **Re-parameterizing** | Same rule, regime-specific settings | `regime_parameters` | Most degrees of freedom, easiest to overfit |\n"
        "| **Position sizing** | Same signal, scaled size | `volatility_targeted`, `regime_sized` | Improvement may just be *holding less* |\n"
        "\n"
        "`adaptive_ensemble` stacks switching and sizing; `ml_regime_conditional` conditions the "
        "ML model on regime. Both inherit the risks of their parts."
    )

    st.info(
        "**Read the exposure number on every result.** Filtering and sizing reduce exposure by "
        "design — that's the mechanism working. But any strategy can halve its drawdown by "
        "halving its position, and that is arithmetic rather than skill. If Sharpe improved, the "
        "strategy was out of the market at the *right* times. If only drawdown improved while "
        "Sharpe fell, it was simply out of the market — and you could have got the same by "
        "trading the original at half size.",
        icon=":material/pie_chart:",
    )

with st.expander("How to read this page", icon=":material/map:"):
    st.markdown(
        """
**The controls**

- **Adaptive strategy** — which wrapper to run. Each one implements a different mechanism;
  the panel underneath names it and says what to watch for.
- **Base strategy** — the underlying rule being adapted. Ignored by `regime_switch` and
  `adaptive_ensemble`, which pick a rule *per regime* rather than adapting a fixed one.
- **Cost (bps)** — charged on every unit of position change. This is not a cosmetic setting
  on this page: adaptive strategies buy their improvement with extra trading, and comparing
  them at 0bps systematically flatters the ones that trade most. Leave it at 5.
- **Target volatility** (sizing strategies only) — the annualized volatility to aim for.
  Set it far above what the asset realizes and the cap never binds, so nothing happens.
- **Regime model** — configured in the sidebar, shared with the Regimes page.

**The output, in order**

- **Performance metrics** — the adapted strategy's headline numbers. Hover any label.
- **Growth of $1** — the adapted equity curve against buy-and-hold.
- **Drawdown** — read this with the equity curve, always. Most of what sizing buys you
  shows up here and nowhere else.
- **Position chart** — the fastest way to identify your mechanism. A *square wave* means
  filtering or switching (in or out). A *breathing line* means volatility targeting
  (continuously resized). Steps that change only at regime boundaries mean regime sizing.
- **Comparison table** — adapted versus unadapted, same data, same costs, same period. This
  is the table that answers "did it help".
- **Evidence table** — for the automatic wrappers, the per-regime Sharpe of each candidate
  measured **only** on the learning window. This is the reasoning behind the choice, shown
  so you can disagree with it.
- **Regime attribution** — the adapted strategy's P&L split by regime, so you can see
  whether adaptation actually changed behaviour where it was supposed to.
- **Out-of-sample** — the only section whose numbers are honest for a fitted rule.
"""
    )

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

how_to_read(
    """
- **Filtering** removes exposure and never adds any, so its worst case is bounded — it can
  only take away the wrong days. Try it first.
- **Switching** replaces one rule with another. You now need the regime label *and* the
  per-regime choice to be right, and every switch is a full position flip.
- **Re-parameterizing** keeps the rule but changes its settings per regime. Most degrees of
  freedom of the four, so the in-sample curve improves almost by construction.
- **Position sizing** leaves the signal untouched and scales the size. It leans on
  volatility, which is forecastable, rather than direction, which isn't.

Whichever you pick, the question below is the same: **did it beat the unadapted rule by more
than the turnover it added?**
""",
    title="What this mechanism actually does",
)

# Mechanism-specific note, so the relevant one is surfaced rather than all six.
if adaptive_name == "regime_filtered":
    quant_note("adaptive_filtering")
elif adaptive_name in ("regime_switch", "adaptive_ensemble"):
    quant_note("adaptive_switching")
elif adaptive_name in ("volatility_targeted", "regime_sized"):
    quant_note("volatility_targeting")
elif adaptive_name in ("regime_parameters", "ml_regime_conditional"):
    quant_note("adaptive_overfitting")

explainer(
    "The four mechanisms, as driving",
    "filtering is staying home in the storm; switching is changing your entire driving style "
    "at every weather change; re-parameterizing is retuning the car for each road type; "
    "volatility targeting is just slowing down on ice.",
    """
The driving analogy separates these four cleanly, and it also predicts how each fails.

**Filtering — sitting out the storm.** You don't drive in the blizzard. Simple, safe, and
its only cost is the trips you didn't take. The risk is being wrong about which days are
blizzards, and the penalty for that is opportunity, not damage.

**Switching — changing your whole driving style.** Rally technique on gravel, defensive
technique in traffic. Powerful *if* you correctly identify the surface — but you have to
change everything at once, at the moment of transition, when you're least certain what
surface you're on. And the changeover itself costs you.

**Re-parameterizing — retuning the car per road.** Same car, different suspension settings
for each surface. Subtle and plausible, and every road type gives you a fresh set of dials
to fiddle with. Fiddle with enough dials against one recorded journey and you'll produce a
setup that was perfect for that journey and useless for the next.

**Volatility targeting — slowing down on ice.** You don't change the car, the route, or the
technique. You go slower when the road is slick. It requires no classification of *what
kind* of bad road it is — only a measurement of how slick it is right now. That is why it
is robust, and why it so often beats the cleverer three.

**And two measurements of your driving:**

- **Exposure** — how often you were actually on the road at all. A driver who only drives on
  perfect days has a wonderful safety record and doesn't get anywhere.
- **Turnover** — how often you changed lanes. Each change is small, and they add up to a
  real bill.
""",
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
metric_row(
    stats,
    ["cagr", "annualized_volatility", "sortino_ratio", "win_rate", "num_trades", "turnover"],
    columns=6,
)
st.caption(
    ":material/info: Hover any metric label for what it measures and what it conceals."
)

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
elif stats["exposure"] < 0.40:
    caveat(
        f"**Exposure is {stats['exposure']:.0%}** — the strategy is in cash most of the time. "
        f"Any drawdown improvement is partly the trivial consequence of holding less, so compare "
        f"risk-*adjusted* metrics rather than raw drawdown. And note that "
        f"{1 - stats['exposure']:.0%} of your capital sat idle; on total capital the return is "
        f"lower than the headline suggests.",
        level="info",
    )

if stats["turnover"] > 3.0:
    caveat(
        f"**Turnover is {stats['turnover']:.1f}x a year** — roughly "
        f"{stats['turnover'] * cost_bps / 10_000:.2%} of annual cost drag at {cost_bps:.0f}bps, "
        f"before slippage. Adaptation buys its improved risk profile with extra trading; the "
        f"comparison below is where you find out whether that trade was worth making."
    )

quant_note("exposure_caveat")

st.altair_chart(equity_chart(result))
st.altair_chart(drawdown_chart(result))
how_to_read(
    """
**Growth of $1** — compare the *shape* against the unadapted version, not just the endpoint.
Adaptation usually flattens the curve in the regimes it filters out, which looks like
underperformance in good times and pays for itself in bad ones.

**Drawdown** — this is where most of what adaptation buys you actually shows up. A
mechanism that leaves return roughly flat and cuts max drawdown materially is a large
improvement, even though the headline return barely moved. Read depth *and* duration.

**A shallower drawdown alone proves nothing.** Check exposure first — if the strategy was
only invested 35% of the time, a smaller drawdown is arithmetic, not skill.
"""
)

st.altair_chart(position_chart(result))
st.caption(
    "The position chart is the fastest way to see which mechanism you're running: a square wave "
    "means filtering or switching, a breathing line means volatility targeting."
)
how_to_read(
    """
- **Square wave, flat-topped at 1.0** — a binary mechanism. Filtering or switching: you're
  either fully in or fully out. Every vertical edge is a full round trip and a real cost.
- **Breathing line that varies daily** — volatility targeting. The position rises in calm
  markets and falls before turbulence, continuously. Look at what it was doing in the month
  *before* the largest drawdown; that's the mechanism earning its keep.
- **Steps that hold for weeks then jump** — regime sizing. The discrete cousin of
  targeting: same idea, far fewer trades.
- **Long stretches at zero** — the strategy sat out. Cross-check those against the regime
  ribbon on the Regimes page: is it avoiding what you'd expect it to avoid?
"""
)
quant_note("turnover_costs")

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
elif 0 < sharpe_gain < 0.10:
    caveat(
        f"**Sharpe improved by only {sharpe_gain:+.2f}.** The standard error on a Sharpe ratio "
        f"over this sample is around ±{np.sqrt(252 / max(len(result), 1)):.2f}, so a gain that "
        f"small is indistinguishable from noise. Adding a whole mechanism for it is complexity "
        f"you cannot justify from this evidence.",
        level="info",
    )

if 0 <= drawdown_gain < 0.05 and stats["exposure"] < base_stats["exposure"]:
    caveat(
        f"**Drawdown improved by only {drawdown_gain:.1%}** while exposure fell from "
        f"{base_stats['exposure']:.0%} to {stats['exposure']:.0%}. Holding less should shrink "
        f"drawdown on its own — this mechanism has barely done better than simply trading the "
        f"unadapted rule at a smaller size, which would have been far simpler.",
        level="info",
    )

how_to_read(
    """
- **Sharpe change is the headline** — it is the risk-adjusted comparison and the only one
  that isn't confounded by exposure. Anything under about 0.1 is inside the noise.
- **Drawdown change: positive is better.** Both figures are negative, so a positive delta
  means a shallower hole. But check the exposure row before crediting it.
- **Extra turnover is the price.** If the adaptation added 10x turnover for 0.03 of Sharpe,
  it did not pay for itself, and saying so plainly is a complete result.
- **The comparison to actually make.** Run `volatility_targeted` and then your chosen
  mechanism. If plain sizing captured most of the improvement, prefer it — the simpler
  explanation of the same outcome is worth more and survives longer.
"""
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

    # Was any per-regime choice actually decisive? Compare the winner's margin
    # against the standard error implied by the number of learning-window days.
    marginal = []
    for regime_name, group in evidence.dropna(subset=["sharpe"]).groupby("regime"):
        ranked = group.sort_values("sharpe", ascending=False)
        if len(ranked) < 2:
            continue
        margin = ranked.iloc[0]["sharpe"] - ranked.iloc[1]["sharpe"]
        error = np.sqrt(252 / max(ranked.iloc[0]["days"], 1))
        if margin < error:
            marginal.append(
                f"**{regime_name}**: {ranked.iloc[0]['strategy']} beat "
                f"{ranked.iloc[1]['strategy']} by only {margin:.2f} over "
                f"{int(ranked.iloc[0]['days'])} days (±{error:.2f})"
            )
    if marginal:
        caveat(
            "**Some choices rest on differences too small to be meaningful.** "
            + "; ".join(marginal)
            + ". A margin inside the standard error is a coin flip wearing a decision's clothes — "
              "the rule learned here will not reproduce on new data."
        )

    how_to_read(
        """
- **Read the evidence table before the choice.** The choice is just the argmax of one column;
  the table is the reasoning, and it is where you can disagree.
- **Check the `Days` column per regime.** A winner picked on 80 days carries a standard error
  of roughly ±1.8 on its Sharpe. Almost any margin is inside that.
- **Look at the margin, not the rank.** If the winner beat the runner-up by 0.05, the ranking
  would likely flip on a different sample — and there is no reason to believe this one.
- **Count the comparisons.** Three candidates across three regimes is nine comparisons on one
  price history. Even on pure noise, the best of nine looks respectable. This is strategy
  mining, and the out-of-sample section below is the only honest read on it.
"""
    )
    quant_note("adaptive_overfitting")

# ---------- Per-regime attribution ----------
st.subheader("Where the P&L came from", divider="gray")
table = performance_by_regime(result, regimes.labels, regimes.names)
if table.empty:
    st.info("No labelled days to attribute yet.", icon=":material/info:")
else:
    st.dataframe(table.drop(columns=["regime"]), hide_index=True,
                 column_config=PERFORMANCE_CONFIG, key="ad_by_regime")

    base_table = performance_by_regime(
        run_backtest(df, STRATEGIES[base_for_comparison](df), cost_bps=cost_bps, regimes=regimes.labels),
        regimes.labels, regimes.names,
    )
    if not base_table.empty:
        merged = table.merge(
            base_table[["regime", "sharpe_ratio", "exposure"]],
            on="regime", suffixes=("", "_base"),
        )
        merged["sharpe_delta"] = merged["sharpe_ratio"] - merged["sharpe_ratio_base"]
        merged["exposure_delta"] = merged["exposure"] - merged["exposure_base"]

        st.markdown("**What adaptation changed, regime by regime**")
        st.dataframe(
            merged[["name", "days", "sharpe_ratio_base", "sharpe_ratio",
                    "sharpe_delta", "exposure_delta"]],
            hide_index=True, key="ad_regime_delta",
            column_config={
                "name": st.column_config.TextColumn("Regime"),
                "days": st.column_config.NumberColumn("Days"),
                "sharpe_ratio_base": st.column_config.NumberColumn("Sharpe (unadapted)", format="%.2f"),
                "sharpe_ratio": st.column_config.NumberColumn("Sharpe (adapted)", format="%.2f"),
                "sharpe_delta": st.column_config.NumberColumn(
                    "Change", format="%.2f",
                    help="Positive means adaptation helped in this regime."),
                "exposure_delta": st.column_config.NumberColumn(
                    "Exposure change", format="percent",
                    help="How much more or less invested the adapted version was in this regime."),
            },
        )

        helped = merged[merged["sharpe_delta"] > 0.1]
        hurt = merged[merged["sharpe_delta"] < -0.1]
        if len(hurt) and len(helped):
            caveat(
                f"**Adaptation helped in {len(helped)} regime(s) and hurt in {len(hurt)}** — "
                + "; ".join(f"{row['name']} {row['sharpe_delta']:+.2f}" for _, row in merged.iterrows())
                + ". Inconsistent results across regimes mean the mechanism is not doing one "
                  "coherent thing. Before accepting the net improvement, check whether it comes "
                  "from a single regime with few days, which would make it an accident rather "
                  "than a mechanism.",
                level="info",
            )

    how_to_read(
        """
- **Compare adapted against unadapted per regime, not just overall.** A mechanism that was
  supposed to sit out the crisis regime should show a large exposure drop there. If it
  doesn't, it is not doing what its description claims.
- **The exposure-change column tells you where it acted.** Zero change in a regime means the
  mechanism was inactive there, whatever the Sharpe column says.
- **Consistency matters more than the total.** Helping in every regime is a mechanism.
  Helping enormously in one 90-day regime and hurting slightly elsewhere is an accident that
  happened to net positive.
- **Read `Days` before every conclusion.** Same rule as everywhere else on this platform.
"""
    )

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

decay = wf["in_sample"]["sharpe_ratio"] - wf["out_sample"]["sharpe_ratio"]
if decay > 0.5:
    caveat(
        f"Sharpe fell {decay:.2f} out-of-sample ({wf['in_sample']['sharpe_ratio']:.2f} → "
        f"{wf['out_sample']['sharpe_ratio']:.2f}). Adaptive wrappers have more moving parts than "
        f"the rules they wrap, and every extra part is another thing that can be fitted to the "
        f"learning window rather than to the market."
    )

how_to_read(
    """
- **This is the only section whose numbers are honest for a fitted rule.** Everything above
  includes the learning window, where the auto-selector already knew the answers.
- **The gap matters more than either level.** A modest drop is normal. A collapse means the
  mechanism was fitted rather than discovered.
- **Compare the decay against the unadapted strategy's decay** on the Validation page. If
  adapting made decay *worse*, the adaptation added overfitting rather than robustness —
  which is the single most common outcome for the more complex mechanisms.
"""
)
quant_note("position_sizing")

# ---------- Where to go next ----------
st.subheader("What to do next", divider="gray")
st.caption("Each page answers a question this one raised but cannot settle on its own.")

next_left, next_right = st.columns(2)
with next_left:
    with st.container(border=True):
        st.page_link("app_pages/regimes.py", label="**Regimes**", icon=":material/layers:")
        st.markdown(
            "Understand the environments behind the adaptation — and check they're real "
            "regimes before trusting any rule conditioned on them."
        )
    with st.container(border=True):
        st.page_link("app_pages/ml_lab.py", label="**ML lab**", icon=":material/network_intelligence:")
        st.markdown(
            "See how ML signals behave when conditioned on regime, and why splitting training "
            "data by regime usually costs more than it buys."
        )
    with st.container(border=True):
        st.page_link("app_pages/validation.py", label="**Validation**", icon=":material/fact_check:")
        st.markdown(
            "Check whether adaptation survives in-sample to out-of-sample across many rolling "
            "windows, and compare all eleven strategies at once with costs on."
        )
with next_right:
    with st.container(border=True):
        st.page_link("app_pages/backtest_lab.py", label="**Backtest**", icon=":material/query_stats:")
        st.markdown(
            "Compare the adapted result against the plain rule on its own terms, with the full "
            "caveat set and the guided interpretation."
        )
    with st.container(border=True):
        st.page_link("app_pages/exercises_lab.py", label="**Exercises**", icon=":material/assignment:")
        st.markdown(
            "Practise adaptive interpretation — including whether position sizing beats signal "
            "engineering, and what happens to the ranking when costs go on."
        )
    with st.container(border=True):
        st.page_link("app_pages/learn.py", label="**Learn**", icon=":material/menu_book:")
        st.markdown(
            "Every quant note in one browser, including the six on this page, plus the three "
            "pitfalls and the full metric glossary."
        )
