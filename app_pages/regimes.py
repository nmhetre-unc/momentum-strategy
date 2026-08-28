"""
Detect market regimes, check whether they're real, and see where a
strategy's return came from.

This page carries the teaching layer for regime detection. The three
questions it is built to make unavoidable, in order: are these actually
regimes, do they differ enough to condition on, and were the labels
knowable at the time. Every chart from the plain version is unchanged --
the explanations, caveats and reading guides are layered on top.
"""

import numpy as np
import pandas as pd
import streamlit as st

from analytics import benchmark_by_regime, performance_by_regime
from backtest import run_backtest
from regime import (
    REGIME_METHOD_DOCS, SMOOTHING_DOCS, detect_regimes, regime_episodes,
    regime_stability, regime_summary,
)
from regime_dashboard import (caveat, chart_caption, common_mistakes, duration_histogram,
    explainer, how_to_read, next_steps, page_intro, performance_by_regime_chart,
    PERFORMANCE_CONFIG, quant_note, regime_feature_chart, regime_palette,
    regime_probability_chart, regime_ribbon_chart, REGIME_SUMMARY_CONFIG, require_regimes,
    show_regime_health, table_caption, transition_heatmap
)
from regime_features import FEATURE_DOCS
from strategies import STRATEGIES, STRATEGY_DOCS

df, regimes = require_regimes()
settings = st.session_state["regime_settings"]

page_intro("regimes")
common_mistakes("regimes")

# --------------------------------------------------------------------------
# Onboarding: what a regime is, before any model output appears
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        "#### New here? A market regime in ninety seconds\n"
        "A **regime** is a persistent market environment with its own statistical "
        "character — its own typical volatility, its own tendency to trend or to reverse. "
        "Calm grinding bull markets, directionless chop, and violent selloffs are different "
        "regimes, and they last weeks to months rather than days."
    )

    why_left, why_right = st.columns(2)
    with why_left:
        st.markdown(
            "**Why this matters for strategies**\n\n"
            "Every strategy in this project is *already* a bet on a regime, whether or not "
            "it says so. Trend-following is a bet that you are in a trending market. "
            "Mean-reversion is the opposite bet. A single full-period Sharpe ratio averages "
            "the environments where a strategy thrived with the ones where it bled, and "
            "describes neither.\n\n"
            "Splitting performance by regime turns *\"my strategy has a Sharpe of 0.6\"* into "
            "something you can act on: **+1.4 in trends, −0.5 in chop** tells you which "
            "days to stop trading."
        )
    with why_right:
        st.markdown(
            "**Why volatility is the primary axis**\n\n"
            "Volatility *clusters* — quiet days follow quiet days, violent days follow "
            "violent ones, for weeks at a time. Direction does not: tomorrow's return is "
            "close to a coin flip.\n\n"
            "So volatility is the thing a regime model can actually find, because it is the "
            "thing that persists. Regime IDs here are always renumbered by realized "
            "volatility, which is what keeps **regime 0 = calmest** true across every "
            "method, refit and ticker."
        )

    st.markdown("**Reading the names and the colors**")
    name_left, name_right = st.columns([3, 2])
    with name_left:
        st.markdown(
            "Names come from where a regime's centre sits on two axes — how volatile, and "
            "which direction:\n\n"
            "| Name | Means |\n|---|---|\n"
            "| **Calm Uptrend** | Low volatility, rising. The easy money environment. |\n"
            "| **Quiet Range / Choppy** | Low-to-mid volatility, going nowhere. Trend strategies bleed here. |\n"
            "| **Turbulent** | High volatility, no clear direction. Violent both ways. |\n"
            "| **Crisis / Selloff** | High volatility, falling. Where drawdowns are made. |\n"
            "| **Warm-up** | *Not a regime.* The first ~1 year has no labels because the features need that much history to exist. |\n\n"
            "When two regimes land in the same quadrant they're distinguished by their actual "
            "volatility, e.g. *Crisis / Selloff (33% vol)*."
        )
    with name_right:
        n_shown = max(len(regimes.names), 1)
        swatches = regime_palette(n_shown)
        st.markdown(
            "Regimes are an **ordered** variable, so they get one colour ramp rather than "
            "unrelated hues. Severity reads as distance from the page background — the "
            "crisis regime is the strongest shade, calm is the faintest."
        )
        legend = pd.DataFrame({
            "Regime": [regimes.names[i] for i in sorted(regimes.names)],
            "Severity": [f"{i} — {'calmest' if i == 0 else 'most violent' if i == max(regimes.names) else 'middle'}"
                         for i in sorted(regimes.names)],
        })
        st.dataframe(legend, hide_index=True, key="regime_legend")
        st.caption(
            "Grey always means Warm-up. Colour never carries meaning alone here — every "
            "chart also has a legend, tooltips, and a table of the same numbers."
        )

with st.expander("How to read this page", icon=":material/map:"):
    st.markdown(
        """
Eight things appear below, in a deliberate order: **first check the regimes are real,
then check they differ, then check what they did to your strategy.**

**1 · Regime ribbon** — price with the detected regimes shaded behind it. This is your
sanity check. You know what 2020 looked like; if the model calls it calm, you have
learned something about the model rather than about 2020.

**2 · Probability bands** — the model's confidence per day, stacked to 100%. Where one
band dominates the model is sure. Where they interleave it is guessing — and those days
are transitions, which is exactly when a regime-switching strategy acts.

**3 · Transition matrix** — P(tomorrow's regime | today's regime). Read the **diagonal**
first; everything else on this page depends on it being high.

**4 · The persistence diagonal** — the diagonal cells specifically. On daily data these
should be **0.95+**. A diagonal near 1/k (0.33 for three regimes) means tomorrow is
independent of today, which means you have found noise, not regimes.

**5 · Expected duration** — `1 / (1 − p_stay)`, straight off that diagonal. It converts
a probability into "this regime typically lasts N days", which is the form you can
actually reason about.

**6 · Episode length histogram** — how long each *actual* visit to a regime lasted. If
the mass piles up at the left edge, your labels are flickering.

**7 · Strategy-by-regime chart** — one strategy's P&L split by the regime in force that
day. Read the `days` column before believing any bar.

**8 · Feature explorer** — the raw inputs the model clustered on, coloured by the regime
they produced. This is where you see *why* a day was labelled the way it was.
"""
    )

st.caption(REGIME_METHOD_DOCS[settings["method"]])
st.caption(f"**Smoothing — {settings['smooth']}:** {SMOOTHING_DOCS[settings['smooth']]}")

stability = regime_stability(regimes.labels)
show_regime_health(regimes, stability)

# --------------------------------------------------------------------------
# Health caveats: the checks that decide whether anything below is meaningful
# --------------------------------------------------------------------------
matrix = regimes.transition_matrix()
diagonal = np.diag(matrix.to_numpy()) if not matrix.empty else np.array([])
expected_durations = (
    {matrix.index[i]: (1 / (1 - diagonal[i])) if diagonal[i] < 1 else float("inf")
     for i in range(len(diagonal))}
    if len(diagonal) else {}
)

if len(diagonal) and diagonal.min() < 0.95:
    weakest = matrix.index[int(diagonal.argmin())]
    caveat(
        f"**Persistence is only {diagonal.min():.2f} for {weakest}.** On daily data a real "
        f"regime should stay put with probability 0.95 or better. Below that, tomorrow's label "
        f"is close to independent of today's — which means it carries little predictive content "
        f"and a strategy conditioned on it will mostly pay transaction costs. Raise the "
        f"confirmation window in the sidebar, or reduce the number of regimes."
    )

short_lived = {name: d for name, d in expected_durations.items() if d < 15}
if short_lived:
    caveat(
        "**Expected duration under 15 days for: "
        + ", ".join(f"{name} ({d:.0f}d)" for name, d in short_lived.items())
        + ".** Real regimes last weeks to months. A regime that typically lasts a fortnight is "
          "hard to trade around: by the time smoothing confirms it, a large part of it is gone."
    )

if settings.get("decode") == "viterbi" and not settings.get("walk_forward"):
    caveat(
        "**Viterbi decoding is not causal.** It finds the single most likely state *path* given "
        "the entire sequence, so the label on any given day is informed by days that hadn't "
        "happened yet. The labels look cleaner for exactly that reason. Use it to describe "
        "history; switch back to `filter` before quoting any performance number conditioned on "
        "these labels."
    )

if settings.get("smooth") == "min_duration" and settings.get("min_duration", 0) > 1:
    st.info(
        f"**Smoothing costs you {settings['min_duration']} days of lag on every real transition.** "
        f"A new regime must repeat for {settings['min_duration']} consecutive days before it is "
        f"accepted. That is the honest price of not being whipsawed by one-day flickers — but it "
        f"means you enter each new regime late, and the start of a new regime is often where the "
        f"largest move is. Set it to 1 to see the unsmoothed labels.",
        icon=":material/schedule:",
    )

quant_note("regime_volatility_clusters")

# ---------- The ribbon ----------
st.subheader("Regimes over time", divider="gray")
st.altair_chart(regime_ribbon_chart(df, regimes))
st.caption(
    "Check the labels against your own reading of the chart. If the model calls a crash calm, "
    "you have learned something about the model rather than about the crash."
)
how_to_read(
    """
- **Long unbroken bands are good.** They mean the model found persistent states rather than
  reacting to every wiggle. Bands should be measured in months.
- **Rapid switching is a red flag.** If the ribbon looks striped, the labels are flickering.
  No strategy can trade that: you'd flip your whole position every few days and pay for the
  privilege. Raise the confirmation window.
- **Check the transitions against events you remember.** Does the model turn turbulent around
  the crashes you know about? Does it turn calm during the quiet stretches? If not, ask what
  it *is* keying on before you trust it.
- **Watch how late transitions arrive.** Smoothing shifts every regime change a few days
  after the fact. Compare where the band changes against where the price actually broke.
"""
)
quant_note("regimes")
explainer(
    "What a regime really is",
    "weather patterns for the market — you can't forecast next month's weather, but knowing "
    "it's hurricane season changes what you do today.",
    """
The weather analogy carries further than most, and it's worth pushing on because it
tells you what regime detection can and cannot do.

**What it gets right:**

- *Persistence.* Storms last days, not minutes. Regimes last weeks to months. Both are
  worth naming precisely because they persist — a "regime" that changed daily would be
  as useless as a forecast that changed hourly.
- *Conditioning, not forecasting.* Knowing it's hurricane season doesn't tell you when
  the storm hits, but it changes how you build. Knowing you're in a high-volatility
  regime doesn't tell you when it ends — it tells you to size smaller now.
- *Uncertainty concentrates at the edges.* The hardest days to forecast are the days the
  weather turns. Regime labels are least reliable exactly at transitions, which is
  precisely when acting on them matters most. This is the central difficulty of the
  entire field, and no smoothing parameter makes it go away.

**Where the analogy breaks:** weather is governed by physics that don't change. Market
regimes are produced by participants who adapt, so the regimes themselves drift over
decades — see [[regime_drift]]. The 2017 "high volatility" regime would be classified as
calm by a model trained through 2008.
""",
)

probability_chart = regime_probability_chart(regimes)
if probability_chart is not None:
    with st.expander("Model confidence over time", icon=":material/percent:"):
        st.altair_chart(probability_chart)
        chart_caption(
            "The model's confidence in each regime, day by day, stacked to 100%.",
    "A dominant band means the model is sure; interleaved bands mean it is guessing.",
    "stretches where no band clears about 60% — those are transitions, and they are when a switching strategy acts.",
        )
        st.markdown(
            "Where the bands are cleanly separated the model is confident. Where they interleave "
            "it is guessing — and those days are transitions, which is exactly when a "
            "regime-switching strategy acts."
        )
        st.markdown(
            "**How to interpret this:** look for the stretches where no single band is above "
            "~60%. Those are days the model genuinely cannot tell which regime it's in. If your "
            "adaptive strategy trades on the argmax label, it is acting with full conviction on "
            "days like these — which is an argument for probability-weighted sizing over hard "
            "switching."
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

    rare = summary[summary["share"] < 0.10]
    if not rare.empty:
        caveat(
            "**Rare regimes: "
            + ", ".join(f"{row['name']} ({row['share']:.0%} of days, {int(row['days'])}d)"
                        for _, row in rare.iterrows())
            + ".** A regime covering under 10% of the sample gives you very few independent "
              "observations, and often only one or two distinct episodes. One episode is an "
              "event, not a regime — you cannot build a rule from it, and any statistic computed "
              "on it will not survive contact with new data.",
            level="info",
        )

how_to_read(
    """
This is the table that decides whether the rest of the page is worth reading. **If the
regimes don't differ here, they aren't regimes.**

- **Compare the `Ann. vol` column across rows.** This is the axis the model is really
  separating on, and the spread should be large — often 2-3x from calmest to most violent.
- **Then compare `Ann. return`.** A big spread here is a bonus, not a requirement. Volatility
  separates cleanly; direction rarely does.
- **Check `Episodes`, not just `Days`.** A regime with 400 days across one visit is a single
  historical event. The same 400 days across eight visits is a recurring state you might
  actually be able to trade.
- **`Avg days` should be weeks to months.** Under ~15 days and you are looking at noise.
"""
)

# ---------- Transitions ----------
st.subheader("Transitions and persistence", divider="gray")
transition_left, transition_right = st.columns([1, 1])
with transition_left:
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
        chart_caption(
            "How long each actual visit to a regime lasted.",
    "Each bar counts episodes of that length.",
    "mass on the right, in weeks and months. Mass piled at the left edge means the labels are flickering.",
        )
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

how_to_read(
    """
**Transition matrix** — each row is "given today is this regime", each cell is the
probability of tomorrow. Rows sum to 1.

- **The diagonal is the whole story.** 0.98 means the regime survives 98% of days. Under
  0.95, tomorrow's label is nearly independent of today's and the model has found noise.
- **Off-diagonal cells show the escalation path.** Markets usually move calm → middle →
  crisis rather than jumping. A large direct calm→crisis probability means either
  mislabelling, or an asset that genuinely gaps — both worth knowing before sizing it.
- **Expected duration** converts persistence into days: `1 / (1 − p_stay)`. 0.95 → 20 days,
  0.98 → 50 days, 0.99 → 100 days.

**Episode histogram** — how long actual visits lasted, as opposed to how long the model
*expects* them to.

- **Mass on the right is what you want** — episodes measured in weeks and months.
- **Mass piled at the left edge means flickering.** Several 1-5 day episodes are the model
  changing its mind, not the market changing state.
- **Compare the histogram against expected duration.** If expected duration says 40 days but
  most episodes ran 5, the transition matrix is being dominated by a few long visits and the
  typical experience is much choppier than the average suggests.
"""
)
quant_note("regime_transition_persistence")
explainer(
    "Persistence, transitions, and why smoothing costs you",
    "persistence is how long the weather lasts; transitions are how often it changes; "
    "smoothing is refusing to call it winter until you've seen five cold days in a row.",
    f"""
**Persistence** is the diagonal — the chance today's weather is still here tomorrow. High
persistence is what makes a regime worth detecting at all: if the state didn't stick, the
label would describe the past with no bearing on tomorrow, and conditioning on it would be
pointless.

**Transitions** are the off-diagonal cells — how often, and in which direction, the weather
changes. This page currently shows about
**{stability['switches_per_year']:.1f} switches a year**, with episodes averaging
**{stability['avg_duration']:.0f} days**.

**Smoothing** is the deliberate lag. The `min_duration` filter refuses to accept a new
regime until it has repeated for N consecutive days — exactly like refusing to declare
winter on the first cold morning. It works, and it costs you:

| Without smoothing | With smoothing |
|---|---|
| Reacts immediately | Reacts N days late |
| Whipsawed by one-day flickers | Ignores flickers |
| High turnover, high costs | Fewer, larger trades |
| Catches the start of the move | **Misses the start of the move** |

That last row is the real trade-off. The beginning of a new regime is frequently where the
largest move happens, and smoothing guarantees you miss it. There is no setting that avoids
both problems — you are choosing which error to make, and the honest thing is to say which
you chose and why.

**Why every smoother here is backward-looking.** A centered filter — one that uses days on
both sides — would produce visibly cleaner bands and would be lookahead bias. See
[[lookahead_bias]].
""",
)

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
    chart_caption(
        "The selected metric for this strategy, split by regime.",
    "Bars are ordered calmest to most violent, matching the colour ramp.",
    "a large gap between bars — and check the day count in the table below before believing it.",
    )

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

    # Is the best-vs-worst gap larger than the noise in the estimate? The
    # standard error of an annualized Sharpe is roughly sqrt(252/N); two
    # independent estimates combine in quadrature.
    if len(table) >= 2:
        best = table.loc[table["sharpe_ratio"].idxmax()]
        worst = table.loc[table["sharpe_ratio"].idxmin()]
        gap = best["sharpe_ratio"] - worst["sharpe_ratio"]
        combined_error = np.sqrt(252 / max(best["days"], 1) + 252 / max(worst["days"], 1))

        if gap < combined_error:
            caveat(
                f"**The regime difference is not statistically meaningful.** {best['name']} "
                f"({best['sharpe_ratio']:.2f}) versus {worst['name']} ({worst['sharpe_ratio']:.2f}) "
                f"is a gap of {gap:.2f}, against a combined standard error of roughly "
                f"±{combined_error:.2f} on those sample sizes. That gap is inside the noise — "
                f"you cannot conclude this strategy prefers one regime over the other, and a "
                f"filter built on it would be fitting randomness."
            )
        else:
            st.success(
                f"**{best['name']}** (Sharpe {best['sharpe_ratio']:.2f}, {int(best['days'])}d) versus "
                f"**{worst['name']}** ({worst['sharpe_ratio']:.2f}, {int(worst['days'])}d) — a gap of "
                f"{gap:.2f} against a combined standard error of about ±{combined_error:.2f}. "
                f"The difference clears the noise, so it is worth acting on: try filtering out "
                f"{worst['name']} on the Adaptive page.",
                icon=":material/insights:",
            )

    how_to_read(
        """
- **Read the `Days` column first, every time.** Standard error on an annualized Sharpe is
  roughly `sqrt(252/N)`. Over 100 days that's ±1.6 — wide enough to contain almost any
  conclusion you might want to draw.
- **The worst regime is more actionable than the best.** Knowing where a strategy loses lets
  you stop trading there. Knowing where it wins usually just confirms what you assumed.
- **Check the benchmark table before crediting the strategy.** A long-only strategy looks
  good in every regime the asset rose in. The question is whether it beat *holding* in that
  regime.
- **Watch `Exposure` per regime.** A strategy that is barely invested in the crisis regime
  isn't skilfully avoiding it — it may just be flat for unrelated reasons.
- **A large, sample-backed gap is the setup for the Adaptive page.** That's the whole case
  for regime filtering, and it's the one mechanism that reliably survives out-of-sample.
"""
    )

quant_note("risk_by_regime")
quant_note("trend_vs_chop")

# ---------- Features ----------
st.subheader("What the model is looking at", divider="gray")
available = [c for c in regimes.features.columns if c in FEATURE_DOCS]
feature = st.selectbox("Regime feature", available, key="regime_feature")
st.caption(FEATURE_DOCS[feature])
feature_chart = regime_feature_chart(regimes, feature)
if feature_chart is not None:
    st.altair_chart(feature_chart)
    chart_caption(
        "One regime feature over time, each day coloured by the regime it produced.",
    "This is why a day was labelled the way it was.",
    "clean colour separation by height on volatility features, and far more overlap on trend features.",
    )

# How strongly does this feature actually separate the regimes? Between-group
# spread over within-group spread -- a plain one-way F-statistic in spirit.
feature_values = regimes.features[feature]
group_means = {
    regimes.names[r]: feature_values[regimes.labels == r].mean()
    for r in sorted(regimes.names)
}
group_means = {k: v for k, v in group_means.items() if not pd.isna(v)}
if len(group_means) >= 2:
    between = np.std(list(group_means.values()))
    within = feature_values[regimes.valid()].std()
    separation = between / within if within else 0.0
    ranked = sorted(group_means.items(), key=lambda kv: kv[1])
    st.markdown(
        f"**Separation score for `{feature}`: {separation:.2f}** — the spread *between* regime "
        f"averages divided by the overall spread. Above ~0.5 means this feature genuinely "
        f"distinguishes the regimes; near 0 means it is along for the ride.\n\n"
        + " · ".join(f"{name}: `{value:.3f}`" for name, value in ranked)
    )

how_to_read(
    """
This is where you see *why* a day was labelled the way it was — each dot is one day,
coloured by the regime it ended up in.

- **Start with `vol_20d` or `vol_percentile`.** You should see the colours stack almost
  cleanly by height. That is volatility doing most of the separating, and it is the
  expected result rather than a disappointment.
- **Then try `trend_60d` or `price_vs_sma200`.** The colours will overlap far more.
  Direction separates regimes much less cleanly than volatility does — which is exactly why
  the ID ordering uses volatility alone.
- **Then try `efficiency_ratio`.** Low values mean the market travelled a long way and got
  nowhere. This is the chop axis, and it explains trend-following's performance better than
  volatility does.
- **Use the separation score above** to compare features quickly. A feature scoring near
  zero contributed nothing to these labels, however sensible it sounds.
- **Look at the transitions.** Watch how a feature moves in the days *before* the colour
  changes. That's what the model reacted to — and how much of the move it missed while
  waiting for confirmation.
"""
)

# ---------- The lookahead demonstration ----------
st.subheader("The lookahead demonstration", divider="gray")
quant_note("regime_lookahead", expanded=True)
explainer(
    "Why this is the most expensive mistake in the field",
    "grading a student's exam using tomorrow's answer key — they'll look brilliant, and "
    "you'll have learned nothing about whether they can pass the test.",
    """
Fitting a clustering model or an HMM on all of history embeds the whole history in every
label. A day in 2015 labelled "low volatility regime" was assigned to a cluster whose
centre was computed partly from 2020 and 2022. In 2015, that cluster did not exist.

The resulting backtest looks excellent for a boring reason: **the strategy knows which
regime it is in with a precision that was never available.**

What makes this trap so effective is that nothing about it looks wrong. The code is clean,
the model is standard, the equity curve is beautiful, and the error is invisible unless you
specifically ask when each label became knowable.

**The comparison below is the point of the whole page.** Run it, and read the label
agreement number: every disagreement is a day where knowing the future changed the answer.
Then note that the honest version has *no labels at all* for its first two years — because
you genuinely had no model then, and pretending otherwise is the entire bias.

This same failure appears anywhere a preprocessing step is fitted on the full sample:
scalers, PCA, feature selection, outlier clipping, percentile ranks. Every one must be
fitted on training data only. It is one of the most common serious errors in submitted
quant work, and it is worth being slightly paranoid about.
""",
)
quant_note("lookahead_bias")
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
        chart_caption(
            "Regimes from a model fitted on the entire history.",
            "Boundaries look crisp because the model already knew what came next.",
            "how closely the bands align with the turning points — that precision was never available.",
        )
    if honest is not None:
        with honest_column:
            st.markdown("**Expanding-window refit** — tradeable")
            st.altair_chart(regime_ribbon_chart(df, honest, height=240))
            chart_caption(
                "The same method, refitted on an expanding window and labelling only forward.",
                "Noisier, later to turn, and blank for the first two years.",
                "the missing start — you genuinely had no model then, and this is what honest looks like.",
            )

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
        how_to_read(
            f"""
- **{1 - agreement:.1%} of days were labelled differently.** Each of those is a day where the
  full-sample model used information that didn't exist yet to reach a different answer.
- **Compare the two ribbons visually.** The leaky one has crisper boundaries and switches
  closer to the actual turning points. That crispness is not skill — it is hindsight.
- **Note the missing start.** The honest version is blank for its first years. A backtest
  that quietly labels those days is claiming a model you did not have.
- **Now ask the question that matters:** if you had run the honest version live, would the
  strategy still have worked? That is what the Adaptive and Validation pages are for.
"""
        )

quant_note("regime_drift")

# --------------------------------------------------------------------------
# Where to go next
# --------------------------------------------------------------------------
next_steps("regimes")
