"""
Backtest one strategy and read the result properly.

This page carries the teaching layer for the core backtest loop: every
number has a tooltip, every chart has a plain-language explainer, and the
warnings fire on the specific conditions that make a backtest number
untrustworthy. The computation is unchanged from the plain version --
same signal, same equity curve, same walk-forward split -- and everything
educational is layered on top of it rather than replacing anything.
"""

import numpy as np
import pandas as pd
import streamlit as st

from analytics import full_report, performance_by_regime, sharpe_ratio
from backtest import run_backtest
from quant_notes import METRIC_DOCS
from regime_dashboard import (cached_regimes, caveat, chart_caption, common_mistakes,
    drawdown_chart, equity_chart, explainer, metric_row, next_steps, page_intro,
    PERFORMANCE_CONFIG, position_chart, quant_note, require_data, show_metric_table,
    table_caption
)
from regime_features import efficiency_ratio
from strategies import PARAM_SPECS, STRATEGIES, STRATEGY_DOCS, default_params
from walk_forward import evaluate_out_of_sample

df = require_data()
ticker = st.session_state["ticker"]

page_intro("backtest")
common_mistakes("backtest")


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

    # Off by default: regime detection is the most expensive thing in the
    # project, and the page must stay fast for someone who just wants a
    # backtest. Turning it on answers "where did the return come from?"
    show_regimes = st.toggle(
        "Attribute P&L by market regime", value=False, key="bt_regimes",
        help=(
            "Splits this strategy's daily P&L by the market regime in force that day, using the "
            "regime model configured in the sidebar. Adds a few seconds on first use, then cached."
        ),
    )

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

# ---------- Orientation for beginners ----------
with st.container(border=True):
    st.markdown("**New to reading backtests? Open these first.**")
    explainer(
        "How to read an equity curve",
        "the health chart of your strategy — a patient's temperature over time, not a final grade.",
        """
An equity curve is the growth of $1 invested in the strategy. Read it in
this order, and resist reading the endpoint first — the endpoint is the
least informative part of the chart.

1. **Shape, not endpoint.** Two curves can both end at 2.0x with completely
   different characters: one rising steadily, one flat for four years then
   doubling in six months. The second one's return belongs to a specific
   market episode. If that episode doesn't repeat, neither does the return.
2. **The flat stretches.** Flat means out of the market, or churning. How
   long is the longest one? A strategy with an 18-month flat period is one
   you would have abandoned in month nine — which means its full-period
   return was never actually available to you.
3. **The slope, late vs. early.** If all the gain is in the first third,
   the strategy may have stopped working. Check the Validation page.
4. **The benchmark line.** A strategy that made 80% while buy-and-hold made
   120% did not make money. It lost 40% of what doing nothing would have paid.
5. **Turn on log scale.** On a linear axis later years always look more
   dramatic, because the same percentage move is a bigger absolute one. Steady
   compounding is a *straight line* in logs.
""",
    )
    explainer(
        "How to interpret the drawdown chart",
        "the valleys — every stretch where the strategy was underwater and you were losing money you had already made.",
        """
The drawdown chart shows how far below the running peak the strategy was,
at every point in time. It is always zero or negative: zero means at a new
high, -20% means you had lost a fifth of your peak value.

**Read two dimensions, not one.**

- **Depth** — the worst point. This is `max_drawdown`, and it's the number
  that decides whether a strategy is survivable.
- **Duration** — how long the curve stayed below zero. A -15% drawdown that
  recovers in a month is an inconvenience. The same -15% taking two years to
  recover is a strategy you would have quit.

**The asymmetry that makes this matter.** -25% needs +33% to recover. -50%
needs +100%. Losses and the gains that undo them are not symmetric, which is
why avoiding the deep hole beats climbing out of it.

Always read this chart *together* with the equity curve above it. The equity
curve shows what you earned; this one shows what you had to endure to earn it.
""",
    )
    explainer(
        "Why buy-and-hold is always shown",
        "the control group in a drug trial — without it you cannot tell whether the medicine did anything.",
        """
The dashed line on the equity chart is simply holding the asset over the same
dates. It is the bar every strategy has to clear, and it is the comparison most
backtests quietly omit.

**Why omitting it is so tempting.** A long-only strategy trading a rising index
inherits most of the index's return. An 80% gain looks impressive until you see
the index made 120% while you were busy.

**What the comparison actually tells you.** Your strategy's real contribution is
the *difference* from holding — which is frequently negative once you count
costs and the good days it sat out.

**When underperforming is still fine.** A strategy that captures most of the
return with half the drawdown, or at 40% average exposure (leaving capital free
for other things), is a genuine result. But you have to make that argument
explicitly, on risk-adjusted terms — not by leaving the benchmark off the chart.
""",
    )

# ---------- Run ----------
signal = STRATEGIES[strategy_name](df, **params)
result = run_backtest(df, signal, cost_bps=cost_bps)
stats = full_report(result)

# Same strategy with costs switched off, so the cost question in the
# guided interpretation below is answered with a number rather than a guess.
free_stats = full_report(run_backtest(df, signal, cost_bps=0.0))

benchmark = result.copy()
benchmark["strategy_return"] = result["daily_return"]
benchmark["position"] = 1.0
benchmark_stats = full_report(benchmark)

st.subheader(f"{ticker} · {strategy_name}", divider="gray")
metric_row(stats, ["total_return", "sharpe_ratio", "max_drawdown", "exposure"])

# Every remaining metric, each with its tooltip from METRIC_DOCS. Hover any
# label to get what the number means AND what it hides.
metric_row(
    stats,
    ["cagr", "annualized_volatility", "sortino_ratio", "win_rate", "num_trades", "turnover"],
    columns=6,
)
st.caption(
    ":material/info: Hover any metric label for what it measures and what it conceals. "
    "No number on this page appears without an explanation attached to it."
)

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

# ---------- Contextual caveats ----------
# Each one fires on a specific condition that makes a headline number less
# trustworthy than it looks. Silence here is meaningful: it means none of
# these particular traps applies to this run.
st.markdown("**Things to check about this specific run**")
caveats_fired = 0

if stats["num_trades"] < 20:
    caveats_fired += 1
    caveat(
        f"**Only {stats['num_trades']} trades.** Every metric above is describing that many "
        f"independent bets, no matter how many days the backtest covers. A Sharpe ratio built on "
        f"a handful of trades is a statement about luck, not process. Widen the date range or use "
        f"a faster parameter set before drawing conclusions."
    )

if stats["exposure"] < 0.25:
    caveats_fired += 1
    caveat(
        f"**Exposure is only {stats['exposure']:.0%}.** The strategy is in cash most of the time, "
        f"so its risk metrics are computed on a small slice of the period and are not directly "
        f"comparable with a fully-invested strategy. It also means most of your capital was idle — "
        f"real money would want somewhere else to be."
    )

if stats["turnover"] > 3.0:
    caveats_fired += 1
    annual_drag = stats["turnover"] * cost_bps / 10_000
    caveat(
        f"**Turnover is {stats['turnover']:.1f}x a year.** At {cost_bps:.0f}bps that is roughly "
        f"{annual_drag:.2%} of annual cost drag before slippage. High-turnover strategies look best "
        f"in frictionless backtests and degrade fastest in reality — drag the cost input upward and "
        f"watch what happens."
    )

if stats["max_drawdown"] < -0.40:
    caveats_fired += 1
    recovery = 1 / (1 + stats["max_drawdown"]) - 1
    caveat(
        f"**Max drawdown of {stats['max_drawdown']:.1%}.** Recovering from that requires a "
        f"{recovery:+.0%} gain — losses and the gains that undo them are not symmetric. At a fund "
        f"a drawdown this size usually means the book is cut before the recovery arrives, so the "
        f"backtest's later returns were not available to you."
    )

if stats["sharpe_ratio"] < 0.5:
    caveats_fired += 1
    caveat(
        f"**Sharpe of {stats['sharpe_ratio']:.2f}** is weak — below roughly 0.5 the return is hard "
        f"to distinguish from noise given how wide the error bar on a Sharpe ratio is "
        f"(about ±{np.sqrt(252 / max(len(result), 1)):.2f} over this sample). Don't tune parameters "
        f"until it looks better; that is how overfitting starts.",
        level="info",
    )

if show_regimes:
    settings = st.session_state["regime_settings"]
    try:
        regimes = cached_regimes(
            df, settings["method"], settings["n_regimes"], settings["fit_frac"],
            settings["smooth"], settings["min_duration"], settings["decode"],
            settings["walk_forward"],
        )
    except ValueError as exc:
        regimes = None
        caveat(f"Regime attribution unavailable: {exc}", level="info")
    if regimes is not None and not regimes.causal:
        caveats_fired += 1
        caveat(
            "**The regime labels are not causal.** The regime model was fitted on the same days "
            "it is labelling, so the labels embed knowledge of the future. That is fine for "
            "*describing* history and invalid for any performance number conditioned on it. Set "
            "the sidebar's fit fraction below 1.0, or switch on walk-forward detection, before "
            "quoting anything from the regime breakdown below."
        )
else:
    regimes = None

if caveats_fired == 0:
    st.success(
        "None of the standard traps fired for this run — trade count, exposure, turnover, "
        "drawdown depth and Sharpe are all in reasonable territory. That is not proof the "
        "strategy works; it means the obvious reasons to distrust these numbers are absent.",
        icon=":material/verified:",
    )

# ---------- Charts ----------
st.altair_chart(equity_chart(result, log_scale=log_scale))
chart_caption(
    "Growth of $1 in the strategy against buy-and-hold.",
    "Both start at 1.0, so the vertical gap is the strategy's contribution.",
    "the shape and the flat stretches, not the endpoint — and whether the dashed benchmark line is above you.",
)
st.altair_chart(drawdown_chart(result))
chart_caption(
    "How far below its running peak the strategy sat, day by day.",
    "Always zero or negative; the trough is the max drawdown.",
    "both depth and duration — a shallow hole you sit in for two years is still a strategy you would have quit.",
)

quant_note("equity_curve")
quant_note("drawdown_vs_return")
quant_note("sharpe_can_mislead")
if doc["family"] == "Trend-following":
    quant_note("trend_in_chop")

detail_left, detail_right = st.columns([1, 1])
with detail_left:
    st.markdown("**Full metrics**")
    show_metric_table(stats, key="bt_metrics")
    st.caption(
        f"Buy-and-hold over the same window for reference: "
        f"{benchmark_stats['total_return']:.1%} return, "
        f"Sharpe {benchmark_stats['sharpe_ratio']:.2f}, "
        f"max drawdown {benchmark_stats['max_drawdown']:.1%}."
    )
with detail_right:
    st.markdown("**Position over time**")
    st.altair_chart(position_chart(result))
    st.caption(
        f"Turnover {stats['turnover']:.1f}x a year at {cost_bps:.0f}bps costs roughly "
        f"{stats['turnover'] * cost_bps / 10_000:.2%} annually — "
        f"{abs(stats['turnover'] * cost_bps / 10_000 / stats['cagr']):.0%} of this strategy's CAGR."
        if stats["cagr"] else "This strategy's CAGR is zero, so the cost ratio is undefined."
    )

    explainer(
        "What exposure means",
        "the share of the game you actually spent on the field rather than on the bench.",
        f"""
Exposure is the **average absolute position** — here **{stats['exposure']:.0%}**. A binary
strategy that is long half the days has 50% exposure; one that is always long has 100%.

**Why it changes how you read everything else.** A Sharpe of 1.0 at 20% exposure and a
Sharpe of 1.0 at 100% exposure are very different results. The first was earned on a fifth
of the days, so it rests on a fifth of the evidence and carries a much wider error bar.

**Two things it reveals.**

- *Capital efficiency.* At {stats['exposure']:.0%} exposure, most of your money sat idle. Real
  money would want somewhere else to be, and the return on total capital is lower than the
  headline suggests.
- *Whether "risk reduction" is real skill.* Any strategy can cut drawdown by simply being
  invested less. The question is whether it was out of the market at the *right* times —
  which is what the risk-adjusted metrics, not the raw drawdown, are there to answer.

**{METRIC_DOCS['exposure']}**
""",
    )
    explainer(
        "What turnover means",
        "how many times a year you replaced the entire contents of your portfolio.",
        f"""
Turnover is total position change per year in full-position units — here
**{stats['turnover']:.1f}x**. Turnover of 20 means you turned the whole book over twenty
times in a year.

**The cost arithmetic.** Roughly `turnover x cost_bps / 10,000` is your annual drag. At the
current {cost_bps:.0f}bps setting that is about **{stats['turnover'] * cost_bps / 10_000:.2%} a
year**, before slippage and market impact, which this model does not include at all.

**Why it reorders rankings.** Costs do not hit strategies evenly. A 200-day crossover trades
twice a decade and barely notices; a daily ML signal trades constantly and can lose half its
edge. Frictionless backtests systematically flatter high-frequency signals — which is exactly
the group that looks best when you forget to turn costs on.

**A habit worth forming.** For any strategy, find the cost level at which it breaks even. If
the answer is 3bps, it is a theoretical object. If it survives 20bps, it deserves more work.

**{METRIC_DOCS['turnover']}**
""",
    )

# ---------- Guided interpretation ----------
st.subheader("What this backtest tells you", divider="gray")
st.caption(
    "Five questions, answered from the numbers above rather than from the strategy's description. "
    "Each one is a claim you should be able to defend."
)

# 1. Is it behaving like a trend-follower? Declared family, checked empirically:
#    compare the trailing 60-day return on days it held vs. days it was flat.
trailing_trend = df["Close"].pct_change(60)
held = result["position"] > 0
trend_when_long = trailing_trend[held].mean()
trend_when_flat = trailing_trend[~held].mean()
trend_gap = (trend_when_long or 0) - (trend_when_flat or 0)
behaves_trend = trend_gap > 0

# 2. Does it bleed in chop? Split days by the CAUSAL expanding percentile of
#    the Kaufman efficiency ratio -- low means the market travelled a long way
#    and went nowhere. No regime model needed, and no lookahead.
chop_score = efficiency_ratio(df["Close"], 60)
chop_rank = chop_score.expanding(min_periods=120).rank(pct=True)
returns = result["strategy_return"].fillna(0)
choppy_days = chop_rank < 0.33
trending_days = chop_rank > 0.67
chop_sharpe = sharpe_ratio(returns[choppy_days]) if choppy_days.sum() > 60 else np.nan
trend_sharpe = sharpe_ratio(returns[trending_days]) if trending_days.sum() > 60 else np.nan
bleeds_in_chop = (
    not pd.isna(chop_sharpe) and not pd.isna(trend_sharpe) and chop_sharpe < trend_sharpe
)

# 3. Do costs matter? Measured, not assumed.
cost_sharpe_hit = free_stats["sharpe_ratio"] - stats["sharpe_ratio"]
costs_matter = cost_sharpe_hit > 0.15

# 4/5. Out-of-sample decay and overall robustness.
wf = evaluate_out_of_sample(df, STRATEGIES[strategy_name], **params)
decay = wf["in_sample"]["sharpe_ratio"] - wf["out_sample"]["sharpe_ratio"]
decayed = decay > 0.5

robust_checks = {
    "Positive out-of-sample Sharpe": wf["out_sample"]["sharpe_ratio"] > 0,
    "Sharpe decay under 0.5": decay <= 0.5,
    "At least 20 trades": stats["num_trades"] >= 20,
    "Survives transaction costs": stats["sharpe_ratio"] > 0,
    "Max drawdown shallower than -40%": stats["max_drawdown"] >= -0.40,
}
robust_score = sum(robust_checks.values())

verdicts = [
    (
        "Is it trend-following?",
        f"{'Yes' if behaves_trend else 'No'} — declared **{doc['family']}**",
        (
            f"On days it held a position the trailing 60-day return averaged "
            f"{trend_when_long:+.2%}, against {trend_when_flat:+.2%} on days it was flat — a gap of "
            f"{trend_gap:+.2%}. Positive means it systematically holds *after* the market has "
            f"already risen, which is what trend-following is."
            if behaves_trend else
            f"On days it held, the trailing 60-day return averaged {trend_when_long:+.2%} versus "
            f"{trend_when_flat:+.2%} when flat. It is not buying strength — consistent with a "
            f"mean-reversion or contrarian bet."
        ),
    ),
    (
        "Does it bleed in sideways markets?",
        "Not measurable on this sample" if pd.isna(chop_sharpe) else
        ("Yes" if bleeds_in_chop else "No"),
        (
            "Not enough days in the low-efficiency bucket to judge — widen the date range."
            if pd.isna(chop_sharpe) else
            f"Sharpe was **{chop_sharpe:.2f}** in the choppiest third of days (lowest efficiency "
            f"ratio — the market travelled far and went nowhere) versus **{trend_sharpe:.2f}** in "
            f"the most trending third. "
            + (
                "That gap is the structural cost of a trend rule, not bad luck."
                if bleeds_in_chop else
                "It held up in chop, which is unusual for a trend rule and worth investigating."
            )
        ),
    ),
    (
        "Do transaction costs matter?",
        "Yes" if costs_matter else ("Barely" if cost_bps else "Not tested — costs are off"),
        (
            f"Sharpe falls {cost_sharpe_hit:.2f} going from 0bps to {cost_bps:.0f}bps "
            f"({free_stats['sharpe_ratio']:.2f} → {stats['sharpe_ratio']:.2f}) on turnover of "
            f"{stats['turnover']:.1f}x a year. "
            + (
                "This strategy's edge is materially eaten by trading. Any conclusion drawn at "
                "zero cost would have been wrong."
                if costs_matter else
                "Low turnover means costs barely register here — one of the quiet advantages of "
                "slow strategies."
            )
            if cost_bps else
            "Costs are set to 0bps, so this is a frictionless result. Set the cost input to 5 and "
            "compare — the ranking of strategies frequently changes."
        ),
    ),
    (
        "Did out-of-sample performance decay?",
        "Yes" if decayed else "Held up",
        (
            f"Sharpe went {wf['in_sample']['sharpe_ratio']:.2f} in-sample → "
            f"{wf['out_sample']['sharpe_ratio']:.2f} out-of-sample, a decay of {decay:+.2f}. "
            + (
                "A gap that size means the in-sample number was substantially fitted to that "
                "specific period."
                if decayed else
                "A small gap is what generalization looks like — and it matters more than the "
                "level of either number."
            )
        ),
    ),
    (
        "Is it robust?",
        f"{robust_score} of {len(robust_checks)} checks passed",
        "  \n".join(
            f"{':material/check_circle:' if passed else ':material/cancel:'} {name}"
            for name, passed in robust_checks.items()
        ),
    ),
]

for question, verdict, reasoning in verdicts:
    with st.container(border=True):
        st.markdown(f"**{question}**  →  {verdict}")
        st.markdown(reasoning)

if robust_score <= 2:
    caveat(
        f"Only {robust_score} of {len(robust_checks)} robustness checks passed. Treat this "
        f"strategy as a teaching example rather than a candidate — and resist the urge to tune "
        f"parameters until the checks go green, because that is precisely how a backtest gets "
        f"overfitted to its own sample."
    )

# ---------- Walk-forward ----------
st.subheader("Walk-forward validation", divider="gray")
st.caption(
    "The numbers above cover the whole period, including data any fitted component saw. "
    "These don't."
)

explainer(
    "How to interpret walk-forward validation",
    "training on the past and testing on the future — like studying last year's exams, then "
    "sitting this year's paper. Only the second score counts.",
    """
The data is cut chronologically. The first 70% is **in-sample**: the period a
strategy's parameters, model or rules could have been influenced by. The last
30% is **out-of-sample**: data that played no part in building it.

**What to read, in order.**

1. **The gap, not the levels.** A modest drop from in- to out-of-sample is
   normal and expected. A collapse — or a sign flip — means the in-sample number
   was mostly curve-fitting. What you want is a *small gap*, because the gap is
   what tells you the process generalizes.
2. **Drawdown, not just return.** A strategy that keeps its return but doubles
   its drawdown out-of-sample has still degraded.
3. **The trade count.** Two trades out-of-sample is two coin flips. No Sharpe
   ratio computed on that means anything, in either direction.

**What this cannot protect you from.** Looking at the out-of-sample result,
adjusting the strategy, and looking again. Do that three times and your holdout
has quietly become in-sample. There is no technical fix — only the discipline of
deciding in advance and looking once.

**One split is still one draw.** A single 30% holdout might just have landed in a
friendly stretch. The Validation page runs ten to fifteen consecutive holdouts
instead, and the share of positive folds is more informative than any single number.
""",
)

wf_left, wf_right = st.columns(2)
with wf_left:
    st.markdown(f"**In-Sample** (to {wf['split_date']})")
    show_metric_table(wf["in_sample"], key="bt_is")
with wf_right:
    st.markdown(f"**Out-of-Sample** (from {wf['split_date']})")
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

# ---------- Optional regime attribution ----------
if show_regimes and regimes is not None:
    st.subheader("Where the P&L came from", divider="gray")
    if regimes.meta.get("walk_forward"):
        fit_description = "refit on an expanding window"
    else:
        fit_description = f"fitted on the first {regimes.meta.get('fit_frac', 1.0):.0%} of history"
    st.caption(
        f"Regime model: **{regimes.method}**, {fit_description} · configure it in the sidebar."
    )
    regime_result = run_backtest(df, signal, cost_bps=cost_bps, regimes=regimes.labels)
    table = performance_by_regime(regime_result, regimes.labels, regimes.names)

    if table.empty:
        st.info("No labelled days to attribute performance to yet.", icon=":material/info:")
    else:
        st.dataframe(
            table.drop(columns=["regime"]), hide_index=True,
            column_config=PERFORMANCE_CONFIG, key="bt_by_regime",
        )
        best = table.loc[table["sharpe_ratio"].idxmax()]
        worst = table.loc[table["sharpe_ratio"].idxmin()]
        st.markdown(
            f"Best regime: **{best['name']}** (Sharpe {best['sharpe_ratio']:.2f} over "
            f"{int(best['days'])} days). Worst: **{worst['name']}** "
            f"(Sharpe {worst['sharpe_ratio']:.2f} over {int(worst['days'])} days). "
            f"Knowing the *worst* regime is the more actionable half — it is the one you can "
            f"choose not to trade."
        )
        thin = table[table["days"] < 200]
        if not thin.empty:
            caveat(
                "Regimes with under 200 days here: "
                + ", ".join(f"**{row['name']}** ({int(row['days'])}d)" for _, row in thin.iterrows())
                + ". Standard error on an annualized Sharpe is roughly sqrt(252/days), so those "
                  "rows carry error bars wide enough to contain almost any conclusion.",
                level="info",
            )
    quant_note("risk_by_regime")

# --------------------------------------------------------------------------
# Where to go next
# --------------------------------------------------------------------------
next_steps("backtest")
