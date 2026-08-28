"""
Next steps: what to do once the curriculum ends.

A standalone page. Computes nothing and reads no prices, so it renders
instantly and works offline.

The through-line: the platform can teach you to evaluate a strategy
someone handed you. Becoming a researcher means running the loop on a
question nobody handed you, and the first section is a concrete recipe
for doing exactly that.
"""

import streamlit as st

from regime_dashboard import (
    caveat, common_mistakes, explainer, how_to_read, next_steps, page_intro, quant_note,
)

page_intro("next_steps")
common_mistakes("next_steps")

with st.expander("How to read this page", icon=":material/map:"):
    st.markdown(
        """
**Four sections, in the order you should use them.**

1. **Build your first research project** — the single highest-value thing to do next. Eight
   steps, using only tools already in the platform. Do this before anything else here.
2. **Extend the simulator** — seven ways to add to the codebase. Genuinely useful, and
   deliberately second: adding a sixth regime model is more fun than validating the three
   that exist, and it is how you end up with more tools and no findings.
3. **Deepen your quant skills** — six directions that take you beyond this repository.
4. **Suggested pathways** — five specialisations, each three to five steps, if you would
   rather follow a track than improvise.

**None of this needs new code to start.** The first project uses the pages you already know.
Every extension idea is scoped to something you could finish in an afternoon.

**The one rule that matters throughout.** State your hypothesis, and what would falsify it,
*before* you run anything. Everything else on this page is downstream of that habit.
"""
    )

with st.container(border=True):
    st.markdown(
        "#### You have finished the core curriculum\n"
        "Backtest, Regimes, Adaptive, ML lab, Validation and Exercises — plus the Graduation "
        "checklist, which tells you which habits actually stuck."
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**What changes from here**\n\n"
            "Until now every question was posed for you: *run this strategy, detect these "
            "regimes, interpret this gap.* Research means posing the question yourself, and "
            "that is a genuinely different skill — mostly because nobody tells you when to "
            "stop looking."
        )
    with right:
        st.markdown(
            "**The goal is a research habit, not a strategy**\n\n"
            "A finished project here is not one that made money. It is one where the "
            "hypothesis was stated first, the holdout was used once, the limitation is named "
            "up front, and someone else could reproduce it. That is the artefact worth having."
        )

# --------------------------------------------------------------------------
# 1. First research project
# --------------------------------------------------------------------------
st.subheader("Build your first research project", divider="gray")
st.caption(
    "Eight steps, using only what is already in the platform. Expect it to take a few "
    "sessions, and expect the answer to be \"no\" — which is a finding, not a failure."
)

PROJECT = [
    ("Pick a strategy you already understand completely",
     "`sma_crossover` or `mean_reversion`, not the ML model. The point is that anything "
     "surprising is then a fact about the market rather than about your code."),
    ("Write the hypothesis down before you run anything",
     "One sentence on what you expect and why, plus one on what would falsify it. "
     "\"Trend-following earns its return in high-efficiency-ratio periods and gives some back "
     "in low ones; if per-regime Sharpe is flat, I am wrong.\""),
    ("Run the baselines first",
     "Buy-and-hold on your ticker, then the unmodified strategy. Record CAGR, Sharpe, max "
     "drawdown and exposure. Everything later is measured against these, so write them down."),
    ("Check the regime structure before conditioning on it",
     "Are the regimes persistent, distinct and recurring? If not, stop — there is nothing to "
     "condition on, and that is itself a reportable result."),
    ("Test one adaptive mechanism, not four",
     "Start with filtering, the most robust. Note the turnover it added and whether the Sharpe "
     "improvement exceeded the cost drag."),
    ("Test whether an ML signal beats the rule you already had",
     "Not whether it is accurate. Compare its out-of-sample Sharpe against the rule-based "
     "baseline, with costs on, and check its accuracy against the base rate."),
    ("Validate with rolling walk-forward, and look once",
     "Read the share of positive folds and the fold *sequence*. Decide your acceptance "
     "criterion before you look, then honour it even when you dislike the answer."),
    ("Write a one-page summary",
     "Hypothesis, method, out-of-sample result, per-regime breakdown, what would falsify it, "
     "what you would do next. Lead with the limitation — it makes the rest more credible, and "
     "it is the version that survives questioning."),
]

for i, (step, detail) in enumerate(PROJECT, start=1):
    with st.container(border=True):
        st.markdown(f"**{i}. {step}**")
        st.caption(detail)

quant_note("walkforward_reason")
quant_note("overfitting")
how_to_read(
    """
- **The write-up is the project, not the backtest.** A result nobody can reproduce or
  falsify is not a finding, however good the number.
- **Expect the answer to be no.** Most hypotheses tested honestly do not survive. A clean
  negative result, stated with its evidence, is a complete piece of research.
- **Do steps 1 and 2 on paper before opening the app.** Once you have seen a chart, your
  hypothesis is no longer independent of the data.
"""
)

# --------------------------------------------------------------------------
# 2. Extend the simulator
# --------------------------------------------------------------------------
st.subheader("Extend the simulator", divider="gray")
st.caption(
    "Seven additions, each scoped to an afternoon. Do these after a project, not instead of one."
)

EXTENSIONS = [
    ("New regime models",
     "`regime.py` takes a method name and returns labels. Add a changepoint detector, a "
     "two-state model on volatility alone, or a jump model. The interface is the contract: "
     "return ordered labels and the rest of the platform works unchanged."),
    ("New adaptive mechanisms",
     "`adaptive.py` wrappers return a position series in [0, 1]. Try a drawdown-throttle that "
     "cuts size after a loss, or a confidence-weighted version that sizes by regime probability "
     "rather than switching on the argmax."),
    ("New ML models",
     "`MODEL_TYPES` in `ml_strategy.py` is a dict of factory functions. Gradient boosting, a "
     "regularised linear model, or an ensemble. Watch what each does to the train/test gap — "
     "that is the interesting part, not the accuracy."),
    ("New validation schemes",
     "Purged and embargoed cross-validation, which removes rows near the split boundary to stop "
     "overlapping windows leaking. Or refit fitted strategies per fold, which this "
     "implementation deliberately does not do."),
    ("New exercises",
     "`exercises.py` holds a dataclass per exercise with an automated check. Add one for a "
     "phenomenon you found confusing — writing the check forces you to define the effect "
     "precisely, which is most of the learning."),
    ("New glossary terms and quant notes",
     "`quant_notes.py` is plain data. A note you write for yourself, in your own words, is "
     "worth more than one you read — and it appears on the Learn page automatically."),
    ("New data sources",
     "`data_loader.py` returns an OHLCV frame. Point it at a different provider, a longer "
     "history, or an asset class with different behaviour. Futures and FX break assumptions "
     "that equities let you get away with."),
]

for title, detail in EXTENSIONS:
    with st.expander(title, icon=":material/extension:"):
        st.markdown(detail)

# --------------------------------------------------------------------------
# 3. Deepen your skills
# --------------------------------------------------------------------------
st.subheader("Deepen your quant skills", divider="gray")
st.caption("Six directions that take you beyond this repository.")

DEEPEN = [
    ("Read academic papers — and check their holdouts",
     "Start with the classics on momentum and volatility clustering. Read them the way this "
     "platform taught you to read a backtest: what was the sample, how many specifications "
     "were tried, and has the effect survived since publication? Many have not."),
    ("Replicate a published strategy",
     "Pick one with a clear rule and try to reproduce its reported numbers. You will usually "
     "fail, and the reasons — different universe, survivorship, unstated filters, costs — teach "
     "more than the strategy does."),
    ("Build your own dataset",
     "Assembling clean data is most of the job and none of the glamour. Corporate actions, "
     "delistings, restatements and timestamp alignment are where real lookahead bias enters, "
     "long before any modelling."),
    ("Experiment with alternative features",
     "Cross-asset signals, term structure, breadth, positioning. The features in `features.py` "
     "are deliberately conventional — the interesting question is what else is knowable at the "
     "close and might matter."),
    ("Explore intraday data",
     "Different problem entirely: microstructure, execution cost as a first-class concern, and "
     "far more observations but not proportionally more independent ones. It will change how "
     "you think about sample size."),
    ("Learn portfolio construction",
     "Everything here trades one asset. Real work combines many, where correlation, position "
     "sizing across a book and risk budgeting matter more than any single signal. This is the "
     "largest gap between this platform and the job."),
]

for title, detail in DEEPEN:
    with st.expander(title, icon=":material/school:"):
        st.markdown(detail)

quant_note("regime_drift")
quant_note("ml_feature_importance")

# --------------------------------------------------------------------------
# 4. Pathways
# --------------------------------------------------------------------------
st.subheader("Suggested pathways", divider="gray")
st.caption(
    "Five specialisations. Pick one and follow it — depth in a single direction is worth more "
    "than a shallow pass through all five."
)

PATHWAYS = {
    "Strategy research": [
        "Replicate one published strategy end to end, including its costs.",
        "Test it on three tickers and two date ranges, and note where it fails.",
        "Write the hypothesis that explains the failures.",
        "Test that hypothesis on data you have not used yet.",
        "Write it up leading with the limitation.",
    ],
    "ML research": [
        "Reproduce the train/test gap with a third model class.",
        "Add proper time-series cross-validation with purging and an embargo.",
        "Measure feature stability across splits, not just importance within one.",
        "Test whether any feature set beats the base rate out-of-sample after costs.",
        "Write up what would have to be true for the answer to be yes.",
    ],
    "Regime modelling": [
        "Implement a changepoint detector alongside the existing five methods.",
        "Compare its labels against the HMM's on agreement and persistence.",
        "Measure how much each method's labels change when refit on more data.",
        "Quantify the lookahead gap for your new method as a number.",
        "Decide, with evidence, whether extra sophistication bought anything.",
    ],
    "Adaptive mechanisms": [
        "Run all four mechanisms separately on the same strategy, with costs on.",
        "Attribute the total improvement to each mechanism individually.",
        "Build one new mechanism — a drawdown throttle or probability-weighted sizing.",
        "Test whether it beats plain volatility targeting, the control group.",
        "Report honestly if it does not; that is the most common outcome.",
    ],
    "Validation and robustness": [
        "Implement purged and embargoed cross-validation.",
        "Refit fitted strategies per fold, which this implementation does not.",
        "Measure how much the reported Sharpe falls once you do.",
        "Build a deflated Sharpe ratio that accounts for the number of trials.",
        "Apply it retrospectively to every result you have produced here.",
    ],
}

for name, steps in PATHWAYS.items():
    with st.expander(f"{name} path", icon=":material/route:"):
        for i, step in enumerate(steps, start=1):
            st.markdown(f"{i}. {step}")

quant_note("is_vs_oos")
quant_note("adaptive")

# --------------------------------------------------------------------------
# Warnings that follow you out of the curriculum
# --------------------------------------------------------------------------
st.subheader("What to keep checking, forever", divider="gray")
st.markdown(
    "The platform fires these for you. Your own projects will not, so these are the checks to "
    "carry out by hand on everything you build from here."
)

caveat(
    "**Small samples.** Ten years of daily data is 2,500 rows and perhaps three or four "
    "genuinely independent market environments. Your effective sample is always smaller than "
    "the row count suggests, and your confidence should shrink accordingly.",
    level="info",
)
caveat(
    "**A Sharpe ratio that looks too good.** Above 2 on daily equity data, assume a leak and "
    "go looking. This reflex will save you more embarrassment than any other habit here."
)
caveat(
    "**Excessive turnover.** Multiply turnover by your realistic cost and compare it against "
    "the expected return before believing any high-frequency result. Compute the cost level at "
    "which the strategy breaks even.",
    level="info",
)
caveat(
    "**Non-causal labels.** Any model fitted on the full sample and then used to backtest over "
    "it encodes the future. This applies to scalers, PCA, feature selection and outlier "
    "clipping just as much as to regime models."
)
caveat(
    "**Model instability.** Refit on a slightly different window. If the fitted model, its "
    "feature importances or its predictions change substantially, you are looking at variance "
    "rather than structure.",
    level="info",
)
caveat(
    "**Regime instability.** Regime definitions drift over decades. A model trained on "
    "2010-2019 never saw a crisis and has no cluster for one — so when it meets a crisis it "
    "must assign those days to something, and whatever it picks will be wrong.",
    level="info",
)

explainer(
    "The habit that outlasts everything on this platform",
    "a scientist reviewing their own paper before submitting it — looking hardest at the "
    "result they most want to be true.",
    """
Every technique here — walk-forward, regime attribution, cost modelling, base rates — is
machinery in service of one disposition: **assume your result is wrong until you have tried
to break it**, applied fastest to results you like.

That disposition transfers to any dataset, any language, any asset class, and it is what an
interviewer is probing for when they ask about your project. The specific strategies in this
repository do not transfer, and were never meant to.

The practical form it takes is unglamorous. Before you show anyone a result, ask: what was
the sample? How many things did I try? Is this in-sample? What is the benchmark? What would
falsify it? Five questions, two minutes, and they catch most of what goes wrong.

If you ask them of your own work before anyone else does, you are doing the job.
""",
)

how_to_read(
    """
- **Pick one pathway rather than sampling all five.** Depth in one direction produces
  something you can show; a shallow pass through all of them produces nothing.
- **Carry the six warnings above by hand.** Your own projects will not fire them for you,
  and that is precisely when they matter most.
- **Re-check the Graduation checklist after your first project.** Several items only become
  checkable once you have run the loop yourself.
"""
)

next_steps("next_steps")
