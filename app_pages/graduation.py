"""
Graduation checklist: thirty abilities the curriculum is meant to install.

A standalone page. It computes nothing and reads no prices, so it renders
instantly and works offline; the only state it writes is which boxes are
ticked, which persist for the session so visiting a linked page and coming
back does not wipe your progress.

Every item is phrased as something you can *do*, not something you have
read. That is deliberate: the checklist is a self-assessment of habits,
and habits are only visible in behaviour.
"""

import streamlit as st

from regime_dashboard import (
    caveat, common_mistakes, how_to_read, next_steps, page_intro, quant_note,
)

page_intro("graduation")
common_mistakes("graduation")

with st.expander("How to read this page", icon=":material/map:"):
    st.markdown(
        """
**The test for every item is the same:** could you do this right now, on a ticker you have
never looked at, without looking anything up? If you would need to re-read a page first, leave
it unticked. Nobody else sees this, and an honest unticked box is worth more than a generous
ticked one.

**Six sections, one per part of the curriculum.** Each maps to a page: A to Backtest, B to
Regimes, C to Adaptive, D to ML lab, E to Validation, F to Exercises and Learn.

**Read the unticked boxes as a reading list.** They name the page to go back to and what to
do differently when you get there. A cluster of unticked items in one section is far more
informative than a total.

**Ticks persist for this session**, including after you visit a linked page and return.

**What this checklist is not.** It does not ask whether you built a profitable strategy. Every
item asks whether you can tell a real result from a fitted one — because that is the skill
that transfers, and profitable strategies found by search usually do not.
"""
    )

# --------------------------------------------------------------------------
# What graduation means here
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        "#### What graduation means here\n"
        "You have worked through the six-step path: Backtest, Regimes, Adaptive, ML lab, "
        "Validation and Exercises. Graduation is not having *seen* those pages — it is being "
        "able to do the things below on data nobody has prepared for you."
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**This measures habits, not facts**\n\n"
            "None of these items asks you to recall a definition. Every one asks whether a "
            "reflex has formed — checking sample size before believing a Sharpe ratio, asking "
            "whether a number is in-sample or out-of-sample, comparing against buy-and-hold "
            "without being prompted.\n\n"
            "You can memorise every quant note on this platform and tick almost nothing here. "
            "The reverse is also true, and far more useful."
        )
    with right:
        st.markdown(
            "**The goal is to think like a quant, not to chase Sharpe**\n\n"
            "A high Sharpe ratio found by trying twenty things is not a result; it is the "
            "maximum of a search. A modest one you can defend — with a stated hypothesis, an "
            "honest holdout and a named limitation — is research.\n\n"
            "The second is what gets you hired, and it is what this checklist is scored on."
        )

# --------------------------------------------------------------------------
# The checklist
# --------------------------------------------------------------------------
# (key, ability, how you know you actually have it)
CHECKLIST = {
    "A. Backtesting skills": [
        ("grad_a1", "Read an equity curve and describe its character, not just its endpoint",
         "You mention the flat stretches and the slope over time before the final number."),
        ("grad_a2", "Identify drawdowns and read both their depth and their duration",
         "You can name the longest underwater stretch, not only the deepest one."),
        ("grad_a3", "Compare any strategy against buy-and-hold without being reminded",
         "The benchmark is the first thing you look for, and its absence bothers you."),
        ("grad_a4", "Detect unrealistic smoothness in a result",
         "A suspiciously clean equity curve makes you look for a leak before celebrating."),
        ("grad_a5", "Spot turnover and exposure problems in a metric table",
         "You check what fraction of days were invested before comparing risk-adjusted numbers."),
    ],
    "B. Regime skills": [
        ("grad_b1", "Interpret a regime ribbon against your own reading of the price chart",
         "You check whether the model calls the crashes you remember by their right names."),
        ("grad_b2", "Explain volatility clustering and why it is the primary regime axis",
         "You can say why volatility is detectable and direction largely is not."),
        ("grad_b3", "Read a transition matrix, starting from the diagonal",
         "You check persistence before reading anything else on the page."),
        ("grad_b4", "Interpret an episode-length histogram",
         "Mass piled at the left edge reads to you as flickering labels, not short regimes."),
        ("grad_b5", "Connect a strategy's behaviour to the regime it depends on",
         "You can name the environment a strategy needs and the one that breaks it."),
    ],
    "C. Adaptive skills": [
        ("grad_c1", "Explain filtering, switching, re-parameterising and sizing, and how each fails",
         "You can name a distinct failure mode for each, not just a description."),
        ("grad_c2", "Identify the mechanism from a position chart alone",
         "A square wave and a breathing line mean different things to you at a glance."),
        ("grad_c3", "Evaluate turnover against the Sharpe it bought",
         "You compute the cost drag before deciding whether an adaptation was worth it."),
        ("grad_c4", "Judge whether adaptation genuinely helped",
         "You check whether a smaller drawdown came from skill or from simply holding less."),
    ],
    "D. ML skills": [
        ("grad_d1", "Explain the base rate and why accuracy is meaningless without it",
         "You ask for the base rate before reacting to any accuracy figure."),
        ("grad_d2", "Interpret the train/test gap as the quantity of interest",
         "A high training accuracy worries you rather than pleasing you."),
        ("grad_d3", "Read a confusion matrix",
         "You look for a column of zeros before reading the diagonal."),
        ("grad_d4", "Detect a model that has collapsed to 'always long'",
         "You recognise it as buy-and-hold with extra cost, not as a working classifier."),
        ("grad_d5", "Judge whether the model improved profit and loss, not just accuracy",
         "You read the out-of-sample Sharpe and the turnover before forming a view."),
    ],
    "E. Validation skills": [
        ("grad_e1", "Explain in-sample and out-of-sample, and why the split must be chronological",
         "You can say why a random holdout leaks on time-series data."),
        ("grad_e2", "Interpret walk-forward folds as a distribution, not a set of numbers",
         "You read the share positive and the sequence, not just the average."),
        ("grad_e3", "Read a stitched out-of-sample equity curve",
         "You treat it as the closest thing to a track record and read its shape."),
        ("grad_e4", "Detect silent fitting in your own workflow",
         "You notice when you are about to adjust something after seeing a holdout result."),
        ("grad_e5", "Judge robustness from the in-sample to out-of-sample gap",
         "A small gap at a modest level impresses you more than a large level with a large gap."),
    ],
    "F. Exercises and Learn skills": [
        ("grad_f1", "Complete the exercises without opening the hints",
         "You form a view first and use hints to check reasoning, not to produce it."),
        ("grad_f2", "Explain a result in your own words before reading the answer",
         "You write the sentence first, then compare — and the gap tells you what to revisit."),
        ("grad_f3", "Use the quant notes as reference rather than as reading",
         "You look up the one note covering the number in front of you and go back."),
        ("grad_f4", "Recognise lookahead bias, overfitting and regime drift in your own work",
         "You check all three on any result you like the look of, without prompting."),
        ("grad_f5", "State a result with its limitation first",
         "Your summary leads with what would falsify it, and is more convincing for it."),
    ],
}

SECTION_NOTES = {
    "A. Backtesting skills": "equity_curve",
    "B. Regime skills": "regime_drift",
    "C. Adaptive skills": "adaptive",
    "D. ML skills": "ml_overfitting",
    "E. Validation skills": "is_vs_oos",
}

SECTION_PAGES = {
    "A. Backtesting skills": ("app_pages/backtest_lab.py", "Backtest"),
    "B. Regime skills": ("app_pages/regimes.py", "Regimes"),
    "C. Adaptive skills": ("app_pages/adaptive_lab.py", "Adaptive"),
    "D. ML skills": ("app_pages/ml_lab.py", "ML lab"),
    "E. Validation skills": ("app_pages/validation.py", "Validation"),
    "F. Exercises and Learn skills": ("app_pages/exercises_lab.py", "Exercises"),
}

ALL_KEYS = [key for items in CHECKLIST.values() for key, *_ in items]
done = sum(1 for key in ALL_KEYS if st.session_state.get(key, False))

st.subheader("The checklist", divider="gray")
st.progress(done / len(ALL_KEYS), text=f"{done} of {len(ALL_KEYS)} abilities ticked")

for section, items in CHECKLIST.items():
    section_done = sum(1 for key, *_ in items if st.session_state.get(key, False))
    with st.container(border=True):
        header = st.columns([3, 1])
        header[0].markdown(f"**{section}**")
        header[1].caption(f"{section_done} of {len(items)}")

        for key, ability, how_you_know in items:
            st.checkbox(
                ability, key=key, persist_state="session",
                help="Tick only if you could do this now, on an unfamiliar ticker, "
                     "without looking anything up.",
            )
            st.caption(f"↳ {how_you_know}")

        if section_done < len(items):
            path, label = SECTION_PAGES[section]
            st.page_link(path, label=f"Revisit **{label}**", icon=":material/arrow_forward:")

        if section in SECTION_NOTES:
            quant_note(SECTION_NOTES[section])

how_to_read(
    """
- **Read the unticked boxes, not the total.** They name the page to revisit and what to do
  differently there. A cluster in one section is far more useful than a score.
- **Re-check this after your first research project.** Several items — silent fitting, stating
  a limitation first — only become checkable once you have run the loop yourself.
- **An honest 18 beats a generous 28.** The point is knowing where you actually are, which is
  the same discipline the rest of the platform teaches about results.
"""
)

# --------------------------------------------------------------------------
# The warning signs a graduate should recognise unprompted
# --------------------------------------------------------------------------
st.subheader("The five warning signs you should catch unprompted", divider="gray")
st.markdown(
    "The platform fires these for you on every page. A graduate notices them **before** the "
    "warning appears — that is the difference between using the tool and having learned from it."
)

caveat(
    "**A Sharpe ratio above 2 on daily data.** The loudest alarm in quant research. It almost "
    "always means lookahead bias rather than genuine edge, so the reflex is to go looking for "
    "the leak rather than to celebrate."
)
caveat(
    "**Too few trades.** A ten-year backtest containing three trades gives you three "
    "independent bets. Check the count before quoting any risk-adjusted number from it.",
    level="info",
)
caveat(
    "**Regime episodes averaging under 15 days.** Real market environments last weeks to "
    "months. At that length the labels are flickering, and any per-regime conclusion drawn "
    "from them is describing noise."
)
caveat(
    "**Model accuracy at or below the base rate.** About 53% of days are up, so a model at 51% "
    "has been beaten by a one-line constant. Accuracy without its base rate is not a number "
    "you can evaluate.",
    level="info",
)
caveat(
    "**Walk-forward folds that swing wildly.** If out-of-sample Sharpe ranges from −1.5 to "
    "+2.0 across test windows, that spread is your real uncertainty. Quoting a single average "
    "to two decimal places is false precision.",
    level="info",
)

# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------
st.subheader("You are ready when…", divider="gray")
st.caption(
    "Seven conditions. None mentions returns — all of them are about judgement, which is what "
    "survives contact with a different dataset."
)

READY = [
    ("You ask \"in-sample or out-of-sample?\" automatically",
     "About any performance number, including — especially — your own."),
    ("You check sample size before believing a risk-adjusted number",
     "Trade count, day count, fold count. The habit fires before you read the Sharpe ratio."),
    ("A suspiciously good result makes you suspicious",
     "Your first response to a beautiful equity curve is to look for the leak."),
    ("You can name what would falsify your own result",
     "Stated in advance, not reverse-engineered after the fact."),
    ("You count every comparison you ran, including the discarded ones",
     "Twenty abandoned attempts spent your statistical power whether or not you report them."),
    ("You prefer the simpler explanation of the same outcome",
     "When plain volatility targeting matches the regime-switching model, you say so and use it."),
    ("You lead with the limitation and find it makes the case stronger",
     "\"My own walk-forward caught this overfitting\" is a better claim than any backtest."),
]

for condition, detail in READY:
    with st.container(border=True):
        st.markdown(f"**{condition}**")
        st.caption(detail)

st.info(
    "**If most of these are true, the curriculum has done its job.** What remains is not more "
    "material but more repetitions — on different tickers, different periods and questions "
    "nobody handed you. The Next steps page is about exactly that.",
    icon=":material/school:",
)

next_steps("graduation")
