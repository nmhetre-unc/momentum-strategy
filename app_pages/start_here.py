"""
Start here: the onboarding page.

The front door for someone who has never used this platform. Its job is
to answer four questions fast -- what is this, what will I be able to do,
what order do I go in, and what should I click first -- and then get out
of the way.

Deliberately computes nothing. It reads no prices and fits no models, so
it renders instantly whether or not data has loaded, and works offline.
The checklist state is the only thing it writes.
"""

import streamlit as st

from quant_notes import LEARNING_PATH, QUANT_NOTES
from regime_dashboard import (caveat, chart_caption, common_mistakes, explainer,
    how_to_read, next_steps, page_intro, quant_note, table_caption
)
st.markdown(
    "#### Welcome — this is a flight simulator for quant research.\n"
    "You will build trading strategies, discover that most of them don't work, and learn to "
    "tell the difference *before* anyone else has to point it out. That last skill is the "
    "entire point."
)

page_intro("start_here")
common_mistakes("start_here")

# --------------------------------------------------------------------------
# What this is
# --------------------------------------------------------------------------
with st.container(border=True):
    intro_left, intro_right = st.columns(2)
    with intro_left:
        st.markdown(
            "**What this platform is**\n\n"
            "A working multi-strategy backtesting engine — rule-based, machine-learning and "
            "regime-adaptive — wrapped in a training environment. Everything runs on real "
            "market data you choose in the sidebar.\n\n"
            "It is built for **exposure to the work**, not production trading. Every number "
            "comes from free daily data and a simple cost model. What transfers is the "
            "reasoning, not the strategies."
        )
    with intro_right:
        st.markdown(
            "**What it teaches: habits, not tricks**\n\n"
            "Nobody needs another moving-average crossover. What separates a quant researcher "
            "from someone who runs backtests is a set of reflexes:\n\n"
            "*Check the sample size before believing a Sharpe ratio. Ask whether a number is "
            "in-sample or out-of-sample. Compare against buy-and-hold. Count every comparison "
            "you ran, including the ones you threw away.*\n\n"
            "Those work on any dataset, in any language, for the rest of your career."
        )

    st.markdown(
        "**The core mindset — one sentence, and everything here serves it**\n\n"
        "> **Assume your result is wrong until you have tried to break it.**\n\n"
        "This is not pessimism. It is the thing that makes research trustworthy, and it is what "
        "an interviewer is actually probing for when they ask about your project."
    )

    st.markdown("**How the pages fit together**")
    st.markdown(
        """
```mermaid
flowchart TD
    S([Start here]) --> B["1 · Backtest<br/>equity curves, drawdown, benchmark"]
    B --> R["2 · Regimes<br/>volatility clustering, market structure"]
    R --> A["3 · Adaptive<br/>filtering, switching, sizing"]
    A --> M["4 · ML lab<br/>overfitting, base rates"]
    M --> V["5 · Validation<br/>in-sample to out-of-sample decay"]
    V --> X["6 · Exercises<br/>practice with automated checks"]
    L["Learn<br/>notes · pitfalls · glossary"]
    L -.->|"reference at every step"| B
    X -.->|"deepen"| L
```
"""
    )
    st.caption(
        "Five simulator pages in sequence, one practice page, and **Learn** running alongside "
        "the whole way as the reference layer."
    )

    st.success(
        "**By the end of the path you will be able to:** read an equity curve and say what its "
        "shape means; identify the market regime a strategy depends on; explain why a random "
        "forest beat logistic regression on training data and lost to it on real data; run "
        "walk-forward validation and interpret the gap; measure lookahead bias as a number on "
        "your own data; and write up a result that leads with its limitation — which is the "
        "version that convinces people.",
        icon=":material/school:",
    )

# --------------------------------------------------------------------------
# How to use the platform
# --------------------------------------------------------------------------
with st.expander("How to read this page", icon=":material/map:"):
    st.markdown(
        """
**Move through the pages in order.** Each one assumes the habits from the last. Regimes only
makes sense once you've seen a strategy's performance vary; Adaptive only makes sense once
you can read a per-regime table. Skipping ahead turns later pages into trivia rather than
answers to problems you've hit.

**Keep Learn open alongside.** Every page surfaces the relevant notes in context, but when a
number confuses you, go to **Learn → Quant notes**, filter by theme, read the one note that
covers it, and come straight back. Each note names the page it applies to. Do not read the
notes front to back — they are reference, and reading them cold produces recognition without
fluency.

**Use each simulator page for one thing:**

- **Backtest** — where you learn to *read* a result. Equity curve shape, drawdown depth and
  duration, exposure, turnover, and the buy-and-hold comparison you must never omit.
- **Regimes** — where you stop treating the market as one thing. Detect regimes, check they
  are actually regimes (persistent, distinct, recurring), and see which environment your
  strategy's return actually came from.
- **Adaptive** — where you try to *act* on regimes: filter out bad ones, switch strategies,
  resize positions. Also where you learn that the simplest mechanism usually wins.
- **ML lab** — where you watch overfitting happen instead of reading about it. Toggle between
  the two models and watch the train/test gap open.
- **Validation** — where you find out whether any of it was real. Run this before trusting
  *any* result, including your favourites. Especially your favourites.
- **Exercises** — ten guided exercises with automated checks against your own loaded data.
  Use these to reinforce a concept right after meeting it, while it's still uncomfortable.

**A rule worth adopting on day one.** Decide what would count as success *before* you look at
the result. "I'll accept this if out-of-sample Sharpe beats 0.5 with at least 30 trades" is a
commitment. Deciding after you've seen the number is how honest people fool themselves.
"""
    )

# --------------------------------------------------------------------------
# Roadmap
# --------------------------------------------------------------------------
st.subheader("Your roadmap", divider="gray")
st.caption("Seven stops. Each one names why it matters — read that before clicking through.")

ROADMAP = [
    ("app_pages/backtest_lab.py", "Backtest", ":material/query_stats:",
     "Equity curves, drawdowns, exposure, turnover, and the benchmark comparison.",
     "Because every later page produces numbers in this format. If you can't read an equity "
     "curve — its shape, its flat stretches, its drawdowns — nothing downstream will mean "
     "anything. This is also where you learn that most strategies lose to buy-and-hold."),
    ("app_pages/regimes.py", "Regimes", ":material/layers:",
     "Volatility clustering, market structure, and performance split by environment.",
     "Because a single full-period Sharpe averages the conditions where a strategy thrived "
     "with the ones where it bled, and describes neither. Knowing *where* a return came from "
     "is what turns a number into something you can act on."),
    ("app_pages/adaptive_lab.py", "Adaptive", ":material/tune:",
     "Filtering, switching, re-parameterizing and position sizing.",
     "Because knowing a strategy fails in one regime raises an obvious question: can you just "
     "not trade it there? Sometimes. This page teaches you to check whether the improvement "
     "paid for the turnover it added — and it usually hasn't."),
    ("app_pages/ml_lab.py", "ML lab", ":material/network_intelligence:",
     "Base rates, the train/test gap, and why accuracy is not profit.",
     "Because ML is where overfitting is most visible and most instructive. Watching a model "
     "hit 86% on training data and 45% on real data teaches more about model capacity than "
     "any textbook chapter."),
    ("app_pages/validation.py", "Validation", ":material/fact_check:",
     "In-sample vs out-of-sample decay, rolling walk-forward, and fair comparison.",
     "Because everything before this point can be fooled by a lucky sample. This is the page "
     "that tells you whether you found something or fitted something — and the gap between "
     "the two numbers matters more than either one."),
    ("app_pages/exercises_lab.py", "Exercises", ":material/assignment:",
     "Ten guided exercises with automated checks against your own data.",
     "Because reading about the effect and measuring it yourself are different experiences. "
     "Each exercise installs one specific habit, and the check runs on whatever ticker you "
     "have loaded."),
    ("app_pages/learn.py", "Learn", ":material/menu_book:",
     "The learning path, the three pitfalls, 35 quant notes, and the glossary.",
     "Because this is the reference layer for everything above — and the pitfalls page is "
     "worth re-reading after every result that looks good. That second reading is the one "
     "that catches things."),
]

for index, (path, label, icon, what, why) in enumerate(ROADMAP, start=1):
    with st.container(border=True):
        header = st.columns([3, 2])
        with header[0]:
            st.page_link(path, label=f"**{index} · {label}**", icon=icon)
        with header[1]:
            st.caption("Reference, use throughout" if label == "Learn" else f"Step {index} of 6")
        st.markdown(what)
        st.caption(f"**Why this matters:** {why}")

how_to_read(
    """
- **Steps 1-6 are a sequence; Learn is a companion.** Don't try to finish Learn before
  starting — it is written to be dipped into when something confuses you.
- **Each step takes an afternoon, not ten minutes.** The goal is the habit, and habits form
  by doing the thing several times on different data.
- **You are not expected to build a profitable strategy.** Most attempts here will lose to
  buy-and-hold. Recognizing that quickly and honestly is the skill being trained.
""",
    title="How to use this roadmap",
)

# --------------------------------------------------------------------------
# Key concepts
# --------------------------------------------------------------------------
st.subheader("Key concepts you'll learn", divider="gray")
st.caption(
    "Five ideas that recur on every page. Skim them now for orientation, then come back when "
    "you meet each one for real — that second reading is when they land."
)

quant_note("equity_curve")
quant_note("lookahead_bias")
quant_note("overfitting")
quant_note("regime_drift")
quant_note("is_vs_oos")

how_to_read(
    """
- **These five explain most of what goes wrong.** Equity curves are how you read a result;
  the other four are the four ways a result can be a lie.
- **You are not meant to absorb them now.** Skim for the shape of each idea. They are
  surfaced again, in context, on the exact page where each one bites.
- **All 35 notes live in Learn → Quant notes**, grouped by theme and filterable.
""",
    title="How to use these",
)

# --------------------------------------------------------------------------
# Metaphors
# --------------------------------------------------------------------------
st.subheader("Five ideas, in plain language", divider="gray")
st.caption("If the formal definitions above felt dense, start here instead.")

explainer(
    "What backtesting is",
    "reading the diary of a strategy — a day-by-day account of what it would have done, "
    "written after the fact by someone who already knows how the story ends.",
    """
A backtest replays history and records what a set of rules would have done each day. That is
genuinely useful: it is the only way to see a strategy's behaviour over decades in seconds.

But notice who is writing the diary. *You* already know 2020 crashed and 2021 rallied. Every
choice you make — which ticker, which dates, which parameters, which strategy to test at all
— is made by someone with that knowledge. The diary is honest about what the rules did; it
cannot be honest about why you chose those rules.

That gap is why every other page exists. The backtest tells you what happened. Validation
tells you whether to believe it.
""",
)

explainer(
    "What a market regime is",
    "weather patterns for the market — you can't forecast next month's weather, but knowing "
    "it's hurricane season changes what you build today.",
    """
A regime is a persistent market environment with its own character: its own typical
volatility, its own tendency to trend or reverse. Calm bull markets, directionless chop and
violent selloffs are different regimes, and they last weeks to months.

The analogy holds in the ways that matter. Regimes **persist**, which is what makes them
worth detecting — a state that changed daily would be as useless as an hourly forecast. And
they support **conditioning, not forecasting**: knowing it's hurricane season doesn't tell you
when the storm hits, but it changes how you prepare.

Where it breaks: weather runs on physics that don't change. Markets are made by participants
who adapt, so the regimes themselves drift over decades.
""",
)

explainer(
    "What adaptive strategies do",
    "changing your driving style for the weather — not changing the car, and not changing "
    "where you're going.",
    """
An adaptive strategy keeps the underlying rule and changes *how* it is applied depending on
conditions. Four ways, in rough order of how much can go wrong:

- **Filtering** — don't drive in the blizzard. Safe, because the worst case is a trip you
  didn't take.
- **Switching** — use rally technique on gravel and defensive technique in traffic. Powerful
  if you identify the surface correctly, and expensive because you change everything at once.
- **Re-parameterizing** — retune the suspension for each road type. Subtle, and every road
  gives you fresh dials to fiddle with.
- **Position sizing** — just slow down on ice. No classification needed, only a measurement
  of how slick it is right now.

The last one usually wins, which surprises people. It leans on volatility, which is
forecastable, rather than direction, which isn't.
""",
)

explainer(
    "What the ML layer is really doing",
    "trying to predict coin flips from noisy clues — and a powerful model will always find "
    "patterns in the noise, because that is what power is for.",
    """
The model here predicts whether tomorrow closes higher than today. On daily equity data that
is close to a coin flip with a slight upward tilt: about 53% of days are up.

So the model is looking for a very small edge inside enormous noise. Hand a random forest
twelve features and 1,700 rows of that, and it will find patterns — there are patterns in any
1,700 rows of noise. It reports high confidence about every one, scores 86% on the data it
learned, and collapses to 45% on data it hasn't seen.

That is not a bug and not a tuning problem. It is what capacity does when there is nothing
real to capture. The counterintuitive lesson: **the less signal there is, the simpler your
model should be.**
""",
)

explainer(
    "What validation is",
    "testing your strategy on new terrain — not the course it trained on.",
    """
Any strategy can be made to look good on a fixed history. You have parameters, the history is
finite, and *some* setting was necessarily the best on it. Finding that setting and reporting
its performance reports the maximum of a search, not the expected performance of a process.

Validation cuts history in two. The first part is where you build; the second part your
strategy has never seen. Only the second number counts — and what you should look at is the
**gap** between them, because the gap measures how much of the performance was specific to
the course you trained on.

A small gap at a modest level beats a large level with a large gap, every time. The first
describes a process you can repeat.
""",
)

# --------------------------------------------------------------------------
# Beginner traps
# --------------------------------------------------------------------------
st.subheader("Common beginner traps", divider="gray")
st.markdown(
    "Six warning signs worth recognizing on day one. **The platform checks all of them for "
    "you** — every page fires a warning when your own numbers cross these thresholds — but "
    "knowing what they mean turns those warnings from noise into information."
)

caveat(
    "**A Sharpe ratio above 2 on daily equity data.** This is the single loudest alarm in "
    "quant research. It almost always means lookahead bias — a decision using information "
    "that didn't exist yet — rather than genuine edge. When you see it, go looking for the "
    "leak before celebrating. *Checked on: Backtest, Learn.*"
)
caveat(
    "**Fewer than 20 trades.** A backtest covering ten years but containing three trades has "
    "given you three independent bets. Its Sharpe ratio, win rate and drawdown are describing "
    "that handful of events, however many days the chart covers. *Checked on: Backtest, "
    "Validation, Exercises.*",
    level="info",
)
caveat(
    "**Regime labels fitted on the full history.** If a regime model is fitted on all your "
    "data and then used to backtest over that same data, the labels encode the future — your "
    "2015 'calm regime' label was computed partly from 2020. The backtest will look wonderful "
    "and the strategy will not work. *Checked on: Regimes, Backtest.*"
)
caveat(
    "**ML accuracy below the base rate.** About 53% of days are up, so predicting 'up' every "
    "day scores 53% with no model at all. A model at 51% has been beaten by a one-line "
    "constant. Always read accuracy next to its base rate — alone it means nothing. "
    "*Checked on: ML lab, Exercises.*",
    level="info",
)
caveat(
    "**An adaptive strategy invested only 20% of the time.** Reducing exposure shrinks "
    "drawdown automatically — that is arithmetic, not skill. If Sharpe improved, the strategy "
    "avoided the market at the right moments. If only drawdown improved, it was simply out of "
    "the market, and trading the original at half size would have done the same. "
    "*Checked on: Adaptive, Learn.*",
    level="info",
)
caveat(
    "**Walk-forward folds that swing wildly.** If out-of-sample Sharpe ranges from −1.5 to "
    "+2.0 across test windows, that spread *is* your uncertainty about the strategy. Quoting "
    "a single average to two decimal places is false precision. *Checked on: Validation.*",
    level="info",
)

how_to_read(
    """
- **Every one of these has a tell you can check in seconds.** Too-good Sharpe, too-few
  trades, non-causal labels, accuracy below base rate, tiny exposure, wild fold spread.
- **The platform will warn you, but the warnings only help if you know what they mean.**
  That is what this section is for.
- **Read Learn → Common pitfalls after your first good-looking result.** Recognizing these in
  work you've already done is far easier than avoiding them in the abstract.
""",
    title="How to use these warnings",
)

# --------------------------------------------------------------------------
# Checklist
# --------------------------------------------------------------------------
st.subheader("Your first session — a checklist", divider="gray")
st.caption(
    "Eight concrete things to do, in order. Tick them off as you go; the list remembers your "
    "progress for this session."
)

CHECKLIST = [
    ("chk_buy_hold", "Run buy-and-hold on SPY and note its CAGR, Sharpe and max drawdown",
     "app_pages/backtest_lab.py", "Backtest",
     "Every strategy you build competes with these three numbers. Learn them first."),
    ("chk_equity", "Read your first equity curve — shape, flat stretches, drawdowns",
     "app_pages/backtest_lab.py", "Backtest",
     "Resist reading the endpoint first; it is the least informative part of the chart."),
    ("chk_regimes", "Detect your first regimes and check they're actually regimes",
     "app_pages/regimes.py", "Regimes",
     "Persistent, statistically distinct, recurring. If not all three, there's nothing to condition on."),
    ("chk_adaptive", "Run your first adaptive strategy and compare it to the plain version",
     "app_pages/adaptive_lab.py", "Adaptive",
     "Ask whether the improvement paid for the turnover it added."),
    ("chk_ml", "Train your first ML model — both types, back to back",
     "app_pages/ml_lab.py", "ML lab",
     "Watch the train/test gap. The more accurate model is usually the worse one."),
    ("chk_walkforward", "Run your first walk-forward validation and read the gap",
     "app_pages/validation.py", "Validation",
     "The gap between in-sample and out-of-sample matters more than either number."),
    ("chk_exercise", "Complete your first exercise and write your own explanation",
     "app_pages/exercises_lab.py", "Exercises",
     "Write the sentence before revealing the answer. The gap between them is the lesson."),
    ("chk_note", "Read your first quant note in full",
     "app_pages/learn.py", "Learn",
     "Start with 'How to read an equity curve' or 'Lookahead bias'."),
]

completed = sum(1 for key, *_ in CHECKLIST if st.session_state.get(key, False))
st.progress(completed / len(CHECKLIST), text=f"{completed} of {len(CHECKLIST)} done")

for key, task, path, label, hint in CHECKLIST:
    with st.container(border=True):
        row = st.columns([6, 2])
        with row[0]:
            # persist_state="session" is essential here: without it Streamlit
            # discards widget state for widgets that weren't rendered, so
            # ticking an item and then visiting the page it links to would
            # silently reset the whole checklist.
            st.checkbox(
                task, key=key, persist_state="session",
                help="Ticks are remembered for this session, including after you visit the "
                     "linked page and come back.",
            )
            st.caption(hint)
        with row[1]:
            st.page_link(path, label=label, icon=":material/arrow_forward:")

if completed == len(CHECKLIST):
    st.success(
        "All eight done. You have now touched every part of the platform — the next step is "
        "depth rather than coverage: work the eight-stage learning path on the Learn page, "
        "which asks harder questions about the same tools.",
        icon=":material/celebration:",
    )

with st.expander("After the checklist: the full learning path", icon=":material/route:"):
    st.markdown(
        "The checklist gets you around the building. The **learning path** on the Learn page "
        "is the actual curriculum — eight stages, each with a *done-when* condition phrased as "
        "a habit rather than a task:"
    )
    for stage in LEARNING_PATH:
        st.markdown(f"- **{stage['stage']}** — {stage['goal']}  \n  _Done when: {stage['done_when']}_")

# --------------------------------------------------------------------------
# What to do next

# --------------------------------------------------------------------------
# Where to go next
# --------------------------------------------------------------------------
next_steps("start_here")
