"""
How to think like a quant: the reference material behind everything else here.

This page is the only one that doesn't compute a strategy. It's the
reference layer: the learning path, the three pitfalls, all the quant
notes the other pages surface contextually, and the glossary behind every
tooltip. Everything here is read from shared data structures, so a note
added for one page appears here automatically.

The one optional computation is the health check -- an opt-in pass that
tests the six warning thresholds taught here against whatever data is
currently loaded, so the thresholds land as facts about your own results
rather than as numbers to memorize.
"""

import numpy as np
import pandas as pd
import streamlit as st

from adaptive import ADAPTIVE_DOCS
from quant_notes import LEARNING_PATH, METRIC_DOCS, PITFALLS, QUANT_NOTES
from regime import REGIME_METHOD_DOCS, SMOOTHING_DOCS, regime_stability
from regime_dashboard import (cached_regimes, caveat, chart_caption, common_mistakes,
    explainer, how_to_read, next_steps, page_intro, quant_note, table_caption
)
from regime_features import FEATURE_DOCS
from strategies import STRATEGY_DOCS

# --------------------------------------------------------------------------
# Note organisation
# --------------------------------------------------------------------------
# 34 notes in one flat list is not navigable. Grouping them by theme, and
# recording which page each one supports, turns the tab into a reference
# you can actually search. Anything unmapped falls into "Other" rather than
# disappearing, so adding a note elsewhere never silently breaks this page.
NOTE_THEMES = {
    "Reading results": [
        "equity_curve", "drawdown_vs_return", "sharpe_can_mislead",
        "exposure_caveat", "turnover_costs", "costs",
    ],
    "Strategy design": ["trend_in_chop", "trend_vs_chop"],
    "Validation": [
        "walkforward_reason", "walk_forward", "is_vs_oos",
        "fold_uncertainty", "fair_comparison",
    ],
    "Bias & leakage": [
        "lookahead_bias", "regime_lookahead", "silent_fitting",
        "survivorship_bias", "regime_drift",
    ],
    "Regimes": [
        "regimes", "regime_volatility_clusters",
        "regime_transition_persistence", "risk_by_regime",
    ],
    "Machine learning": [
        "ml_base_rate", "ml_overfitting", "bias_variance",
        "ml_accuracy_vs_pnl", "ml_feature_importance", "ml_regime_conditional",
    ],
    "Adaptive": [
        "adaptive", "adaptive_filtering", "adaptive_switching",
        "adaptive_overfitting", "volatility_targeting", "position_sizing",
    ],
}

NOTE_PAGES = {
    "Reading results": ("app_pages/backtest_lab.py", "Backtest"),
    "Strategy design": ("app_pages/backtest_lab.py", "Backtest"),
    "Validation": ("app_pages/validation.py", "Validation"),
    "Bias & leakage": ("app_pages/regimes.py", "Regimes"),
    "Regimes": ("app_pages/regimes.py", "Regimes"),
    "Machine learning": ("app_pages/ml_lab.py", "ML lab"),
    "Adaptive": ("app_pages/adaptive_lab.py", "Adaptive"),
}

_mapped = {key for keys in NOTE_THEMES.values() for key in keys}
_unmapped = [key for key in QUANT_NOTES if key not in _mapped]
if _unmapped:
    NOTE_THEMES["Other"] = _unmapped

THEME_OF_NOTE = {key: theme for theme, keys in NOTE_THEMES.items() for key in keys}


# --------------------------------------------------------------------------
# Optional health check: the taught thresholds, measured on your own data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl="1h", max_entries=5)
def health_check(prices: pd.DataFrame, settings: dict) -> dict:
    """
    Evaluates the six warning thresholds this page teaches against the
    currently loaded data. Deliberately opt-in: it fits a regime model, a
    classifier and a rolling walk-forward, which is far too much work to
    run on a reference page nobody asked to compute anything on.
    """
    from adaptive import regime_filtered
    from analytics import full_report
    from backtest import run_backtest
    from ml_strategy import model_report
    from strategies import STRATEGIES
    from walk_forward import rolling_walk_forward

    out = {}
    signal = STRATEGIES["sma_crossover"](prices)
    stats = full_report(run_backtest(prices, signal, cost_bps=5))
    out["sharpe"] = stats["sharpe_ratio"]
    out["num_trades"] = stats["num_trades"]

    try:
        regimes = cached_regimes(
            prices, settings["method"], settings["n_regimes"], settings["fit_frac"],
            settings["smooth"], settings["min_duration"], settings["decode"],
            settings["walk_forward"],
        )
        out["regime_duration"] = regime_stability(regimes.labels)["avg_duration"]
        adapted = regime_filtered(prices, base="sma_crossover", regimes=regimes)
        out["adaptive_exposure"] = full_report(run_backtest(prices, adapted, cost_bps=5))["exposure"]
    except Exception:
        out["regime_duration"] = None
        out["adaptive_exposure"] = None

    try:
        report = model_report(prices, model_type="random_forest")
        out["ml_accuracy"] = report["test_accuracy"]
        out["ml_base_rate"] = report["test_base_rate"]
    except Exception:
        out["ml_accuracy"] = out["ml_base_rate"] = None

    try:
        out["fold_std"] = rolling_walk_forward(prices, STRATEGIES["sma_crossover"], cost_bps=5)["sharpe_std"]
    except Exception:
        out["fold_std"] = None
    return out


def diagnostic_caveat(diagnostics: dict, check: str):
    """Fires one threshold warning, if the health check was run and it applies."""
    if not diagnostics:
        return
    value = diagnostics.get(check)

    if check == "sharpe" and value is not None and value > 2.0:
        caveat(
            f"**Your current setup shows a Sharpe of {value:.2f}.** Above 2 on a daily equity "
            f"strategy, assume a lookahead leak and go looking for it — that is far more often "
            f"the explanation than genuine edge."
        )
    elif check == "num_trades" and value is not None and value < 20:
        caveat(
            f"**Your current setup produced {int(value)} trades.** Every metric you read from it "
            f"describes that many independent bets, no matter how many days the backtest covers.",
            level="info",
        )
    elif check == "regime_duration" and value is not None and value < 15:
        caveat(
            f"**Your regime episodes average {value:.0f} days.** Real regimes last weeks to "
            f"months — at this length the labels are flickering rather than describing states. "
            f"Raise the confirmation window in the sidebar."
        )
    elif check == "adaptive_exposure" and value is not None and value < 0.30:
        caveat(
            f"**Regime filtering currently leaves you {value:.0%} invested.** Its risk metrics "
            f"are computed on that slice of days, and any drawdown improvement is partly the "
            f"arithmetic of holding less rather than skill at choosing when.",
            level="info",
        )
    elif check == "ml_accuracy":
        accuracy, base = diagnostics.get("ml_accuracy"), diagnostics.get("ml_base_rate")
        if accuracy is not None and base is not None and accuracy <= base:
            caveat(
                f"**On your data the random forest scores {accuracy:.1%} against a base rate of "
                f"{base:.1%}.** It has been beaten by always predicting the majority class, which "
                f"is the honest outcome rather than a broken model."
            )
    elif check == "fold_std" and value is not None and value > 1.0:
        caveat(
            f"**Your walk-forward folds have a Sharpe standard deviation of {value:.2f}.** The "
            f"result swings widely between periods — that spread is your real uncertainty, and it "
            f"is much larger than any single split suggests.",
            level="info",
        )


# --------------------------------------------------------------------------
# Onboarding
# --------------------------------------------------------------------------
st.markdown(
    "This platform is built to teach one habit above all the others: **assume your result is "
    "wrong until you have tried to break it.** Everything below is in service of that."
)

page_intro("learn")
common_mistakes("learn")

with st.container(border=True):
    st.markdown(
        "#### New here? What this tab is for\n"
        "Every other page *does* something — runs a backtest, fits a model, detects regimes. "
        "This one explains **why** those things are done the way they are. It is the reference "
        "layer, and the place to come back to when a number on another page doesn't make sense."
    )

    intro_left, intro_right = st.columns(2)
    with intro_left:
        st.markdown(
            "**It teaches habits, not tasks**\n\n"
            "Nothing here is a checklist to complete. The content is aimed at reflexes a quant "
            "researcher uses without thinking: *check the sample size before believing a Sharpe "
            "ratio; ask whether a number is in-sample or out-of-sample; compare against "
            "buy-and-hold; count every comparison you ran, including the ones you discarded.*\n\n"
            "Those transfer to any dataset and any language. The specific strategies here do not."
        )
    with intro_right:
        st.markdown(
            "**How it connects to the rest of the terminal**\n\n"
            "| Page | What this tab explains about it |\n|---|---|\n"
            "| **Backtest** | How to read an equity curve, drawdown, exposure, turnover |\n"
            "| **Regimes** | Why volatility is the primary axis; what a real regime looks like |\n"
            "| **Adaptive** | The four mechanisms and how each one fails |\n"
            "| **ML lab** | Base rates, bias/variance, why accuracy isn't profit |\n"
            "| **Validation** | In-sample vs out-of-sample, folds, silent fitting |\n"
            "| **Exercises** | The habit each exercise is designed to install |\n"
        )

    st.markdown(
        "**The four sections below**\n\n"
        "**Learning path** — eight stages in order, each with a *done-when* condition. Start "
        "here if you're new. · **Common pitfalls** — the three ways to produce a backtest that "
        "is confidently wrong. Re-read after every project. · **Quant notes** — all "
        f"{len(QUANT_NOTES)} explanations the other pages surface contextually, grouped by theme "
        "and searchable. · **Glossary** — every metric, strategy, mechanism and feature, with "
        "the same text as the tooltips elsewhere."
    )

with st.expander("How to read this page", icon=":material/map:"):
    st.markdown(
        """
**Progressing through the learning path.** Work the stages in order — each assumes the
habits from the one before. Stage 3 (validation) only makes sense once you've built
something in stage 2 that you might be tempted to believe. Don't rush: a stage takes an
afternoon, not ten minutes.

**Reading the "done-when" conditions.** These are deliberately *not* "you have run X". They
describe a change in how you behave: *"you instinctively ask 'in-sample or out-of-sample?'
about any performance number, including your own."* You can complete every task on this page
without satisfying a single done-when, and the done-when is the part that matters. Judge
yourself honestly — nobody else is checking.

**Using quant notes and the glossary.** These are reference, not reading. Don't work through
them front to back. Come here when a specific number confuses you, read the one note that
covers it, and go back to the page you came from. The theme filter and the page column exist
to make that round trip fast.

**Revisiting the pitfalls.** Read them once now, then again *after* your first result that
looks good. The second reading is the useful one: lookahead bias and silent fitting are far
easier to recognize in work you've already done than to avoid in the abstract.

**Connecting lessons back to the simulator.** Every note names the page it applies to. The
fastest way to learn any of this is to read a note, open that page, and go looking for the
thing it describes in your own numbers — which is what the health check below does
automatically for the six warning thresholds.
"""
    )

# --------------------------------------------------------------------------
# Optional: measure the taught thresholds against the loaded data
# --------------------------------------------------------------------------
prices = st.session_state.get("prices")
diagnostics = None

with st.container(border=True):
    st.markdown("**Check these lessons against your own data**")
    if prices is None or prices.empty:
        st.caption(
            "Load a ticker in the sidebar to have the six warning thresholds taught here "
            "measured against your own setup, rather than read as numbers to memorize."
        )
    else:
        run_check = st.toggle(
            "Measure the warning thresholds on my currently loaded data", value=False,
            key="learn_health",
            help=(
                "Fits a regime model, a classifier and a rolling walk-forward on the loaded "
                "ticker, then flags which of the taught thresholds your own setup crosses. "
                "Takes a few seconds the first time, then cached."
            ),
        )
        if run_check:
            with st.spinner("Measuring..."):
                diagnostics = health_check(prices, st.session_state["regime_settings"])
            st.caption(
                f"Measured on **{st.session_state['ticker']}**, "
                f"{prices.index[0].date()} to {prices.index[-1].date()}. Warnings appear beside "
                f"the learning-path stage and glossary entry they belong to."
            )
        else:
            st.caption(
                "Off by default — this page is reference material and should stay instant. "
                "Turn it on to see which thresholds your current setup actually crosses."
            )

# Keyed so the selected tab survives a rerun rather than bouncing back to the first.
tab_path, tab_pitfalls, tab_notes, tab_glossary = st.tabs(
    ["Learning path", "Common pitfalls", "Quant notes", "Glossary"],
    on_change="rerun", key="learn_tabs",
)

# --------------------------------------------------------------------------
with tab_path:
    st.markdown(
        "Work through these in order. Each stage has a **done-when** condition that is about a "
        "habit rather than a completed task — the habit is the transferable part."
    )

    explainer(
        "What this path actually is",
        "a quant apprenticeship — you learn by producing results, having them questioned, and "
        "learning to question them yourself before anyone else does.",
        """
Trading research is not taught by lecture, because the skill is not knowledge. It's a
disposition: reflexive scepticism about your own output, applied fastest to results you like.

An apprenticeship works by putting you in front of real work with someone experienced asking
the awkward questions — *is that in-sample? how many trades? what's the benchmark? how many
things did you try before this one?* Eventually you ask them before they do, and at that point
you're a researcher rather than someone who runs backtests.

This path compresses that. Each stage produces something, then hands you the question that
undermines it. Stage 2 gets you a strategy that works; stage 3 shows you it might not. Stage 4
gets you an ML model; the same stage shows you it memorized noise. That sequence is not
discouragement — it is the actual job, and being comfortable with it is what separates
research from marketing.
""",
    )

    # Warning thresholds attached to the stage where each one is taught.
    stage_checks = {
        "3. Validation": ("num_trades", "fold_std"),
        "4. The ML layer": ("ml_accuracy",),
        "5. Regimes": ("regime_duration",),
        "7. Adaptive strategies": ("adaptive_exposure",),
        "8. Write it up": ("sharpe",),
    }

    for stage in LEARNING_PATH:
        with st.container(border=True):
            st.markdown(f"**{stage['stage']} — {stage['goal']}**")
            st.markdown(stage["do"])
            st.caption(f"Done when: {stage['done_when']}")

            for check in stage_checks.get(stage["stage"], ()):
                diagnostic_caveat(diagnostics, check)

    quant_note("is_vs_oos")
    quant_note("adaptive")
    how_to_read(
        """
- **Each stage installs one habit, and they compound.** Stage 1 gives you a benchmark to
  compare against; stage 3 gives you the reflex to distrust the comparison; stage 6 gives you
  the reflex to ask when each input became knowable. Skipping ahead means later stages land as
  trivia rather than as answers to problems you've actually hit.
- **The done-when conditions are the assessment.** They are all behavioural, and none of them
  can be completed by running something. If you can't honestly say you've reached one, do the
  stage again on a different ticker.
- **Stage 8 is not optional decoration.** Writing a result up — hypothesis, method,
  out-of-sample number, what would falsify it — is where you discover which parts you don't
  actually understand. It is also the form the job takes.
"""
    )

# --------------------------------------------------------------------------
with tab_pitfalls:
    st.markdown(
        "Three ways to produce a backtest that is confidently wrong. They are listed in the order "
        "people tend to hit them, and the third one catches people who have already learned to "
        "avoid the first two."
    )

    PITFALL_NOTES = {
        "Lookahead bias": ("lookahead_bias", "Grading yourself with tomorrow's answers",
                           "grading your own exam with tomorrow's answer key — you score brilliantly "
                           "and learn nothing about whether you can pass.",
                           """
Lookahead bias means a decision used information that did not exist when the decision was
made. The reason it is so dangerous is that nothing looks wrong: the code is clean, the model
is standard, and the equity curve is beautiful. The only way to find it is to ask, of every
input, *what date was this knowable?*

The subtle version is fitting anything on the full sample — a scaler, a PCA, a percentile
rank, a regime model — and then backtesting over that same sample. The cluster centres, or
the sample mean, encode the future. Your 2015 label was computed partly from 2020.
"""),
        "Overfitting": ("ml_overfitting", "Memorizing the practice test",
                        "memorizing the practice test instead of learning the subject — perfect on "
                        "the questions you've seen, lost on any others.",
                        """
A flexible model handed a low-signal problem will memorize the training rows, because there
are always patterns in any 1,700 rows of noise and finding them is exactly what capacity is
for. It reports high confidence about every one.

The tell is the gap: 86% on the practice test, 45% on the real one. And note the direction of
the correct response — with this little signal, you want a *less* flexible model, not a
better-tuned one. That inverts most people's instinct, which is to reach for more capacity
when results disappoint.

The same thing happens without any model at all: trying twenty parameter sets and reporting
the best is memorizing the practice test by hand.
"""),
        "Regime drift": ("regime_drift", "The market changing its language",
                         "the market changing its language mid-conversation — your vocabulary was "
                         "correct, and it stopped being the language being spoken.",
                         """
This one catches people who have already learned to avoid the first two. You did everything
right: clean split, honest holdout, no leakage. And the strategy still stops working, because
the relationship it relied on was real and then ended.

A model trained on 2010–2019 learned one long low-volatility bull market. Its idea of "high
volatility" would be classified as calm by a model that had seen 2008. When conditions change,
the vocabulary is still internally consistent and no longer describes anything.

A single walk-forward split cannot show you this — you need the fold *sequence*. Consistently
positive early and consistently negative late is drift, and averaging across it produces a
number describing a market that no longer exists.
"""),
    }

    for name, pitfall in PITFALLS.items():
        with st.container(border=True):
            st.markdown(f"**{name}** — {pitfall['summary']}")
            st.markdown("**Where it hides**")
            for place in pitfall["where_it_hides"]:
                st.markdown(f"- {place}")
            st.markdown(f"**The tell:** {pitfall['tell']}")
            st.markdown(f"**The fix:** {pitfall['fix']}")

            if name in PITFALL_NOTES:
                note_key, title, metaphor, body = PITFALL_NOTES[name]
                explainer(title, metaphor, body)
                quant_note(note_key)
                if name == "Lookahead bias":
                    diagnostic_caveat(diagnostics, "sharpe")

    how_to_read(
        """
- **Spot them by their tells, not by intention.** Nobody sets out to leak data. Lookahead
  announces itself as an implausibly high Sharpe; overfitting as a large in-sample to
  out-of-sample gap; drift as folds that decay in sequence. Learn the three tells and you catch
  most of it.
- **Check your own results against all three, in order.** Is the Sharpe too good? Is the gap
  large? Do the folds trend down? Three questions, two minutes, and they cover the majority of
  what goes wrong.
- **The third one is the one that gets experienced people.** You can do everything correctly
  and still be describing a market that has moved on. That is not a mistake to avoid — it is a
  reason to keep re-validating rather than trusting a result indefinitely.
"""
    )

# --------------------------------------------------------------------------
with tab_notes:
    st.markdown(
        f"All {len(QUANT_NOTES)} notes the other pages surface in context, grouped by theme. "
        "These are reference — come here when a specific number confuses you, read the one note "
        "that covers it, and go back."
    )

    explainer(
        "Two ideas that run through most of these notes",
        "validation is testing your strategy on new terrain rather than the course you trained "
        "on; adaptation is changing your driving style for the weather rather than the car.",
        """
**Validation as new terrain.** A strategy tuned on one stretch of history has, in effect,
memorized that course. The question is never "how fast did it go on the course it learned"
but "how does it handle ground it has never seen". That is why the *gap* between in-sample
and out-of-sample matters more than either number: the gap measures how much of the
performance was course-specific.

**Adaptation as driving style.** The four adaptive mechanisms are all versions of "behave
differently depending on conditions". Filtering is staying home in the storm; switching is
changing technique for the surface; re-parameterizing is retuning the car per road; volatility
targeting is simply slowing down on ice. They differ enormously in how much can go wrong —
and the simplest, slowing down, is usually the one that works.
""",
    )

    themes = st.pills(
        "Theme", list(NOTE_THEMES), selection_mode="multi",
        default=list(NOTE_THEMES), key="learn_note_themes",
        help="Narrow the list to the area you are working in. Deselecting everything shows all notes.",
    )
    active_themes = themes or list(NOTE_THEMES)
    available = [k for theme in active_themes for k in NOTE_THEMES[theme] if k in QUANT_NOTES]

    if not available:
        st.info("No notes in the selected themes.", icon=":material/info:")
    else:
        topic = st.selectbox(
            "Topic", available, key="learn_topic",
            format_func=lambda key: QUANT_NOTES[key]["title"],
            help="Each note is written to be read against a number you are currently looking at, "
                 "not front to back.",
        )
        theme = THEME_OF_NOTE.get(topic, "Other")
        page = NOTE_PAGES.get(theme)
        if page:
            path, label = page
            with st.container(horizontal=True):
                st.caption(f"Theme: **{theme}** · applies to:")
                st.page_link(path, label=label, icon=":material/open_in_new:")
        st.markdown(QUANT_NOTES[topic]["body"])

    with st.expander(f"All {len(QUANT_NOTES)} notes, by theme", icon=":material/list:"):
        table_caption(
            "Every quant note, grouped by theme.",
            "The 'applies to' column names the page each note supports, for the round trip back.",
        )
        st.dataframe(
            pd.DataFrame([
                {"Theme": theme, "Note": QUANT_NOTES[key]["title"],
                 "Applies to": NOTE_PAGES.get(theme, ("", "—"))[1]}
                for theme, keys in NOTE_THEMES.items() for key in keys if key in QUANT_NOTES
            ]),
            hide_index=True, key="learn_note_index",
            column_config={
                "Theme": st.column_config.TextColumn(width="small"),
                "Applies to": st.column_config.TextColumn(width="small"),
            },
        )

    how_to_read(
        """
- **Read one note, then go apply it.** These are written to be used against a number you are
  currently looking at. Reading them front to back produces recognition without fluency.
- **The "applies to" link is the point.** The fastest way to internalize any of this is to
  read the note, open that page, and go find the thing it describes in your own results.
- **Notes cross-reference each other** with `[[links]]`. Following those chains is usually
  more useful than browsing by theme, because the connections are where the reasoning lives.
"""
    )

# --------------------------------------------------------------------------
with tab_glossary:
    st.markdown("**Metrics** — the same tooltips attached to every number in the dashboard.")
    table_caption(
        "Every metric in the dashboard, with what it measures and what it conceals.",
        "Read the second column before quoting the first.",
    )
    st.dataframe(
        pd.DataFrame(METRIC_DOCS.items(), columns=["Metric", "What it means and what it hides"]),
        hide_index=True, key="learn_metrics",
        column_config={"Metric": st.column_config.TextColumn(width="small")},
    )
    diagnostic_caveat(diagnostics, "sharpe")
    diagnostic_caveat(diagnostics, "num_trades")
    quant_note("sharpe_can_mislead")
    quant_note("equity_curve")

    st.markdown("**Strategies**")
    table_caption(
        "The four base strategies and the market behaviour each one bets on.",
        "Read 'fails when' before running one — it names the evidence that would falsify it.",
    )
    st.dataframe(
        pd.DataFrame([
            {"Strategy": name, "Family": doc["family"], "Bets on": doc["what"],
             "Fails when": doc["fails_when"]}
            for name, doc in STRATEGY_DOCS.items()
        ]),
        hide_index=True, key="learn_strategies",
        column_config={"Strategy": st.column_config.TextColumn(width="small"),
                       "Family": st.column_config.TextColumn(width="small")},
    )

    st.markdown("**Adaptive mechanisms**")
    table_caption(
        "The seven adaptive wrappers and the mechanism each implements.",
        "'Watch for' names the way each one typically fails.",
    )
    st.dataframe(
        pd.DataFrame([
            {"Wrapper": name, "Mechanism": doc["mechanism"], "What it does": doc["what"],
             "Watch for": doc["watch_for"]}
            for name, doc in ADAPTIVE_DOCS.items()
        ]),
        hide_index=True, key="learn_adaptive",
        column_config={"Wrapper": st.column_config.TextColumn(width="small"),
                       "Mechanism": st.column_config.TextColumn(width="small")},
    )
    diagnostic_caveat(diagnostics, "adaptive_exposure")

    st.markdown("**Regime detection methods**")
    table_caption(
        "The five regime detection methods.",
        "'rules' fits nothing and so cannot leak; the others trade transparency for flexibility.",
    )
    st.dataframe(
        pd.DataFrame(REGIME_METHOD_DOCS.items(), columns=["Method", "What it is and what it costs"]),
        hide_index=True, key="learn_methods",
        column_config={"Method": st.column_config.TextColumn(width="small")},
    )

    st.markdown("**Label smoothing**")
    table_caption(
        "The four label smoothers.",
        "All are backward-looking only; each trades responsiveness for stability.",
    )
    st.dataframe(
        pd.DataFrame(SMOOTHING_DOCS.items(), columns=["Smoother", "Behaviour"]),
        hide_index=True, key="learn_smoothing",
        column_config={"Smoother": st.column_config.TextColumn(width="small")},
    )
    diagnostic_caveat(diagnostics, "regime_duration")

    st.markdown("**Regime features**")
    table_caption(
        "Every input the regime models cluster on.",
        "Volatility features do most of the separating; trend features far less.",
    )
    st.dataframe(
        pd.DataFrame(FEATURE_DOCS.items(), columns=["Feature", "What it measures"]),
        hide_index=True, key="learn_features",
        column_config={"Feature": st.column_config.TextColumn(width="small")},
    )

    how_to_read(
        """
- **Use metrics as a diagnostic set, not a scorecard.** No single number describes a strategy.
  Sharpe with exposure tells you whether the risk-adjusted return was earned on enough days;
  drawdown with duration tells you whether you'd have survived it; turnover with cost tells you
  whether it survives a broker.
- **Every metric here has a failure mode, and the second column names it.** Win rate is high
  for strategies that pick up pennies in front of steamrollers; Sharpe understates tail risk;
  total return says nothing about the path. Read the second column before quoting the first.
- **The strategy and mechanism tables are for choosing what to test next.** Read "fails when"
  and "watch for" *before* running something — they tell you what evidence would falsify it,
  which is the question worth deciding in advance.
"""
    )

# --------------------------------------------------------------------------
# Where to go next

# --------------------------------------------------------------------------
# Where to go next
# --------------------------------------------------------------------------
next_steps("learn")
