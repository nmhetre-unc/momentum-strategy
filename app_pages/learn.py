"""How to think like a quant: the reference material behind everything else here."""

import pandas as pd
import streamlit as st

from adaptive import ADAPTIVE_DOCS
from quant_notes import LEARNING_PATH, METRIC_DOCS, PITFALLS, QUANT_NOTES
from regime import REGIME_METHOD_DOCS, SMOOTHING_DOCS
from regime_features import FEATURE_DOCS
from strategies import STRATEGY_DOCS

st.markdown(
    "This platform is built to teach one habit above all the others: **assume your result is "
    "wrong until you have tried to break it.** Everything below is in service of that."
)

tab_path, tab_pitfalls, tab_notes, tab_glossary = st.tabs(
    ["Learning path", "Common pitfalls", "Quant notes", "Glossary"], on_change="rerun",
)

# --------------------------------------------------------------------------
with tab_path:
    st.markdown(
        "Work through these in order. Each stage has a **done-when** condition that is about a "
        "habit rather than a completed task — the habit is the transferable part."
    )
    for stage in LEARNING_PATH:
        with st.container(border=True):
            st.markdown(f"**{stage['stage']} — {stage['goal']}**")
            st.markdown(stage["do"])
            st.caption(f"Done when: {stage['done_when']}")

# --------------------------------------------------------------------------
with tab_pitfalls:
    st.markdown(
        "Three ways to produce a backtest that is confidently wrong. They are listed in the order "
        "people tend to hit them, and the third one catches people who have already learned to "
        "avoid the first two."
    )
    for name, pitfall in PITFALLS.items():
        with st.container(border=True):
            st.markdown(f"**{name}** — {pitfall['summary']}")
            st.markdown("**Where it hides**")
            for place in pitfall["where_it_hides"]:
                st.markdown(f"- {place}")
            st.markdown(f"**The tell:** {pitfall['tell']}")
            st.markdown(f"**The fix:** {pitfall['fix']}")

# --------------------------------------------------------------------------
with tab_notes:
    topic = st.selectbox(
        "Topic", list(QUANT_NOTES), key="learn_topic",
        format_func=lambda key: QUANT_NOTES[key]["title"],
    )
    st.markdown(QUANT_NOTES[topic]["body"])

# --------------------------------------------------------------------------
with tab_glossary:
    st.markdown("**Metrics** — the same tooltips attached to every number in the dashboard.")
    st.dataframe(
        pd.DataFrame(METRIC_DOCS.items(), columns=["Metric", "What it means and what it hides"]),
        hide_index=True, key="learn_metrics",
        column_config={"Metric": st.column_config.TextColumn(width="small")},
    )

    st.markdown("**Strategies**")
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

    st.markdown("**Regime detection methods**")
    st.dataframe(
        pd.DataFrame(REGIME_METHOD_DOCS.items(), columns=["Method", "What it is and what it costs"]),
        hide_index=True, key="learn_methods",
        column_config={"Method": st.column_config.TextColumn(width="small")},
    )

    st.markdown("**Label smoothing**")
    st.dataframe(
        pd.DataFrame(SMOOTHING_DOCS.items(), columns=["Smoother", "Behaviour"]),
        hide_index=True, key="learn_smoothing",
        column_config={"Smoother": st.column_config.TextColumn(width="small")},
    )

    st.markdown("**Regime features**")
    st.dataframe(
        pd.DataFrame(FEATURE_DOCS.items(), columns=["Feature", "What it measures"]),
        hide_index=True, key="learn_features",
        column_config={"Feature": st.column_config.TextColumn(width="small")},
    )
