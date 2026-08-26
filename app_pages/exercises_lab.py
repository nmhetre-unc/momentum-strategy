"""Quant intern exercises, checked against the data currently loaded."""

import pandas as pd
import streamlit as st

from exercises import EXERCISES, LEVELS
from regime_dashboard import require_data
from strategies import STRATEGIES

df = require_data()
ticker = st.session_state["ticker"]

st.session_state.setdefault("exercise_results", {})

st.markdown(
    f"Ten exercises, checked against **{ticker}** as currently loaded in the sidebar. Change the "
    "ticker or date range and every check re-runs on the new data — which is itself the point of "
    "several of them."
)
st.info(
    "**A check that reports 'not confirmed' is not a failing grade.** Several of these test whether "
    "a well-known effect shows up in *your* data, and sometimes it doesn't. Reading "
    "'the expected effect is absent here, and here is the number' as information rather than as an "
    "error is most of the job.",
    icon=":material/psychology:",
)

chosen_levels = st.pills(
    "Filter by level", LEVELS, selection_mode="multi", default=LEVELS, key="ex_levels",
)
visible = [e for e in EXERCISES if e.level in (chosen_levels or LEVELS)]

progress = st.session_state["exercise_results"]
attempted = sum(1 for e in visible if e.key in progress)
st.progress(attempted / len(visible) if visible else 0.0,
            text=f"{attempted} of {len(visible)} exercises run")

for exercise in visible:
    with st.container(border=True):
        st.markdown(f"**{exercise.title}**")
        st.caption(exercise.level)
        st.markdown(exercise.prompt)

        with st.expander("Expected output", icon=":material/visibility:"):
            st.markdown(exercise.expected)

        if exercise.hints:
            with st.expander("Hints", icon=":material/lightbulb:"):
                for hint in exercise.hints:
                    st.markdown(f"- {hint}")

        answer = None
        if exercise.answer_prompt:
            if exercise.answer_options:
                answer = st.radio(
                    exercise.answer_prompt, exercise.answer_options,
                    index=None, key=f"ex_answer_{exercise.key}",
                )
            elif exercise.key == "best_regime":
                # Options depend on the regimes actually detected, so they're
                # built here rather than hard-coded in exercises.py.
                settings = st.session_state["regime_settings"]
                try:
                    from regime_dashboard import cached_regimes

                    regimes = cached_regimes(
                        df, settings["method"], settings["n_regimes"], settings["fit_frac"],
                        settings["smooth"], settings["min_duration"], settings["decode"],
                        settings["walk_forward"],
                    )
                    options = [regimes.names[i] for i in sorted(regimes.names)]
                except ValueError:
                    options = []
                if options:
                    answer = st.radio(
                        exercise.answer_prompt, options, index=None, key=f"ex_answer_{exercise.key}",
                    )

        strategy_choice = None
        if exercise.key in ("best_regime", "walk_forward", "benchmark"):
            strategy_choice = st.selectbox(
                "Strategy to check", list(STRATEGIES), key=f"ex_strategy_{exercise.key}",
            )

        actions = st.columns([1, 1, 4])
        run = actions[0].button("Run check", icon=":material/play_arrow:", key=f"ex_run_{exercise.key}")
        reveal = actions[1].toggle("Show answer", key=f"ex_reveal_{exercise.key}")

        if run and exercise.check is not None:
            context = {"df": df, "ticker": ticker, "answer": answer}
            if strategy_choice:
                context["strategy"] = strategy_choice
            with st.spinner("Checking..."):
                try:
                    progress[exercise.key] = exercise.check(context)
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}", icon=":material/error:")
                    progress.pop(exercise.key, None)

        outcome = progress.get(exercise.key)
        if outcome is not None:
            if outcome.passed:
                st.success(outcome.message, icon=":material/check_circle:")
            else:
                st.info(outcome.message, icon=":material/info:")

            if isinstance(outcome.evidence, pd.DataFrame) and not outcome.evidence.empty:
                st.dataframe(outcome.evidence, hide_index=True, key=f"ex_evidence_{exercise.key}")
            elif isinstance(outcome.evidence, dict):
                st.dataframe(
                    pd.DataFrame(outcome.evidence.items(), columns=["Measure", "Value"]),
                    hide_index=True, key=f"ex_evidence_{exercise.key}",
                )

        if reveal:
            st.markdown("---")
            st.markdown(exercise.explanation)

st.caption(
    "Every check here is also runnable without the dashboard: "
    "`from exercises import run_all; run_all(df)` returns the same results as a table."
)
