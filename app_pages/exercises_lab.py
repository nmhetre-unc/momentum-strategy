"""
Quant intern exercises, checked against the data currently loaded.

This page carries the teaching layer for the exercise set. The design
principle throughout: an exercise is a *habit* being installed, not a task
being completed, so every automated check is followed by an interpretation
that says what the numbers mean AND what it means when the expected effect
is absent. Every control, card and button from the plain version is
unchanged.
"""

import numpy as np
import pandas as pd
import streamlit as st

from exercises import EXERCISES, LEVELS
from regime import regime_stability
from regime_dashboard import (cached_regimes, caveat, chart_caption, common_mistakes,
    explainer, next_steps, page_intro, quant_note, require_data, table_caption
)
from strategies import STRATEGIES

df = require_data()
ticker = st.session_state["ticker"]

st.session_state.setdefault("exercise_results", {})

page_intro("exercises")
common_mistakes("exercises")

# --------------------------------------------------------------------------
# Teaching content, keyed by exercise
# --------------------------------------------------------------------------
# The quant note that belongs under each exercise. Reusing the shared notes
# rather than writing exercise-specific copy keeps one canonical explanation
# per concept -- the Learn page browses exactly the same text.
EXERCISE_NOTES = {
    "sma_sideways": "trend_vs_chop",
    "ml_accuracy": "ml_base_rate",
    "best_regime": "risk_by_regime",
    "rf_overfit": "bias_variance",
    "walk_forward": "walk_forward",
    "param_sensitivity": "walkforward_reason",
    "lookahead": "lookahead_bias",
    "costs": "turnover_costs",
    "vol_targeting": "volatility_targeting",
    "benchmark": "fair_comparison",
}

EXERCISE_METAPHORS = {
    "sma_sideways": (
        "What an SMA crossover actually is",
        "a trend-following thermometer — it reads the temperature that has already been "
        "reached, and it reads it slowly.",
        """
A moving average is a *lagging* measurement by construction: it averages the recent past,
so it always reports where price has been rather than where it is going. The crossover
fires when the fast reading has moved decisively past the slow one.

In a genuine trend that lag is acceptable — the move continues long enough that entering
late still leaves most of it to capture. In a range the same lag is fatal: by the time the
thermometer registers "warm", price is already at the top of its range and about to fall
back. The rule then buys near local highs and sells near local lows, repeatedly and by
design.

That's why the exercise is about *mechanism*, not tuning. No choice of window fixes a
thermometer being used to predict the weather.
""",
    ),
    "rf_overfit": (
        "What the random forest is doing",
        "an over-eager pattern collector — it catalogues every coincidence in the training "
        "data and reports each one with total confidence.",
        """
Two hundred trees of depth five give you thousands of leaves, each fitting a handful of
training rows. Show that machinery a dataset with almost no signal and it will still fill
the catalogue — because there are always patterns in 1,700 rows of noise, and finding them
is exactly what it is built to do.

On the training set the catalogue looks like deep understanding: 86% accuracy. On new data
the entries are arbitrary, because the patterns were never properties of the market. Test
accuracy lands at chance, and can land *below* it, because memorized noise is not neutral —
it actively misleads.

Logistic regression, by contrast, can only draw one straight boundary. That constraint
looks like a weakness and is the whole reason it does better here.
""",
    ),
    "walk_forward": (
        "What walk-forward validation is",
        "multiple practice laps before the race — and only the race counts.",
        """
Nobody judges a runner by their best training session on their favourite course. You judge
them by whether the training transfers to race day, under conditions they didn't choose.

The in-sample period is practice: where parameters were picked, models fitted, charts
studied. The out-of-sample period is the race. A large gap between them means the athlete
trained for the training.

And note what ruins the whole arrangement: looking at the race result, going back to adjust
the training, then reporting the race time as if it were still an independent test. That is
silent fitting, and it is the one bias here with no technical fix.
""",
    ),
    "lookahead": (
        "What regime lookahead actually is",
        "grading a student's exam with tomorrow's answer key — they look brilliant, and you "
        "have learned nothing about whether they can pass the test.",
        """
Fitting a regime model on all of history embeds the whole history in every label. A day in
2015 labelled "low volatility" was assigned to a cluster whose centre was computed partly
from 2020 and 2022. In 2015 that cluster did not exist.

The resulting backtest looks excellent for an entirely boring reason: the strategy knows
which regime it is in with a precision that was never available at the time.

What makes it so effective a trap is that nothing looks wrong. The code is clean, the model
is standard, the equity curve is beautiful. The error is invisible unless you specifically
ask, of each label, *when did this become knowable?*
""",
    ),
    "vol_targeting": (
        "What volatility targeting does",
        "driving slower on icy roads — you don't change the car or the route, only the speed.",
        """
The entry signal is untouched. Only position size changes, scaled inversely to trailing
realized volatility: when volatility doubles, the position halves.

It works because it leans on the one market property that is genuinely forecastable.
Volatility clusters — today's is a good predictor of tomorrow's. Direction does not. So
rather than trying to classify *what kind* of bad road you're on, targeting just measures
how slick it is right now and adjusts.

The practical effect on equity data is usually that returns are roughly preserved while
maximum drawdown falls, because the position was already small when the crash arrived
rather than being cut afterwards at the worst prices.
""",
    ),
}

# Two or three bullets after each check: what the numbers mean, what the
# effect is meant to demonstrate, and -- the important one -- how to read
# the result when the expected effect does not appear.
INTERPRETATIONS = {
    "sma_sideways": """
- **The two Sharpe ratios are the finding.** Compare the sideways regime's row against the
  best other regime. A large negative gap is the structural cost of a lagging trend rule.
- **This is meant to show a mechanism, not bad luck.** The crossover buys near local highs
  and sells near local lows in a range *by construction*, so no parameter choice repairs it.
- **If the effect is absent**, check the `days` column first — a short regime gives a Sharpe
  with an enormous error bar. Then check whether the "choppy" regime really was choppy, using
  the efficiency ratio on the Regimes page.
""",
    "ml_accuracy": """
- **Read the `gap` column, not the accuracy columns.** Train minus test is the overfitting
  signature; either level alone tells you little.
- **Then read `base_rate`.** A model at or below it has added nothing at all — it has been
  beaten by a one-line constant.
- **If the forest's gap isn't larger than logistic's**, that is unusual and worth
  investigating rather than accepting: check whether the train window is so small that even
  the in-sample accuracy is modest.
""",
    "best_regime": """
- **The `days` column decides how much of this to believe.** Standard error on an annualized
  Sharpe is roughly sqrt(252/N) — over 120 days that is about ±1.5, wide enough to swallow
  most gaps you'll see.
- **The worst regime is more useful than the best.** Knowing where a strategy loses is
  directly actionable; knowing where it wins usually confirms what you assumed.
- **If your prediction was wrong**, that is the more interesting outcome: the strategy is not
  doing what its description implies, and finding that out is the point of the exercise.
""",
    "rf_overfit": """
- **The accuracy gap and the Sharpe collapse are the same fact** seen twice — once in
  classification terms, once in money.
- **This demonstrates variance, not a bug.** A high-capacity model on a low-signal problem
  fits noise; that is what capacity does when there is nothing else to fit.
- **If the gap is under 10%**, either this ticker has more structure than most or the training
  window is small. Compare test accuracy against the base rate before concluding the model
  works.
""",
    "walk_forward": """
- **The gap matters more than either level.** A small drop means the process generalizes,
  which is worth more than a high out-of-sample number with a large gap.
- **Check `num_trades` out-of-sample before accepting any verdict.** Under ten trades and
  the Sharpe describes that many coin flips.
- **If performance held up**, resist concluding the strategy works — one split is one draw.
  Run rolling walk-forward on the Validation page to see the distribution.
""",
    "param_sensitivity": """
- **The spread across windows is your real uncertainty.** If out-of-sample Sharpe ranges from
  −0.2 to 0.9, reporting the 0.9 reports the luckiest draw from a distribution you could not
  have sampled in advance.
- **Compare the best in-sample window against the best out-of-sample one.** When they differ,
  you have direct evidence that choosing parameters on in-sample results does not work.
- **Prefer plateaus to peaks.** A broad region of similar values suggests a real effect; a
  single spike is noise you would never have located prospectively.
""",
    "lookahead": """
- **The Sharpe difference is the bias, measured on your data.** It is not an estimate or an
  analogy — it is the performance that exists only because the model saw the future.
- **Read the exposure column too.** The walk-forward version produces no labels for its first
  two years, so it trades a shorter effective history and that alone changes the numbers.
- **If the honest version scored higher**, that happens: walk-forward labels are noisier but
  not automatically worse. Compare exposure and the labelled-day count before concluding
  anything.
""",
    "costs": """
- **Multiply `turnover` by cost and divide by 10,000** to get each strategy's annual drag.
  Do it by hand once for the ML strategy; the number is usually larger than people expect.
- **The point is that costs are not uniform.** They penalize precisely the strategies that
  look best in a frictionless backtest, which is why the ranking can invert.
- **If the ranking survives**, look at the `sharpe_lost` column anyway — the damage lands very
  unevenly even when the order happens to hold.
""",
    "vol_targeting": """
- **Compare max drawdown and Sharpe together.** Drawdown falling on its own is not
  impressive; any strategy can achieve that by holding less.
- **Check the `exposure` row.** If targeting cut average exposure substantially, part of the
  improvement is simply reduced participation rather than better timing.
- **If neither improved**, the target is probably far from what the asset realizes — set
  target_vol near the asset's own annualized volatility and try again.
""",
    "benchmark": """
- **Compare Sharpe and max drawdown, not just total return.** A strategy can justify itself on
  risk even while underperforming on return.
- **Note the exposure.** Matching most of buy-and-hold's return at 40% average exposure is a
  real result — but you have to make that argument explicitly.
- **If it does not clear the bar**, that is the most common outcome and the most commonly
  omitted comparison. It is a finding to report, not a failure to hide.
""",
}


# --------------------------------------------------------------------------
# Contextual caveats, computed from each check's own evidence
# --------------------------------------------------------------------------
def _pct(value) -> float:
    """Parses '45.1%' or 0.451 into a float fraction; None if it can't."""
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    try:
        return float(str(value).strip().rstrip("%")) / 100.0
    except (ValueError, AttributeError):
        return None


@st.cache_data(show_spinner=False, ttl="1h", max_entries=10)
def _regime_health(prices: pd.DataFrame, settings: dict):
    """Regime stability and causality, for the exercises that depend on labels."""
    result = cached_regimes(
        prices, settings["method"], settings["n_regimes"], settings["fit_frac"],
        settings["smooth"], settings["min_duration"], settings["decode"],
        settings["walk_forward"],
    )
    return regime_stability(result.labels), result.causal, result.method


def result_caveats(key: str, outcome, prices: pd.DataFrame):
    """
    Warnings attached to a specific check's result. Each one names a reason
    the numbers just produced are less trustworthy than they look.
    """
    evidence = outcome.evidence
    settings = st.session_state["regime_settings"]

    # --- sample size in a per-regime table -------------------------------
    if isinstance(evidence, pd.DataFrame) and "days" in evidence.columns:
        thin = evidence[evidence["days"] < 200]
        if not thin.empty:
            caveat(
                "Rows built on under 200 days: "
                + ", ".join(f"**{r.get('name', '?')}** ({int(r['days'])}d)"
                            for _, r in thin.iterrows())
                + ". Standard error on an annualized Sharpe is roughly sqrt(252/days), so those "
                  "rows carry error bars wide enough to contain almost any conclusion.",
                level="info",
            )

    # --- trade count ------------------------------------------------------
    if isinstance(evidence, pd.DataFrame) and "num_trades" in evidence.index:
        for column in evidence.columns:
            trades = evidence.loc["num_trades", column]
            if isinstance(trades, (int, float, np.number)) and trades < 10:
                caveat(
                    f"**Only {int(trades)} trades in the {column} column.** Every metric there "
                    f"describes that many independent bets. No conclusion is supportable on that "
                    f"evidence, in either direction.",
                    level="info",
                )

    # --- model accuracy against its own base rate -------------------------
    if isinstance(evidence, pd.DataFrame) and {"test_accuracy", "base_rate"}.issubset(evidence.columns):
        below = evidence[evidence["test_accuracy"] <= evidence["base_rate"]]
        if not below.empty:
            names = ", ".join(f"**{r.get('model', '?')}**" for _, r in below.iterrows())
            caveat(
                f"{names} did not beat the base rate. A model at or below it has been beaten by "
                f"a one-line constant, so its accuracy figure describes nothing it learned."
            )
    if isinstance(evidence, dict):
        test_accuracy, base_rate = _pct(evidence.get("test_accuracy")), _pct(evidence.get("test_base_rate"))
        if test_accuracy is not None and base_rate is not None and test_accuracy <= base_rate:
            caveat(
                f"Test accuracy ({test_accuracy:.1%}) is at or below the base rate "
                f"({base_rate:.1%}) — the model has been beaten by always predicting the "
                f"majority class."
            )

    # --- turnover bought nothing -----------------------------------------
    if isinstance(evidence, pd.DataFrame) and "turnover" in evidence.index and len(evidence.columns) == 2:
        base_col, adapted_col = evidence.columns[0], evidence.columns[1]
        try:
            extra = evidence.loc["turnover", adapted_col] - evidence.loc["turnover", base_col]
            gained = evidence.loc["sharpe_ratio", adapted_col] - evidence.loc["sharpe_ratio", base_col]
            if extra > 0.5 and gained <= 0:
                caveat(
                    f"**{adapted_col} added {extra:.1f}x of annual turnover and did not improve "
                    f"Sharpe** ({gained:+.2f}). The adaptation did not pay for its own trading — "
                    f"which is a complete result, and the simpler strategy wins here."
                )
        except (KeyError, TypeError):
            pass

    # --- regime-dependent exercises --------------------------------------
    if key in ("sma_sideways", "best_regime", "lookahead"):
        try:
            stability, causal, method = _regime_health(prices, settings)
        except Exception:
            return
        if stability["avg_duration"] < 15 and stability["n_episodes"]:
            caveat(
                f"Regime episodes average only {stability['avg_duration']:.0f} days under the "
                f"current **{method}** settings. Real regimes last weeks to months — at this "
                f"length the labels are flickering, and any per-regime conclusion is describing "
                f"noise. Raise the confirmation window in the sidebar and re-run."
            )
        if not causal and key != "lookahead":
            caveat(
                "The sidebar's regime labels are **not causal** — the model was fitted on the "
                "same days it is labelling, so these labels embed knowledge of the future. Fine "
                "for describing history, invalid for the performance numbers above. Set the fit "
                "fraction below 1.0 or switch on walk-forward detection."
            )

    # --- walk-forward fold count -----------------------------------------
    if key in ("walk_forward", "param_sensitivity"):
        possible_folds = max((len(prices) - 756) // 126, 0)
        if possible_folds < 5:
            caveat(
                f"This date range supports only about {possible_folds} rolling walk-forward "
                f"folds at the default 756/126 window. One split is one draw, and a handful of "
                f"folds is barely better — widen the date range before treating either result as "
                f"settled.",
                level="info",
            )


# --------------------------------------------------------------------------
# Onboarding
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        "#### New here? What this page is for\n"
        "Ten exercises that run against **whatever data you have loaded in the sidebar**. "
        "Each one asks you to predict something, then checks it against the actual numbers "
        "from your ticker and date range."
    )

    intro_left, intro_right = st.columns(2)
    with intro_left:
        st.markdown(
            "**These install habits, not answers**\n\n"
            "The point is not to complete ten tasks. It is to build the reflexes a quant "
            "researcher uses automatically: *check the sample size before believing a "
            "Sharpe ratio; ask whether a number is in-sample or out-of-sample; compare "
            "against buy-and-hold; count the comparisons you ran.*\n\n"
            "So the useful part of each exercise is the sentence you write afterwards, not "
            "the green tick."
        )
    with intro_right:
        st.markdown(
            "**\"Not confirmed\" is not a failing grade**\n\n"
            "Several exercises test whether a well-known effect shows up in *your* data — "
            "and sometimes it doesn't. A random forest that didn't overfit on your ticker, "
            "or a choppy regime where trend-following happened to do fine, is a real "
            "result.\n\n"
            "Learning to read *\"the expected effect is absent here, and here is the number\"* "
            "as information rather than as an error is most of the job."
        )

    st.markdown("**The four levels build on each other**")
    st.markdown(
        "| Level | What it installs |\n|---|---|\n"
        "| **1 · Foundations** | Compare against a benchmark; know that parameters move risk, not just return. |\n"
        "| **2 · Validation** | Distrust your own backtests; read the in-sample/out-of-sample gap first. |\n"
        "| **3 · Regimes** | Stop treating the market as one thing; ask *where* a return came from. |\n"
        "| **4 · Adaptive** | Judge whether added complexity paid for its own turnover. |\n"
        "\n"
        "Work roughly in order — level 3 assumes you already distrust a single backtest, and "
        "level 4 assumes you can read a per-regime table."
    )

    st.info(
        f"**Everything here re-runs on the data in the sidebar.** You are currently on "
        f"**{ticker}**, {df.index[0].date()} to {df.index[-1].date()}. Change the ticker or "
        f"dates and every check recomputes against the new data — which is itself the lesson "
        f"in several exercises. A result that holds on SPY and vanishes on a different asset "
        f"was a property of SPY, not of the strategy.",
        icon=":material/refresh:",
    )

with st.expander("How to use this page", icon=":material/map:"):
    st.markdown(
        """
**1 · Pick an exercise.** Use the level filter below. If you're starting out, do the two
Foundations exercises first — they take five minutes and set up everything else.

**2 · Read the prompt and commit to an answer before running anything.** This matters more
than it sounds. Predicting first is what turns the check into a test of your model of the
market; running first turns it into a demonstration you'll nod along with and forget.

**3 · Open "Expected output" only after you've formed a view.** It tells you what the effect
should look like, which is a strong hint. Use it to check your reasoning, not to form it.

**4 · Use hints when stuck, not when uncertain.** Being uncertain is the productive state.
The hints point at *where to look* rather than what you'll find.

**5 · Run the check.** It computes the relevant numbers on your loaded data and reports
whether the effect appeared, with the evidence table underneath.

**6 · Read the interpretation, then the caveats.** The interpretation says what the numbers
mean and what their absence would mean. The caveats flag specific reasons this particular
run is less trustworthy than it looks — small samples, non-causal labels, too few trades.

**7 · Write your own explanation before opening "Show answer".** One or two sentences, in
your own words, on the *mechanism*. Then compare. Where your version and the written one
differ is exactly where your understanding is thin — that gap is the whole value of the
exercise, and skipping straight to the answer discards it.

**A note on the automated checks.** They verify whether an effect is present in your data.
They cannot verify whether you understood it. Only the sentence you write does that.
"""
    )

# --------------------------------------------------------------------------
# Existing UI: filters, progress, cards
# --------------------------------------------------------------------------
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

        # Concept behind the exercise: the shared quant note, plus a metaphor
        # where one helps. Both collapsed, so the card stays scannable.
        note_key = EXERCISE_NOTES.get(exercise.key)
        if note_key:
            quant_note(note_key)
        if exercise.key in EXERCISE_METAPHORS:
            title, metaphor, body = EXERCISE_METAPHORS[exercise.key]
            explainer(title, metaphor, body)

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

            if exercise.key in INTERPRETATIONS:
                with st.container(border=True):
                    st.markdown("**How to interpret this result**")
                    st.markdown(INTERPRETATIONS[exercise.key])

            result_caveats(exercise.key, outcome, df)

            st.caption(
                "Before opening **Show answer**: write one or two sentences of your own on the "
                "*mechanism* behind this result. Where your version differs from the written one "
                "is where your understanding is thin — and that gap is the point of the exercise."
            )

        if reveal:
            st.markdown("---")
            st.markdown(exercise.explanation)

st.caption(
    "Every check here is also runnable without the dashboard: "
    "`from exercises import run_all; run_all(df)` returns the same results as a table."
)

# --------------------------------------------------------------------------
# Where to go next

# --------------------------------------------------------------------------
# Where to go next
# --------------------------------------------------------------------------
next_steps("exercises")
