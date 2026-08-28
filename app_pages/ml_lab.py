"""
Train the direction classifier and watch it overfit.

This page carries the teaching layer for the ML strategy. It is built
around one uncomfortable idea: on daily equity direction the honest
outcome is usually "this model learned nothing", and recognizing that --
rather than tuning until the number looks better -- is the skill being
taught. Every control and chart from the plain version is unchanged.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ml_strategy import MIN_REGIME_TRAIN_ROWS, model_report
from regime_dashboard import (caveat, chart_caption, common_mistakes, explainer,
    how_to_read, next_steps, page_intro, quant_note, require_regimes, table_caption, _ink
)
from strategies import STRATEGIES
from walk_forward import evaluate_out_of_sample

df, regimes = require_regimes()

page_intro("ml_lab")
common_mistakes("ml_lab")

# The shortest-horizon inputs. On a liquid index a one- or two-day move is
# close to pure noise, so a model leaning hardest on these is leaning on
# noise whatever its importance score says. Used as a heuristic below.
NOISE_PRONE_FEATURES = {"return_1d", "return_2d", "return_3d", "volume_change_5d"}

# --------------------------------------------------------------------------
# Onboarding: what this page is for, and why the answer is usually "no"
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        "#### New here? The ML lab in ninety seconds\n"
        "This page trains a classifier to predict whether **tomorrow closes higher than "
        "today**, then converts that prediction into the same long/flat signal every other "
        "strategy here uses. The lab exists less to build a working model than to let you "
        "watch a model fail in the specific, recognizable ways that ML fails in finance."
    )

    intro_left, intro_right = st.columns(2)
    with intro_left:
        st.markdown(
            "**Why daily direction is so hard**\n\n"
            "Daily returns have almost no autocorrelation — tomorrow's direction is close to "
            "a coin flip with a slight upward tilt. There is very little signal, and the "
            "noise around it is enormous.\n\n"
            "Hand a flexible model twelve features and 1,700 rows of that, and it will find "
            "patterns. There are patterns in *any* 1,700 rows of noise; finding them is what "
            "a high-capacity model is good at. That is not a bug in the model — it is the "
            "model working exactly as designed on a problem that doesn't reward it."
        )
    with intro_right:
        st.markdown(
            "**The base rate — read this before any accuracy number**\n\n"
            "About **53% of days are up**, because indices drift upward. So predicting \"up\" "
            "every single day — no model, no features — scores about 53%.\n\n"
            "Which means 54% accuracy is one point of edge, not a triumph, and 51% is *worse "
            "than a constant*. An accuracy figure quoted without its base rate cannot be "
            "evaluated at all. This page always shows them together and computes the "
            "difference for you."
        )

    st.markdown("**And accuracy still isn't the thing that matters**")
    st.markdown(
        "Accuracy counts predictions; markets pay in magnitude. A model right on 60 small "
        "days and wrong on 40 large ones is 60% accurate and loses money — daily returns are "
        "fat-tailed, so a handful of days carry most of the year's move. It runs the other "
        "way too: trend-following is wrong most of the time and profitable anyway.\n\n"
        "Then costs finish the job. A daily model turns the book over 30-80x a year; at 5bps "
        "that is 1.5-4% of annual drag, which one point of edge cannot pay for. "
        "**Read out-of-sample Sharpe, with turnover beside it.** Accuracy only tells you "
        "whether the model learned anything at all."
    )

    st.info(
        "**The result you should expect is a negative one.** A random forest showing 86% train "
        "accuracy and 45% test accuracy has not malfunctioned — it memorized noise, which is "
        "what capacity does when there is no signal. Reporting that honestly is a stronger "
        "demonstration of judgement than any good backtest, because a good backtest here "
        "invites the question of what you did wrong to get it.",
        icon=":material/psychology:",
    )

with st.expander("How to read this page", icon=":material/map:"):
    st.markdown(
        """
**The split.** History is cut chronologically. The model is fitted on the first
`train_frac` (default 70%) and evaluated on the rest. It never sees the future relative to
any row it is tested on.

**The four accuracy numbers, in the order they matter:**

- **Base rate** — what "always predict the majority class" scores on the test window. The
  bar. Roughly 53% on daily equity data, but it depends on the window, so it is computed
  fresh each time.
- **Test accuracy** — accuracy on held-out data. The only accuracy figure worth quoting,
  and it means nothing without the base rate beside it.
- **Train accuracy** — accuracy on data the model was fitted on. Always optimistic. Never
  quote it; use it only to compute the gap.
- **Train − test gap** — the overfitting signature. Large and positive means the model
  memorized rows rather than learning structure. This is the quantity of interest, not
  either level.

**Why accuracy ≠ profitability.** Accuracy weights every day equally; P&L weights each day
by how far the market moved. Being right on typical days is nearly irrelevant when a few
fat-tailed days carry the year. Read Sharpe.

**Confusion matrix.** Rows are what actually happened, columns are what the model said.
The diagonal is correct calls. What you are hunting for is a **column of zeros** — if
"Predicted down" is empty, the model has collapsed to always-long, which is buy-and-hold
with extra machinery and cost.

**Feature importance.** Tells you what the model *leaned on*, never whether leaning on it
was right. Logistic shows signed coefficients (direction is meaningful, magnitudes
comparable because features are standardized). Random forest shows unsigned impurity
importance (no direction, and biased toward continuous features — treat small rank
differences as meaningless).

**Regime conditioning:**

- **off** — one model, no regime information at all.
- **feature** — one model with the regime one-hot encoded as extra columns. Keeps every
  training row. The conservative option.
- **conditional** — a separate model per regime, each fitted only on its own days. Maximum
  flexibility, and it splits your training data while the parameter count stays put.

**Sharpe and turnover.** Sharpe is the risk-adjusted read on whether the signal made money.
Turnover is what it cost to find out — daily models trade constantly, and that drag is
usually larger than the edge.
"""
    )

quant_note("ml_base_rate")
quant_note("ml_overfitting", expanded=True)

explainer(
    "The two models, and what each one is",
    "the random forest is an over-eager pattern collector; logistic regression is an honest "
    "coin-flip calibrator; and the base rate is the market's natural tilt that both have to beat.",
    """
**Random forest — the over-eager pattern collector.** Two hundred trees of depth five give
you thousands of leaves, each fitting a handful of training rows. Show it noise and it will
catalogue that noise in exhaustive detail, then report high confidence about every entry in
the catalogue. On the training set it looks brilliant. On new data the catalogue is
arbitrary, because the patterns it recorded never existed outside those specific rows.

**Logistic regression — the honest coin-flip calibrator.** It can only fit one linear
combination of the features. That constraint looks like a weakness and is the entire point:
with so little signal available, low capacity means less room to fit noise. It will tell you
"about 52%, slightly tilted up" and be roughly right, which is the correct answer to this
problem.

**The base rate — the market's natural tilt.** Indices drift upward, so ~53% of days are up
before anyone models anything. Both models are competing against a constant. Neither
usually wins.

**The rule this teaches, which runs against instinct:** *the less signal there is, the
simpler your model should be.* Most people's reaction to a disappointing result is to reach
for more capacity. Here that makes it strictly worse, and you can watch it happen by
toggling the Model control.
""",
)

with st.container(border=True):
    controls = st.columns([1, 1, 1])
    model_type = controls[0].segmented_control(
        "Model", ["logistic", "random_forest"], default="logistic", key="ml_model",
    ) or "logistic"
    train_frac = controls[1].slider(
        "Train fraction", 0.5, 0.9, 0.7, step=0.05, key="ml_train_frac",
        help="Share of history used for fitting. Only what comes after it means anything.",
    )
    regime_mode = controls[2].segmented_control(
        "Regime conditioning", ["off", "feature", "conditional"], default="off", key="ml_regime_mode",
    ) or "off"

if regime_mode == "conditional":
    st.caption(
        f"A separate model per regime, each fitted only on that regime's training days. Regimes "
        f"with fewer than {MIN_REGIME_TRAIN_ROWS} training rows fall back to a model fitted on "
        f"everything — the table below says which did."
    )
elif regime_mode == "feature":
    st.caption("One model, with the regime one-hot encoded as extra input columns. Keeps all the training rows.")

report_kwargs = {"train_frac": train_frac, "model_type": model_type}
if regime_mode != "off":
    report_kwargs["regimes"] = regimes.labels
    report_kwargs["regime_mode"] = regime_mode

report = model_report(df, **report_kwargs)
gap = report["train_accuracy"] - report["test_accuracy"]

st.subheader("Model diagnostics", divider="gray")
accuracy = st.columns(4)
accuracy[0].metric("Train accuracy", f"{report['train_accuracy']:.1%}",
                   help="Accuracy on data the model was fitted on. Always optimistic; never quote it.")
accuracy[1].metric("Test accuracy", f"{report['test_accuracy']:.1%}",
                   help="Accuracy on held-out data. The only accuracy number worth reporting.")
accuracy[2].metric("Base rate", f"{report['test_base_rate']:.1%}",
                   help="What 'always predict the majority class' would score. The real bar to clear.")
accuracy[3].metric("Train − test gap", f"{gap:+.1%}",
                   help="The overfitting signature. Large and positive means the model memorized.")

edge = report["test_accuracy"] - report["test_base_rate"]
st.metric(
    "Edge over base rate", f"{edge:+.1%}",
    help="Test accuracy minus what a constant prediction would have scored. This is the only "
         "number here that represents anything the model added.",
)

if report["test_accuracy"] <= report["test_base_rate"]:
    caveat(
        f"Test accuracy ({report['test_accuracy']:.1%}) does not beat the base rate "
        f"({report['test_base_rate']:.1%}). This model has learned nothing usable. It is not "
        f"broken — daily direction is close to a coin flip, and this is the honest outcome."
    )
if gap > 0.10:
    caveat(
        f"A {gap:.1%} train/test gap is the textbook overfitting signature. The model found "
        f"patterns in the training rows; those patterns were noise, so they do not transfer. "
        f"Reaching for a bigger model here makes it strictly worse."
    )

how_to_read(
    """
- **Read right to left: base rate first, then test accuracy, then the gap.** The train
  number is only useful as an input to the gap.
- **The edge is what the model added.** Under about +1% it is inside the noise of the test
  window and you should not act on it.
- **A gap over ~10% means memorization.** The correct response is a *simpler* model, not a
  better-tuned one — which is the opposite of most people's instinct.
- **A gap near zero with accuracy at the base rate** is the other common outcome: the model
  didn't overfit, it just found nothing. That is an honest, reportable result.
- **Move the train fraction and watch both numbers.** If they swing around, you're reading
  sampling variation rather than a property of the model.
"""
)

# ---------- Accuracy is not P&L ----------
st.subheader("Accuracy is not P&L", divider="gray")
wf = evaluate_out_of_sample(
    df, STRATEGIES["ml_direction"], train_frac=train_frac, model_type=model_type,
)
pnl = st.columns(4)
pnl[0].metric("In-sample Sharpe", f"{wf['in_sample']['sharpe_ratio']:.2f}")
pnl[1].metric("Out-of-sample Sharpe", f"{wf['out_sample']['sharpe_ratio']:.2f}")
pnl[2].metric("Out-of-sample return", f"{wf['out_sample']['total_return']:.1%}")
pnl[3].metric("Turnover", f"{wf['out_sample']['turnover']:.0f}x",
              help="Daily direction models trade constantly. At 5bps this is a large annual drag.")
st.caption(
    "A model can be 55% accurate and still lose money, if the 45% it gets wrong land on the "
    "large-move days. Read the out-of-sample Sharpe, not the accuracy."
)

turnover = wf["out_sample"]["turnover"]
if turnover > 10:
    caveat(
        f"**Turnover of {turnover:.0f}x a year.** At 5bps that is roughly "
        f"{turnover * 5 / 10_000:.1%} of annual cost drag, before slippage — and the backtest "
        f"above charges nothing. A model with one point of edge over the base rate does not "
        f"generate enough to pay for trading this often. Compute the cost level at which this "
        f"strategy breaks even before taking its Sharpe seriously."
    )
if wf["out_sample"]["sharpe_ratio"] < 0.5:
    caveat(
        f"**Out-of-sample Sharpe of {wf['out_sample']['sharpe_ratio']:.2f}** is weak — below "
        f"roughly 0.5 the return is not distinguishable from noise, given a standard error of "
        f"about ±{np.sqrt(252 / max(len(df) * (1 - train_frac), 1)):.2f} on a window this size. "
        f"Whatever the accuracy figures say, this signal did not make money reliably.",
        level="info",
    )

quant_note("ml_accuracy_vs_pnl")
how_to_read(
    """
- **Compare in-sample against out-of-sample Sharpe.** A large fall is the same overfitting
  story the accuracy gap told, now expressed in money.
- **A positive edge with a negative Sharpe** means the model was right on the small days and
  wrong on the big ones. Accuracy cannot show you this; only P&L can.
- **Turnover is the tax on being clever.** Daily models trade constantly. Multiply turnover
  by your realistic cost in bps and divide by 10,000 to get the annual drag, then compare
  that against the return above.
- **The honest summary of most runs here** is: small or negative edge, weak Sharpe, high
  turnover. That is what "daily direction is hard" looks like in numbers.
"""
)

# ---------- Feature importance ----------
importance_left, importance_right = st.columns([3, 2])
with importance_left:
    st.markdown("**Feature importance**")
    importance = pd.DataFrame(
        sorted(report["feature_importance"].items(), key=lambda kv: -abs(kv[1])),
        columns=["Feature", "Importance"],
    )
    st.altair_chart(
        alt.Chart(importance)
        .mark_bar(cornerRadiusEnd=3, color=_ink()["strategy"])
        .encode(
            x=alt.X("Importance:Q", title="Importance" if model_type == "random_forest" else "Coefficient"),
            y=alt.Y("Feature:N", sort=list(importance["Feature"]), title=None),
            tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".4f")],
        )
        .properties(height=28 * len(importance))
    )
    chart_caption(
        "What the model leaned on when making its predictions.",
        "Logistic shows signed coefficients; random forest shows unsigned importance.",
        "whether the top features are the shortest-horizon ones — and remember this describes "
        "the model, not the market. An overfit model reports confident importances for noise.",
    )
with importance_right:
    st.markdown("**Confusion matrix (test)**")
    matrix = pd.DataFrame(
        report["test_confusion_matrix"],
        index=["Actual down", "Actual up"], columns=["Predicted down", "Predicted up"],
    )
    table_caption(
        "What actually happened (rows) against what the model predicted (columns).",
        "The diagonal is correct calls; an entire column of zeros means the model collapsed to a constant.",
    )
    st.dataframe(matrix, key="ml_confusion")
    predicted_up = matrix["Predicted up"].sum()
    total = matrix.to_numpy().sum()
    st.caption(
        f"The model predicted 'up' on {predicted_up / total:.0%} of test days. If that is near "
        f"100%, it has collapsed to 'always long' — which is buy-and-hold with extra steps."
    )

# Did the model collapse to a constant? A column of zeros is the strongest
# version; predicting one class on >95% of days is the softer version.
up_share = predicted_up / total if total else 0.0
predicted_down = matrix["Predicted down"].sum()
if predicted_down == 0 or predicted_up == 0:
    collapsed_to = "always long" if predicted_down == 0 else "always flat"
    caveat(
        f"**The model has collapsed to {collapsed_to}.** It never once predicted the other "
        f"class on the test window — an entire column of the matrix is zero. It is not "
        f"classifying at all; it has learned the base rate and nothing else. As a strategy "
        f"this is buy-and-hold with extra machinery and extra cost."
    )
elif up_share > 0.95:
    caveat(
        f"**The model predicted 'up' on {up_share:.0%} of test days.** It is very close to a "
        f"constant, so its accuracy is essentially the base rate wearing a model's clothes. "
        f"Check whether the handful of 'down' calls were right — if not, there is no "
        f"classification happening here."
    )

# Heuristic: is the model leaning hardest on the noisiest inputs available?
total_importance = importance["Importance"].abs().sum()
if total_importance > 0:
    noise_share = (
        importance[importance["Feature"].isin(NOISE_PRONE_FEATURES)]["Importance"].abs().sum()
        / total_importance
    )
    top_feature = importance.iloc[0]["Feature"]
    concentration = abs(importance.iloc[0]["Importance"]) / total_importance

    if top_feature in NOISE_PRONE_FEATURES or noise_share > 0.40:
        caveat(
            f"**The model is leaning on the noisiest inputs available.** Its top feature is "
            f"`{top_feature}`, and short-horizon returns plus volume change account for "
            f"{noise_share:.0%} of total importance. A one- or two-day move on a liquid index "
            f"is close to pure noise, so importance concentrated there is a sign the model "
            f"fitted sampling variation rather than structure. Change the train fraction and "
            f"see whether the ranking survives.",
            level="info",
        )
    if concentration > 0.40:
        caveat(
            f"**`{top_feature}` alone holds {concentration:.0%} of total importance.** The model "
            f"is close to a single-variable rule and the other {len(importance) - 1} features are "
            f"largely decoration. Not automatically bad — simple is good — but you should be "
            f"able to state why that one feature would predict tomorrow's direction.",
            level="info",
        )

quant_note("ml_feature_importance")
explainer(
    "Reading the two right-hand panels",
    "the confusion matrix is a report card of correct versus incorrect guesses; feature "
    "importance is a note about what the student studied — which is not the same as whether "
    "they got the answers right.",
    """
**Confusion matrix.** Rows are what actually happened; columns are what the model said.

|  | Predicted down | Predicted up |
|---|---|---|
| **Actual down** | correct (true negative) | wrong (false positive) |
| **Actual up** | wrong (false negative) | correct (true positive) |

The diagonal is what it got right. But the thing to hunt for is a **column of zeros**: if
"Predicted down" is empty, the model never once said down, which means it isn't classifying
— it has learned the base rate and stopped. That failure produces a respectable-looking
accuracy figure and a completely useless strategy, which is why this page checks for it
explicitly rather than leaving it to you to notice.

**Feature importance.** It is a description of the *model*, not of the market. An overfit
forest memorized the training rows *using* some features, and those features score highly.
The scores are accurate about the model's internals and say nothing about whether the
relationship is real. Read them next to the train/test gap: confident importances from a
model with a 20-point gap are describing a fantasy in detail.

Three checks worth running: is one feature dominating (a single-variable rule in disguise);
are the top features the shortest-horizon ones (noise); and does the ranking survive a
change in the train fraction (if not, you're reading sampling variation).
""",
)

# ---------- Per-regime breakdown ----------
if "by_regime" in report and report["by_regime"]:
    st.subheader("Accuracy by regime", divider="gray")
    st.caption(
        "The most informative table here. It separates a model that is 56% in one regime and 46% "
        "in another (worth conditioning on) from one that is 51% everywhere (nothing to condition on)."
    )
    by_regime = pd.DataFrame(report["by_regime"])
    by_regime["name"] = by_regime["regime"].map(lambda r: regimes.names.get(r, str(r)))
    by_regime["edge"] = by_regime["test_accuracy"] - by_regime["base_rate"]
    table_caption(
        "Test accuracy by regime, each measured against its own base rate.",
        "Only the edge column is comparable across regimes, since every regime has a different base rate.",
    )
    st.dataframe(
        by_regime[["name", "test_days", "test_accuracy", "base_rate", "edge", "train_rows", "own_model"]],
        hide_index=True, key="ml_by_regime",
        column_config={
            "name": st.column_config.TextColumn("Regime"),
            "test_days": st.column_config.NumberColumn("Test days"),
            "test_accuracy": st.column_config.NumberColumn("Test accuracy", format="percent"),
            "base_rate": st.column_config.NumberColumn("Base rate", format="percent"),
            "edge": st.column_config.NumberColumn("Edge over base rate", format="percent",
                                                  help="The only column that matters. Positive means the model beat 'always predict the majority class' in that regime."),
            "train_rows": st.column_config.NumberColumn("Train rows",
                                                        help="How much data this regime's model was fitted on. Conditional mode splits your data while the parameter count stays put."),
            "own_model": st.column_config.CheckboxColumn("Own model?",
                                                         help="False means this regime borrowed the global fallback model — too few training rows to justify its own."),
        },
    )
    if (by_regime["edge"] <= 0).all():
        caveat(
            "No regime shows a positive edge over its own base rate. Regime conditioning has not "
            "found structure here — which is itself the answer, and a more useful one than a "
            "marginal improvement you would not be able to trust.",
            level="info",
        )

    edge_spread = by_regime["edge"].max() - by_regime["edge"].min()
    if edge_spread > 0.10 and (by_regime["edge"] > 0).any() and (by_regime["edge"] < 0).any():
        best = by_regime.loc[by_regime["edge"].idxmax()]
        worst = by_regime.loc[by_regime["edge"].idxmin()]
        caveat(
            f"**Per-regime accuracy is unstable:** {best['name']} shows {best['edge']:+.1%} edge "
            f"over {int(best['test_days'])} days while {worst['name']} shows {worst['edge']:+.1%} "
            f"over {int(worst['test_days'])} — a spread of {edge_spread:.1%}. On sample sizes this "
            f"small, a spread that wide is what randomness looks like, not what a regime effect "
            f"looks like. Before concluding the model 'works in one regime', check whether the "
            f"same regime wins at a different train fraction."
        )

    fell_back = by_regime[~by_regime["own_model"]]
    if regime_mode == "conditional" and not fell_back.empty:
        caveat(
            "Regimes that borrowed the global fallback model: "
            + ", ".join(f"**{row['name']}** ({row['train_rows']} train rows)"
                        for _, row in fell_back.iterrows())
            + f". Fewer than {MIN_REGIME_TRAIN_ROWS} training rows means a per-regime model would "
              "have been memorizing rather than learning. Conditional mode is doing less than it "
              "appears to here — any difference you see comes from the regimes that did get their "
              "own model.",
            level="info",
        )

    quant_note("ml_regime_conditional")
    how_to_read(
        """
- **Only the edge column is comparable across regimes.** Each regime has its own base rate —
  a crisis regime might be 45% up-days — so raw accuracy figures cannot be compared directly.
- **Check `Test days` before every conclusion.** A regime with 80 test days gives an accuracy
  estimate with an error bar of roughly ±5 percentage points, which swamps any plausible edge.
- **Check `Own model?` in conditional mode.** If most regimes fell back to the global model,
  conditional mode is barely active and any difference is coming from somewhere else.
- **What a real finding looks like:** consistent positive edge in one regime, over enough
  days to matter, with a mechanism you can state out loud. A scatter of small positive and
  negative edges with no pattern is the honest "nothing here" answer.
"""
    )

with st.expander("Try this: run both models back to back", icon=":material/science:"):
    st.markdown(
        "Switch **Model** between logistic and random forest without changing anything else, and "
        "watch four numbers:\n\n"
        "1. Train accuracy — the forest's will be far higher.\n"
        "2. Test accuracy — the forest's will usually be *lower*, sometimes below the base rate.\n"
        "3. The gap — this is the quantity of interest, not either level.\n"
        "4. Out-of-sample Sharpe — where the two diverge in a way that costs money.\n\n"
        "The lesson runs against most people's instincts: **with this little signal, the less "
        "flexible model is the better one**, and a disappointing result is not a reason to reach "
        "for more capacity."
    )

# --------------------------------------------------------------------------
# Where to go next
# --------------------------------------------------------------------------
next_steps("ml_lab")
