"""Train the direction classifier and watch it overfit."""

import altair as alt
import pandas as pd
import streamlit as st

from ml_strategy import MIN_REGIME_TRAIN_ROWS, model_report
from regime_dashboard import caveat, quant_note, require_regimes, _ink
from strategies import STRATEGIES
from walk_forward import evaluate_out_of_sample

df, regimes = require_regimes()

quant_note("ml_overfitting", expanded=True)

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
    st.caption(
        "Importance tells you what the model leaned on, not whether leaning on it was correct. "
        "An overfit model reports confident importances for features that carry no signal."
    )
with importance_right:
    st.markdown("**Confusion matrix (test)**")
    matrix = pd.DataFrame(
        report["test_confusion_matrix"],
        index=["Actual down", "Actual up"], columns=["Predicted down", "Predicted up"],
    )
    st.dataframe(matrix, key="ml_confusion")
    predicted_up = matrix["Predicted up"].sum()
    total = matrix.to_numpy().sum()
    st.caption(
        f"The model predicted 'up' on {predicted_up / total:.0%} of test days. If that is near "
        f"100%, it has collapsed to 'always long' — which is buy-and-hold with extra steps."
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
