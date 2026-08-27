"""
Quant training terminal — entry point.

Run with: streamlit run app.py

This file owns everything shared across pages: the data selection, the
regime-detection settings, and the navigation. Each page in app_pages/ is
a plain script that reads that shared state and renders one workspace.

The split exists for a performance reason as much as a tidiness one:
Streamlit reruns everything visible on every widget interaction, and
regime detection plus model fitting is far too expensive to run seven
times per click. With st.navigation only the active page's script runs,
and the expensive pieces are cached on their exact parameter sets.
"""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Quant training terminal",
    page_icon=":material/monitoring:",
    layout="wide",
)

from regime_dashboard import MAX_REGIMES, load_prices  # noqa: E402  (must follow set_page_config)
from regime import REGIME_METHODS  # noqa: E402

# --------------------------------------------------------------------------
# Shared state, initialized in one place
# --------------------------------------------------------------------------
st.session_state.setdefault("prices", None)
st.session_state.setdefault("ticker", "SPY")
st.session_state.setdefault("load_error", None)
st.session_state.setdefault("regime_settings", {
    "method": "hmm", "n_regimes": 3, "fit_frac": 0.6, "smooth": "min_duration",
    "min_duration": 5, "decode": "filter", "walk_forward": False,
})

# Each regime control gets a stable widget key and seeds its default through
# session_state rather than through the widget's own value argument. Passing
# settings["fit_frac"] as the default of the slider that also WRITES
# settings["fit_frac"] makes the widget's identity change every time it moves,
# so Streamlit re-registers it and the new value can silently fail to stick.
REGIME_DEFAULTS = {
    "rg_method": "hmm", "rg_n_regimes": 3, "rg_fit_frac": 0.6,
    "rg_smooth": "min_duration", "rg_min_duration": 5, "rg_decode": "filter",
    "rg_walk_forward": False,
}
for _key, _default in REGIME_DEFAULTS.items():
    st.session_state.setdefault(_key, _default)


def _load(ticker: str, start, end):
    """Fetches prices into session state, keeping any error for display."""
    try:
        with st.spinner(f"Fetching {ticker}..."):
            st.session_state["prices"] = load_prices(ticker, str(start), str(end))
        st.session_state["ticker"] = ticker
        st.session_state["load_error"] = None
    except Exception as exc:
        st.session_state["prices"] = None
        st.session_state["load_error"] = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Sidebar: data and regime settings, shared by every page
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Data", divider="gray")
    with st.form("data_form", border=False):
        ticker = st.text_input("Ticker", value=st.session_state["ticker"]).upper().strip()
        start = st.date_input("Start date", value=pd.to_datetime("2008-01-01"))
        end = st.date_input("End date", value=pd.to_datetime("2025-01-01"))
        loaded = st.form_submit_button("Load data", icon=":material/download:", width="stretch")

    if loaded:
        _load(ticker, start, end)
    elif st.session_state["prices"] is None and st.session_state["load_error"] is None:
        # First visit: load the default so the terminal opens with something
        # on screen rather than an empty shell.
        _load(ticker, start, end)

    if st.session_state["load_error"]:
        st.error(st.session_state["load_error"], icon=":material/cloud_off:")
        st.caption(
            "No network here? Every page still works on synthetic data — run "
            "`python test_logic.py` and `python test_regime.py` from a terminal instead."
        )
    elif st.session_state["prices"] is not None:
        prices = st.session_state["prices"]
        st.caption(
            f"{st.session_state['ticker']} · {len(prices):,} rows · "
            f"{prices.index[0].date()} to {prices.index[-1].date()}"
        )

    st.subheader("Regime model", divider="gray")
    st.caption("Shared by every page that shows regimes.")

    st.selectbox(
        "Detection method", REGIME_METHODS, key="rg_method",
        help="Explained in full on the Regimes page. Start with 'rules' — it fits nothing, so nothing can leak.",
    )
    st.slider(
        "Number of regimes", 2, MAX_REGIMES, key="rg_n_regimes",
        help=(
            "Ignored by 'rules' and 'supervised', which define four by construction. Capped at "
            f"{MAX_REGIMES}: more than that over-segments a decade of daily data, and the ordinal "
            "color ramp stops being distinguishable."
        ),
    )
    st.toggle(
        "Walk-forward detection", key="rg_walk_forward",
        help=(
            "Refit the regime model on an expanding window and label only forward. Slower, and it "
            "produces no labels for the first two years — because you genuinely had no model then. "
            "This is the honest setting."
        ),
    )
    if not st.session_state["rg_walk_forward"]:
        st.slider(
            "Fit fraction", 0.4, 1.0, step=0.05, key="rg_fit_frac",
            help=(
                "Share of history the model is fitted on. At 1.0 the labels embed knowledge of the "
                "future — useful for describing history, invalid for backtesting."
            ),
        )
    st.selectbox(
        "Label smoothing", ["min_duration", "ema_prob", "median", "none"], key="rg_smooth",
        help="All options are backward-looking only. A centered filter would look tidier and be lookahead bias.",
    )
    st.slider(
        "Confirmation days", 1, 21, key="rg_min_duration",
        help="How long a new regime must persist before it's accepted. Higher means fewer head-fakes and more lag.",
    )
    if st.session_state["rg_method"] == "hmm" and not st.session_state["rg_walk_forward"]:
        st.selectbox(
            "HMM decoding", ["filter", "smooth", "viterbi"], key="rg_decode",
            help=(
                "'filter' uses data up to today only — the one you could have traded. 'smooth' and "
                "'viterbi' condition on the whole sequence: cleaner labels, not available in real time."
            ),
        )

    # Conditionally-rendered widgets can have their session_state entry dropped
    # on a run where they aren't drawn, so read every value with a fallback.
    st.session_state["regime_settings"] = {
        key[len("rg_"):]: st.session_state.get(key, default)
        for key, default in REGIME_DEFAULTS.items()
    }

# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
page = st.navigation(
    {
        # Ungrouped pages render above the named sections, so the onboarding
        # page is the first thing a new user sees and the landing default.
        "": [
            st.Page("app_pages/start_here.py", title="Start here",
                    icon=":material/rocket_launch:", default=True),
        ],
        "Research": [
            st.Page("app_pages/backtest_lab.py", title="Backtest", icon=":material/query_stats:"),
            st.Page("app_pages/regimes.py", title="Regimes", icon=":material/layers:"),
            st.Page("app_pages/adaptive_lab.py", title="Adaptive", icon=":material/tune:"),
            st.Page("app_pages/ml_lab.py", title="ML lab", icon=":material/network_intelligence:"),
            st.Page("app_pages/validation.py", title="Validation", icon=":material/fact_check:"),
        ],
        "Training": [
            st.Page("app_pages/exercises_lab.py", title="Exercises", icon=":material/assignment:"),
            st.Page("app_pages/learn.py", title="Learn", icon=":material/menu_book:"),
        ],
    },
    position="sidebar",
)

st.title(page.title)
page.run()
