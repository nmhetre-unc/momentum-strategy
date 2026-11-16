"""
Headless smoke test of every dashboard page (streamlit.testing AppTest):
each page renders without an exception or an error element, including
the states that depend on learning strategies (their out-of-sample
scoring window, rolling folds that can't hold a position).

Runs offline: synthetic prices go straight into session state, so the
app never fetches data.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from qbt.adaptive import ADAPTIVE_STRATEGIES  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
N_DAYS = 2000


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2012-01-02", periods=N_DAYS)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, N_DAYS))
    spread = close * 0.005
    return pd.DataFrame({
        "Close": close, "High": close + spread, "Low": close - spread,
        "Open": close, "Volume": rng.integers(50_000_000, 150_000_000, N_DAYS).astype(float),
    }, index=index)


PRICES = _prices()


def _render(page: str, state: dict | None = None,
            before_page_run: Callable[[], None] | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["prices"] = PRICES
    at.session_state["ticker"] = "SYN"
    for key, value in (state or {}).items():
        at.session_state[key] = value
    at.run()
    at.switch_page(page)
    if before_page_run:
        before_page_run()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


PAGE_TITLES = {
    "pages/backtest_lab.py": "Backtest",
    "pages/regimes.py": "Regimes",
    "pages/adaptive_lab.py": "Adaptive",
    "pages/ml_lab.py": "ML lab",
    "pages/validation.py": "Validation",
}


@pytest.mark.parametrize("page", list(PAGE_TITLES))
def test_every_page_renders(page):
    _render(page)


@pytest.mark.parametrize("page", list(PAGE_TITLES))
def test_every_page_has_exactly_one_title(page, monkeypatch):
    """Each page sets its own st.title, and nothing else adds one.

    Counts st.title calls rather than reading at.title, which keeps only
    one of two titles written before and inside page.run() while a browser
    shows both. Two runs are checked:

    - the first, the default page reached through app.py: this is where a
      title written by app.py itself shows up (it did, as a duplicate);
    - the run after switch_page, which AppTest makes by running the page
      script directly: the page's own title, once.
    """
    import streamlit as st

    titles: list[str] = []
    through_app: list[str] = []
    real_title = st.title

    def spy(body, *args, **kwargs):
        titles.append(body)
        return real_title(body, *args, **kwargs)

    def first_run_done():
        through_app.extend(titles)
        titles.clear()

    monkeypatch.setattr(st, "title", spy)
    at = _render(page, before_page_run=first_run_done)
    assert through_app == [PAGE_TITLES["pages/backtest_lab.py"]]
    assert titles == [PAGE_TITLES[page]]
    assert [t.value for t in at.title] == [PAGE_TITLES[page]]


@pytest.mark.parametrize("tab", ["Where it broke", "Compare strategies"])
def test_every_validation_tab_renders(tab):
    _render("pages/validation.py", {"v_tabs": tab})


@pytest.mark.parametrize("name", list(ADAPTIVE_STRATEGIES))
def test_adaptive_page_renders_every_strategy(name):
    _render("pages/adaptive_lab.py", {"ad_strategy": name})


def test_learning_strategy_is_scored_on_its_out_of_sample_window():
    at = _render("pages/backtest_lab.py", {"bt_strategy": "ml_direction"})
    assert any("out-of-sample window" in c.value for c in at.caption)


def test_fixed_model_rolling_counts_only_folds_that_can_hold_a_position():
    at = _render("pages/validation.py",
                 {"v_strategy": "ml_direction", "v_refit_per_fold": False,
                  "v_train": 504, "v_test": 126})
    folds = next(m.value for m in at.metric if m.label == "Folds")
    scored, total = (int(x) for x in folds.split(" of "))
    assert 0 < scored < total
