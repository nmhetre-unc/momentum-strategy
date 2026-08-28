"""
Shared Streamlit rendering for the quant training dashboard.

Chart builders, table formatters and the teaching-note widgets live here
so the page scripts in app_pages/ stay short and readable. Named
regime_dashboard.py because the regime visualizations are the bulk of it,
but the equity/drawdown/metric helpers are shared by every page.

COLOR. Regimes are an ORDERED variable -- regime.py guarantees ID 0 is the
calmest and the highest ID the most violent -- so they get an ordinal
single-hue ramp rather than a set of unrelated categorical hues. Severity
maps to distance from the page surface: on a light theme the crisis regime
is the darkest blue, on a dark theme it's the brightest. Steps are taken
from a validated blue ramp and were checked for monotonic lightness, a
minimum lightness gap between adjacent steps, and contrast against both
surfaces. The k=5 ramp fails the lightness-gap check, which is why the
dashboard caps regime count at 4 -- and four regimes on fifteen years of
daily data is already close to over-segmentation.

Identity is never carried by color alone: every regime chart ships a
legend, tooltips, and a table view of the same numbers.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from analytics import drawdown_series
from quant_notes import METRIC_DOCS, QUANT_NOTES
from regime import UNKNOWN, regime_episodes

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
# Ordinal blue ramp: calm -> violent, expressed as increasing contrast
# against the surface. Validated in both modes for k = 2, 3 and 4.
REGIME_RAMP = {
    "light": {
        1: ["#3987e5"],
        2: ["#86b6ef", "#256abf"],
        3: ["#86b6ef", "#3987e5", "#184f95"],
        4: ["#86b6ef", "#5598e7", "#256abf", "#104281"],
    },
    "dark": {
        1: ["#3987e5"],
        2: ["#184f95", "#86b6ef"],
        3: ["#184f95", "#2a78d6", "#86b6ef"],
        4: ["#184f95", "#256abf", "#3987e5", "#86b6ef"],
    },
}

# Two-series categorical slots (strategy vs. benchmark) and single-series
# marks, from the same validated palette.
SERIES = {
    "light": {"strategy": "#2a78d6", "benchmark": "#eb6834", "drawdown": "#e34948",
              "muted": "#898781", "grid": "#e1e0d9"},
    "dark": {"strategy": "#3987e5", "benchmark": "#d95926", "drawdown": "#e66767",
             "muted": "#898781", "grid": "#2c2c2a"},
}

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# Four chart heights, used everywhere. A drawdown strip should not be as
# tall as the equity curve it sits under, but it should be exactly as tall
# as every other drawdown strip in the app.
CHART_TALL = 320      # primary time series: equity, regime ribbon
CHART_MEDIUM = 240    # secondary analysis: heatmaps, fold bars, scatters
CHART_SHORT = 180     # supporting strips: drawdown, probabilities, histograms
CHART_MINI = 120      # inline detail: position over time

MAX_REGIMES = 4
UNKNOWN_COLOR = {"light": "#c3c2b7", "dark": "#383835"}


def active_theme() -> str:
    """Whichever theme the viewer is actually looking at."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def regime_palette(n: int, theme: str = None) -> list:
    theme = theme or active_theme()
    ramp = REGIME_RAMP[theme]
    return ramp.get(min(max(n, 1), MAX_REGIMES), ramp[MAX_REGIMES])[:n]


def regime_color_scale(names: dict, theme: str = None) -> alt.Scale:
    """
    An ordinal color scale whose domain is in severity order, so the legend
    reads calm at the top and crisis at the bottom regardless of which
    regimes happen to appear in the visible window.
    """
    theme = theme or active_theme()
    ordered_ids = sorted(names)
    colors = regime_palette(len(ordered_ids), theme)
    domain = [names[i] for i in ordered_ids] + ["Warm-up"]
    return alt.Scale(domain=domain, range=colors + [UNKNOWN_COLOR[theme]])


def _ink(theme: str = None) -> dict:
    return SERIES[theme or active_theme()]


# --------------------------------------------------------------------------
# Teaching widgets
# --------------------------------------------------------------------------
def quant_note(key: str, expanded: bool = False):
    """Renders one collapsible note from quant_notes.QUANT_NOTES."""
    note = QUANT_NOTES.get(key)
    if note is None:
        return
    with st.expander(note["title"], icon=":material/school:", expanded=expanded):
        st.markdown(note["body"])


def page_intro(page_key: str):
    """
    The standard opening every page shares: a breadcrumb, a three-sentence
    "why this matters", and a pointer for anyone already lost.

    Rendered from page_guide.PAGE_GUIDE rather than written per page, so
    the eight pages cannot drift into different structures or voices.
    """
    from page_guide import PAGE_GUIDE, breadcrumb

    guide = PAGE_GUIDE[page_key]
    st.caption(f":material/my_location: You are here → {breadcrumb(page_key)}")

    with st.container(border=True):
        st.markdown(
            f"**Why this matters**  \n"
            f"{guide['teaches']} {guide['why']}"
        )
        st.caption(f"**The habit it builds:** {guide['habit']}")

    confused = guide["confused"]
    pointers = [f"read the quant note *{QUANT_NOTES[confused['note']]['title']}*"]
    if confused.get("glossary"):
        pointers.append(confused["glossary"].rstrip("."))
    if confused.get("exercise"):
        pointers.append(f"try the exercise *{confused['exercise']}*")

    with st.expander("If you're confused, start here", icon=":material/help:"):
        st.markdown(
            "Three places to look, in order:\n\n"
            + "\n".join(f"{i}. {p.capitalize() if p[0].islower() else p}."
                        for i, p in enumerate(pointers, start=1))
        )
        quant_note(confused["note"])


def common_mistakes(page_key: str):
    """The two or three mistakes beginners make on this specific page."""
    from page_guide import PAGE_GUIDE

    mistakes = PAGE_GUIDE[page_key].get("mistakes", [])
    if not mistakes:
        return
    with st.expander("Common mistakes on this page", icon=":material/error_outline:"):
        for title, body in mistakes:
            st.markdown(f"**{title}.** {body}")


def next_steps(page_key: str):
    """
    The closing "What to do next" grid. Destinations and copy come from the
    registry, so every page ends the same way and no link goes stale.
    """
    from page_guide import PAGE_GUIDE

    guide = PAGE_GUIDE[page_key]
    st.subheader("What to do next", divider="gray")
    st.caption("Each page answers a question this one raised but cannot settle on its own.")

    destinations = guide["next"]
    halfway = (len(destinations) + 1) // 2
    for column, keys in zip(st.columns(2), (destinations[:halfway], destinations[halfway:])):
        with column:
            for key in keys:
                target = PAGE_GUIDE[key]
                with st.container(border=True):
                    st.page_link(target["path"], label=f"**{target['title']}**", icon=target["icon"])
                    st.markdown(guide["next_blurbs"][key])


def chart_caption(shows: str, read: str, look_for: str):
    """
    One-line caption under every chart: what it shows, how to read it, what
    to look for. Same three-part shape everywhere, so the eye learns where
    to find each part.
    """
    st.caption(f":material/insights: **{shows}** {read} *Look for:* {look_for}")


def table_caption(summarises: str, interpret: str):
    """One-line caption above every table: what it summarises, how to read it."""
    st.caption(f":material/table_rows: **{summarises}** {interpret}")


def explainer(title: str, metaphor: str, body: str, icon: str = ":material/menu_book:"):
    """
    One collapsible teaching section, opening with a visual metaphor.

    The metaphor goes first because a concrete image is what makes the
    technical explanation underneath it stick -- and because someone
    skimming reads one line, not five.
    """
    with st.expander(title, icon=icon):
        st.markdown(f"> :material/lightbulb: **Picture it as:** {metaphor}")
        st.markdown(body)


def how_to_read(body: str, title: str = "How to interpret this"):
    """
    A short, actionable reading guide placed directly under a chart or
    table. Deliberately not an expander: the whole point is that it is
    visible at the moment someone is looking at the thing it describes.
    """
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(body)


def metric_row(stats: dict, keys: list, columns: int = 4):
    """
    A row of st.metric tiles with the tooltip for each pulled from
    METRIC_DOCS, so no number on this dashboard appears without an
    explanation attached to it.
    """
    formatters = {
        "total_return": lambda v: f"{v:.1%}", "cagr": lambda v: f"{v:.1%}",
        "annualized_volatility": lambda v: f"{v:.1%}", "max_drawdown": lambda v: f"{v:.1%}",
        "win_rate": lambda v: f"{v:.1%}", "exposure": lambda v: f"{v:.0%}",
        "sharpe_ratio": lambda v: f"{v:.2f}", "sortino_ratio": lambda v: f"{v:.2f}",
        "turnover": lambda v: f"{v:.1f}x", "num_trades": lambda v: f"{int(v)}",
    }
    labels = {
        "total_return": "Total return", "cagr": "CAGR", "annualized_volatility": "Volatility",
        "sharpe_ratio": "Sharpe", "sortino_ratio": "Sortino", "max_drawdown": "Max drawdown",
        "win_rate": "Win rate", "num_trades": "Trades", "exposure": "Exposure",
        "turnover": "Turnover",
    }
    cols = st.columns(min(columns, len(keys)))
    for i, key in enumerate(keys):
        value = stats.get(key)
        text = formatters.get(key, str)(value) if value is not None else "—"
        cols[i % len(cols)].metric(labels.get(key, key), text, help=METRIC_DOCS.get(key))


def metric_table(stats: dict) -> pd.DataFrame:
    """The full metric set as a two-column table, for side-by-side comparisons."""
    rows = [
        ("Total return", f"{stats['total_return']:.2%}"),
        ("CAGR", f"{stats['cagr']:.2%}"),
        ("Annualized volatility", f"{stats['annualized_volatility']:.2%}"),
        ("Sharpe ratio", f"{stats['sharpe_ratio']:.2f}"),
        ("Sortino ratio", f"{stats['sortino_ratio']:.2f}"),
        ("Max drawdown", f"{stats['max_drawdown']:.2%}"),
        ("Number of trades", f"{stats['num_trades']}"),
        ("Win rate", f"{stats['win_rate']:.2%}"),
        ("Exposure", f"{stats.get('exposure', 0):.0%}"),
        ("Turnover", f"{stats.get('turnover', 0):.1f}x"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def show_metric_table(stats: dict, key: str = None, caption: tuple = None):
    """
    The full metric set as a table. `caption` is an optional
    (summarises, interpret) pair rendered above it, so these tables carry
    the same captions as every other table in the app.
    """
    if caption:
        table_caption(*caption)
    st.dataframe(
        metric_table(stats), hide_index=True, key=key,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="small"),
        },
    )


def caveat(message: str, level: str = "warning"):
    """A short inline caution, phrased as something to check rather than an error."""
    getattr(st, level)(message, icon=":material/info:")


# --------------------------------------------------------------------------
# Core charts
# --------------------------------------------------------------------------
def equity_chart(result: pd.DataFrame, log_scale: bool = False, height: int = CHART_TALL) -> alt.Chart:
    """Strategy equity against buy-and-hold, both normalized to $1."""
    ink = _ink()
    data = pd.DataFrame({
        "Date": np.repeat(result.index, 2),
        "Series": ["Strategy", "Buy & hold"] * len(result),
        "Growth of $1": np.column_stack(
            [result["equity_curve"], result["benchmark_curve"]]
        ).ravel(),
    }).dropna()

    scale = alt.Scale(type="log") if log_scale else alt.Scale(zero=False)
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Growth of $1:Q", title="Growth of $1", scale=scale),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Strategy", "Buy & hold"],
                                range=[ink["strategy"], ink["benchmark"]]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "Series:N", scale=alt.Scale(domain=["Strategy", "Buy & hold"], range=[[1, 0], [5, 3]]),
                legend=None,
            ),
            tooltip=["Date:T", "Series:N", alt.Tooltip("Growth of $1:Q", format=".3f")],
        )
        .properties(height=height)
        .interactive(bind_y=False)
    )


def drawdown_chart(result: pd.DataFrame, height: int = CHART_SHORT) -> alt.Chart:
    """Depth below the running peak — read this with every equity curve."""
    ink = _ink()
    data = pd.DataFrame({
        "Date": result.index,
        "Drawdown": drawdown_series(result["equity_curve"]),
    }).dropna()

    return (
        alt.Chart(data)
        .mark_area(opacity=0.65, color=ink["drawdown"], line={"color": ink["drawdown"], "strokeWidth": 1})
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Drawdown:Q", title="Drawdown", axis=alt.Axis(format="%")),
            tooltip=["Date:T", alt.Tooltip("Drawdown:Q", format=".1%")],
        )
        .properties(height=height)
    )


def position_chart(result: pd.DataFrame, height: int = CHART_MINI) -> alt.Chart:
    """
    Position size over time. Worth showing for any adaptive strategy: a
    binary strategy is a square wave, a volatility-targeted one breathes,
    and seeing which you have is faster than reading the turnover number.
    """
    ink = _ink()
    data = pd.DataFrame({"Date": result.index, "Position": result["position"]}).dropna()
    return (
        alt.Chart(data)
        .mark_area(opacity=0.7, color=ink["strategy"])
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Position:Q", title="Position", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Date:T", alt.Tooltip("Position:Q", format=".2f")],
        )
        .properties(height=height)
    )


# --------------------------------------------------------------------------
# Regime charts
# --------------------------------------------------------------------------
def _episode_frame(regime_result, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Regime episodes as date intervals, with each interval extended to the
    next episode's start so the ribbon has no hairline gaps between blocks.
    """
    episodes = regime_episodes(regime_result.labels, regime_result.names)
    if episodes.empty:
        return episodes

    positions = {timestamp: i for i, timestamp in enumerate(index)}
    ends = []
    for _, row in episodes.iterrows():
        stop = positions.get(row["end"], len(index) - 1)
        ends.append(index[min(stop + 1, len(index) - 1)])
    episodes = episodes.copy()
    episodes["end_extended"] = ends
    return episodes


def regime_ribbon_chart(df: pd.DataFrame, regime_result, height: int = CHART_TALL,
                        log_scale: bool = True) -> alt.LayerChart:
    """
    Price with the detected regimes shaded behind it.

    The single most useful regime chart, because it lets you check the
    labels against your own reading of the chart. If the model calls
    2020 calm, you have learned something about the model, not about 2020.
    """
    ink = _ink()
    episodes = _episode_frame(regime_result, df.index)
    price = pd.DataFrame({"Date": df.index, "Price": df["Close"]}).dropna()

    line = (
        alt.Chart(price)
        .mark_line(strokeWidth=1.5, color=ink["strategy"])
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Price:Q", title="Price",
                    scale=alt.Scale(type="log" if log_scale else "linear", zero=False)),
            tooltip=["Date:T", alt.Tooltip("Price:Q", format=".2f")],
        )
    )
    if episodes.empty:
        return alt.layer(line).properties(height=height)

    bands = (
        alt.Chart(episodes)
        .mark_rect(opacity=0.30)
        .encode(
            x=alt.X("start:T", title=None),
            x2="end_extended:T",
            color=alt.Color("name:N", scale=regime_color_scale(regime_result.names),
                            legend=alt.Legend(title="Regime", orient="top", columns=2)),
            tooltip=[alt.Tooltip("name:N", title="Regime"), alt.Tooltip("start:T", title="From"),
                     alt.Tooltip("end:T", title="To"), alt.Tooltip("days:Q", title="Days")],
        )
    )
    return alt.layer(bands, line).resolve_scale(color="independent").properties(height=height)


def regime_probability_chart(regime_result, height: int = CHART_SHORT) -> alt.Chart:
    """
    Model confidence over time, as a stacked area of P(regime).

    Read the transitions: where the bands are cleanly separated the model
    is sure; where they interleave it is guessing, and those are exactly
    the days a regime-switching strategy acts on.
    """
    if regime_result.probabilities is None:
        return None
    probabilities = regime_result.probabilities.dropna(how="all")
    if probabilities.empty:
        return None

    tidy = probabilities.rename(columns=regime_result.names).reset_index()
    tidy = tidy.rename(columns={tidy.columns[0]: "Date"}).melt(
        id_vars="Date", var_name="Regime", value_name="Probability"
    ).dropna()

    return (
        alt.Chart(tidy)
        .mark_area(opacity=0.85)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Probability:Q", title="P(regime)", stack="normalize", axis=alt.Axis(format="%")),
            color=alt.Color("Regime:N", scale=regime_color_scale(regime_result.names),
                            legend=alt.Legend(title=None, orient="top", columns=2)),
            tooltip=["Date:T", "Regime:N", alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=height)
    )


def regime_feature_chart(regime_result, feature: str, height: int = CHART_SHORT) -> alt.Chart:
    """One regime feature over time, colored by the regime it helped produce."""
    if feature not in regime_result.features.columns:
        return None
    data = pd.DataFrame({
        "Date": regime_result.features.index,
        "Value": regime_result.features[feature],
        "Regime": regime_result.named_labels(),
    }).dropna()

    return (
        alt.Chart(data)
        .mark_circle(size=9, opacity=0.6)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Value:Q", title=feature),
            color=alt.Color("Regime:N", scale=regime_color_scale(regime_result.names),
                            legend=alt.Legend(title=None, orient="top", columns=2)),
            tooltip=["Date:T", "Regime:N", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(height=height)
    )


def transition_heatmap(matrix: pd.DataFrame, height: int = CHART_MEDIUM) -> alt.LayerChart:
    """
    P(tomorrow's regime | today's regime), as a heatmap with the numbers
    written on the cells.

    The diagonal is the whole story. Values of 0.95+ mean the model found
    persistent states; values near 1/k mean it found noise, and everything
    downstream of it is built on sand.
    """
    if matrix.empty:
        return None
    tidy = matrix.reset_index().melt(id_vars=matrix.index.name or "index",
                                     var_name="To", value_name="Probability")
    tidy = tidy.rename(columns={tidy.columns[0]: "From"}).dropna()

    base = alt.Chart(tidy).encode(
        x=alt.X("To:N", title="Tomorrow", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("From:N", title="Today"),
    )
    cells = base.mark_rect().encode(
        color=alt.Color("Probability:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                        legend=alt.Legend(title="P", format=".0%")),
        tooltip=["From:N", "To:N", alt.Tooltip("Probability:Q", format=".2%")],
    )
    # Numbers on every cell: the ramp's light steps sit below 3:1 against
    # the surface, so the value is never carried by the fill alone.
    labels = base.mark_text(fontSize=11).encode(
        text=alt.Text("Probability:Q", format=".2f"),
        color=alt.condition(alt.datum.Probability > 0.5, alt.value("white"), alt.value("#0b0b0b")),
    )
    return alt.layer(cells, labels).properties(height=height)


def performance_by_regime_chart(table: pd.DataFrame, metric: str = "sharpe_ratio",
                                names: dict = None, height: int = CHART_MEDIUM) -> alt.Chart:
    """Bar chart of one metric per regime, with the day count in the tooltip."""
    if table.empty:
        return None
    axis_format = "%" if metric in ("total_return", "ann_return", "max_drawdown", "win_rate", "exposure") else ".2f"
    return (
        alt.Chart(table)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("name:N", title=None, sort=list(table["name"]), axis=alt.Axis(labelAngle=-20)),
            y=alt.Y(f"{metric}:Q", title=metric.replace("_", " "), axis=alt.Axis(format=axis_format)),
            color=alt.Color("name:N", scale=regime_color_scale(names or dict(zip(table["regime"], table["name"]))),
                            legend=None),
            tooltip=[alt.Tooltip("name:N", title="Regime"), alt.Tooltip("days:Q", title="Days"),
                     alt.Tooltip(f"{metric}:Q", format=axis_format)],
        )
        .properties(height=height)
    )


def duration_histogram(episodes: pd.DataFrame, names: dict = None, height: int = CHART_SHORT) -> alt.Chart:
    """
    How long regimes last. If the mass is at the left edge — episodes of a
    handful of days — the labels are noise, not regimes.
    """
    if episodes.empty:
        return None
    return (
        alt.Chart(episodes)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("days:Q", bin=alt.Bin(maxbins=25), title="Episode length (trading days)"),
            y=alt.Y("count():Q", title="Episodes"),
            color=alt.Color("name:N", scale=regime_color_scale(names or {}),
                            legend=alt.Legend(title=None, orient="top", columns=2)),
            tooltip=[alt.Tooltip("name:N", title="Regime"), alt.Tooltip("count():Q", title="Episodes")],
        )
        .properties(height=height)
    )


def fold_chart(folds: pd.DataFrame, height: int = CHART_MEDIUM) -> alt.Chart:
    """
    Out-of-sample Sharpe per walk-forward fold.

    What you want is most bars above zero and none catastrophic. Two huge
    bars and eight flat ones means the strategy's headline number belongs
    to two specific market episodes.
    """
    if folds.empty:
        return None
    ink = _ink()
    data = folds.copy()
    data["positive"] = data["sharpe_ratio"] > 0
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("test_start:T", title="Out-of-sample window start"),
            y=alt.Y("sharpe_ratio:Q", title="Sharpe (out-of-sample)"),
            color=alt.Color("positive:N",
                            scale=alt.Scale(domain=[True, False], range=[ink["strategy"], ink["drawdown"]]),
                            legend=alt.Legend(title=None, labelExpr="datum.label == 'true' ? 'Positive' : 'Negative'",
                                              orient="top")),
            tooltip=[alt.Tooltip("fold:Q", title="Fold"), alt.Tooltip("test_start:T", title="From"),
                     alt.Tooltip("test_end:T", title="To"), alt.Tooltip("sharpe_ratio:Q", format=".2f"),
                     alt.Tooltip("total_return:Q", format=".1%")],
        )
        .properties(height=height)
    )


def comparison_chart(table: pd.DataFrame, height: int = CHART_TALL) -> alt.Chart:
    """
    In-sample against out-of-sample Sharpe for every strategy. Points below
    the diagonal decayed; points far below it were fitted.
    """
    data = table.dropna(subset=["is_sharpe", "oos_sharpe"])
    if data.empty:
        return None
    ink = _ink()
    limit = float(np.nanmax(np.abs(data[["is_sharpe", "oos_sharpe"]].to_numpy()))) * 1.15 or 1.0

    diagonal = (
        alt.Chart(pd.DataFrame({"x": [-limit, limit], "y": [-limit, limit]}))
        .mark_line(strokeDash=[4, 4], color=ink["muted"], strokeWidth=1)
        .encode(x="x:Q", y="y:Q")
    )
    points = (
        alt.Chart(data)
        .mark_circle(size=140, opacity=0.85, color=ink["strategy"])
        .encode(
            x=alt.X("is_sharpe:Q", title="In-sample Sharpe", scale=alt.Scale(domain=[-limit, limit])),
            y=alt.Y("oos_sharpe:Q", title="Out-of-sample Sharpe", scale=alt.Scale(domain=[-limit, limit])),
            tooltip=["strategy:N", alt.Tooltip("is_sharpe:Q", format=".2f"),
                     alt.Tooltip("oos_sharpe:Q", format=".2f"),
                     alt.Tooltip("turnover:Q", format=".1f")],
        )
    )
    labels = points.mark_text(align="left", dx=10, dy=-2, fontSize=11).encode(text="strategy:N")
    return alt.layer(diagonal, points, labels).properties(height=height)


# --------------------------------------------------------------------------
# Table formatters
# --------------------------------------------------------------------------
REGIME_SUMMARY_CONFIG = {
    "name": st.column_config.TextColumn("Regime"),
    "days": st.column_config.NumberColumn("Days", help="Sample size. Read this before believing any other number in the row."),
    "share": st.column_config.NumberColumn("Share", format="percent"),
    "episodes": st.column_config.NumberColumn("Episodes", help="Number of separate visits to this regime. One visit is an event, not a regime."),
    "avg_duration": st.column_config.NumberColumn("Avg days", format="%.0f"),
    "ann_return": st.column_config.NumberColumn("Ann. return", format="percent"),
    "ann_volatility": st.column_config.NumberColumn("Ann. vol", format="percent"),
    "max_drawdown": st.column_config.NumberColumn("Max DD", format="percent"),
    "pct_up_days": st.column_config.NumberColumn("Up days", format="percent"),
}

PERFORMANCE_CONFIG = {
    "name": st.column_config.TextColumn("Regime"),
    "days": st.column_config.NumberColumn("Days", help="Standard error on an annualized Sharpe is roughly sqrt(252/days)."),
    "total_return": st.column_config.NumberColumn("Return", format="percent"),
    "ann_return": st.column_config.NumberColumn("Ann. return", format="percent"),
    "annualized_volatility": st.column_config.NumberColumn("Ann. vol", format="percent"),
    "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f", help=METRIC_DOCS["sharpe_ratio"]),
    "sortino_ratio": st.column_config.NumberColumn("Sortino", format="%.2f"),
    "max_drawdown": st.column_config.NumberColumn("Max DD", format="percent"),
    "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
    "exposure": st.column_config.NumberColumn("Exposure", format="percent", help=METRIC_DOCS["exposure"]),
}

COMPARISON_CONFIG = {
    "strategy": st.column_config.TextColumn("Strategy", width="medium"),
    "is_sharpe": st.column_config.NumberColumn("IS Sharpe", format="%.2f"),
    "oos_sharpe": st.column_config.NumberColumn("OOS Sharpe", format="%.2f"),
    "sharpe_decay": st.column_config.NumberColumn("Decay", format="%.2f", help=METRIC_DOCS["sharpe_decay"]),
    "oos_return": st.column_config.NumberColumn("OOS return", format="percent"),
    "oos_max_dd": st.column_config.NumberColumn("OOS max DD", format="percent"),
    "oos_exposure": st.column_config.NumberColumn("Exposure", format="percent"),
    "turnover": st.column_config.NumberColumn("Turnover", format="%.1f", help=METRIC_DOCS["turnover"]),
}


def show_regime_health(regime_result, stability: dict):
    """
    The three questions to ask of any regime labelling, answered up front
    so nobody builds a strategy on labels that were never regimes.
    """
    with st.container(border=True):
        st.markdown("**Are these actually regimes?**")
        cols = st.columns(4)
        cols[0].metric("Episodes", stability["n_episodes"],
                       help="Separate visits to any regime. Very high means the labels are flickering.")
        cols[1].metric("Avg duration", f"{stability['avg_duration']:.0f}d",
                       help="Real regimes last weeks to months. Under ~15 days, you are looking at noise.")
        cols[2].metric("Switches / yr", f"{stability['switches_per_year']:.1f}",
                       help="How often a regime-switching strategy would flip its whole position.")
        cols[3].metric("Labelled days", stability["labelled_days"],
                       help="Days with a regime. The rest are warm-up, where the features don't exist yet.")

        if stability["avg_duration"] < 15 and stability["n_episodes"] > 0:
            caveat(
                f"Average episode is {stability['avg_duration']:.0f} days. That is short enough to "
                "be noise rather than regime structure — raise the smoothing window, or reduce the "
                "number of regimes, before building anything on these labels."
            )
        if not regime_result.causal:
            caveat(
                "These labels are **not causal**: the model was fitted on data it is also "
                "labelling, so they embed knowledge of the future. Fine for describing history, "
                "not for feeding a backtest. Set the fit fraction below 1.0, or switch on "
                "walk-forward detection, before quoting any performance number from them."
            )


# --------------------------------------------------------------------------
# Cached data + shared page state
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl="6h", max_entries=20)
def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Price history, cached so switching pages doesn't re-hit Yahoo."""
    from data_loader import fetch_ohlcv

    return fetch_ohlcv(ticker, start, end)


@st.cache_data(show_spinner=False, ttl="1h", max_entries=30)
def cached_regimes(df: pd.DataFrame, method: str, n_regimes: int, fit_frac: float,
                   smooth: str, min_duration: int, decode: str, walk_forward: bool):
    """
    Regime detection is the most expensive thing on this dashboard and its
    result is needed on four different pages, so it's cached on the exact
    parameter set. Changing any control refits; nothing else does.
    """
    from regime import detect_regimes, detect_regimes_walk_forward

    if walk_forward:
        return detect_regimes_walk_forward(
            df, method=method, n_regimes=n_regimes,
            smooth=smooth, min_duration=min_duration,
        )
    return detect_regimes(
        df, method=method, n_regimes=n_regimes, fit_frac=fit_frac,
        smooth=smooth, min_duration=min_duration, decode=decode,
    )


def require_data():
    """
    Every page needs prices. Rather than each one re-implementing the
    loading and error handling, they call this and get a DataFrame or a
    stopped script with an explanation.
    """
    df = st.session_state.get("prices")
    if df is None or df.empty:
        st.info(
            "Choose a ticker and date range in the sidebar, then select **Load data**.",
            icon=":material/database:",
        )
        st.stop()
    return df


def require_regimes():
    """Detects regimes with the sidebar's settings, or explains why it can't."""
    df = require_data()
    settings = st.session_state["regime_settings"]
    try:
        return df, cached_regimes(
            df, settings["method"], settings["n_regimes"], settings["fit_frac"],
            settings["smooth"], settings["min_duration"], settings["decode"],
            settings["walk_forward"],
        )
    except ValueError as exc:
        st.warning(str(exc), icon=":material/warning:")
        st.stop()
