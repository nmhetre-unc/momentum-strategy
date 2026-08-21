"""
Interactive dashboard on top of the existing backtest engine.
Run with: streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import fetch_ohlcv
from strategies import STRATEGIES
from backtest import run_backtest
from analytics import full_report, drawdown_series
from walk_forward import evaluate_out_of_sample

st.set_page_config(page_title="Quant Strategy Dashboard", layout="wide")

st.title("Quant Strategy Dashboard")
st.caption(
    "Backtest rule-based trading strategies with risk-adjusted performance "
    "metrics and walk-forward validation."
)

# ---------- Sidebar: configuration ----------
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker", value="SPY").upper().strip()

    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", value=pd.to_datetime("2015-01-01"))
    end = col2.date_input("End date", value=pd.to_datetime("2025-01-01"))

    strategy_name = st.selectbox("Strategy", list(STRATEGIES.keys()))

    st.subheader("Parameters")
    params = {}
    if strategy_name == "sma_crossover":
        params["short_window"] = st.slider("Short SMA window", 5, 100, 50)
        params["long_window"] = st.slider("Long SMA window", 50, 300, 200)
    elif strategy_name == "momentum":
        params["lookback"] = st.slider("Lookback (days)", 5, 60, 20)
        params["threshold"] = st.slider("Return threshold", -0.05, 0.05, 0.0, step=0.005)
    elif strategy_name == "mean_reversion":
        params["period"] = st.slider("RSI period", 5, 30, 14)
        params["oversold"] = st.slider("Oversold threshold", 10, 40, 30)
        params["overbought"] = st.slider("Overbought threshold", 60, 90, 70)

    show_walk_forward = st.checkbox("Show walk-forward validation", value=True)
    run_clicked = st.button("Run Backtest", type="primary", use_container_width=True)


# ---------- Cached data fetch ----------
@st.cache_data(show_spinner=False)
def load_data(ticker: str, start, end) -> pd.DataFrame:
    return fetch_ohlcv(ticker, str(start), str(end))


# ---------- Chart + table builders ----------
def metric_table(stats: dict) -> pd.DataFrame:
    rows = [
        ("Total Return", f"{stats['total_return']:.2%}"),
        ("CAGR", f"{stats['cagr']:.2%}"),
        ("Annualized Volatility", f"{stats['annualized_volatility']:.2%}"),
        ("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}"),
        ("Sortino Ratio", f"{stats['sortino_ratio']:.2f}"),
        ("Max Drawdown", f"{stats['max_drawdown']:.2%}"),
        ("Number of Trades", f"{stats['num_trades']}"),
        ("Win Rate", f"{stats['win_rate']:.2%}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def equity_chart(result: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=result["equity_curve"], name="Strategy"))
    fig.add_trace(go.Scatter(
        x=result.index, y=result["benchmark_curve"], name="Buy & Hold", line=dict(dash="dash")
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Growth of $1", hovermode="x unified",
    )
    return fig


def drawdown_chart(result: pd.DataFrame) -> go.Figure:
    dd = drawdown_series(result["equity_curve"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.index, y=dd, fill="tozeroy", name="Drawdown"))
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Drawdown", yaxis_tickformat=".0%",
    )
    return fig


# ---------- Run + persist across reruns ----------
# Streamlit reruns the whole script on every widget interaction, so we
# stash the last computed result in session_state -- otherwise moving a
# slider elsewhere on the page would wipe the chart you just generated.
if run_clicked:
    try:
        with st.spinner(f"Fetching {ticker} data..."):
            df = load_data(ticker, start, end)
    except Exception as e:
        st.error(f"Couldn't load data for {ticker}: {e}")
        st.stop()

    strategy_fn = STRATEGIES[strategy_name]
    signal = strategy_fn(df, **params)
    result = run_backtest(df, signal)

    st.session_state["result"] = result
    st.session_state["df"] = df
    st.session_state["strategy_fn"] = strategy_fn
    st.session_state["params"] = params
    st.session_state["ticker"] = ticker
    st.session_state["strategy_name"] = strategy_name

if "result" in st.session_state:
    result = st.session_state["result"]
    df = st.session_state["df"]
    stats = full_report(result)

    st.subheader(f"{st.session_state['ticker']} — {st.session_state['strategy_name']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{stats['total_return']:.1%}")
    c2.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
    c3.metric("Max Drawdown", f"{stats['max_drawdown']:.1%}")
    c4.metric("Win Rate", f"{stats['win_rate']:.1%}")

    st.plotly_chart(equity_chart(result), use_container_width=True)
    st.plotly_chart(drawdown_chart(result), use_container_width=True)
    st.dataframe(metric_table(stats), hide_index=True, use_container_width=True)

    if show_walk_forward:
        st.subheader("Walk-Forward Validation")
        wf = evaluate_out_of_sample(
            df, st.session_state["strategy_fn"], **st.session_state["params"]
        )
        st.caption(f"Split date: {wf['split_date']} (70% in-sample / 30% out-of-sample)")

        wf_col1, wf_col2 = st.columns(2)
        with wf_col1:
            st.markdown("**In-Sample**")
            st.dataframe(metric_table(wf["in_sample"]), hide_index=True, use_container_width=True)
        with wf_col2:
            st.markdown("**Out-of-Sample**")
            st.dataframe(metric_table(wf["out_sample"]), hide_index=True, use_container_width=True)

        sharpe_drop = wf["in_sample"]["sharpe_ratio"] - wf["out_sample"]["sharpe_ratio"]
        if sharpe_drop > 0.5:
            st.warning(
                f"Sharpe ratio dropped by {sharpe_drop:.2f} out-of-sample — "
                "possible sign of overfitting to the in-sample period."
            )
else:
    st.info("Configure a ticker and strategy in the sidebar, then click **Run Backtest**.")
