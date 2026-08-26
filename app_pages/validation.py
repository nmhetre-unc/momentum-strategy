"""Rolling walk-forward, regime-attributed decay, and a fair comparison of everything."""

import pandas as pd
import streamlit as st

from adaptive import ADAPTIVE_STRATEGIES, ALL_STRATEGIES
from regime_dashboard import (
    COMPARISON_CONFIG, PERFORMANCE_CONFIG, caveat, comparison_chart, fold_chart,
    quant_note, require_regimes,
)
from walk_forward import compare_strategies, evaluate_with_regimes, rolling_walk_forward

df, regimes = require_regimes()

quant_note("walk_forward", expanded=True)

tab_rolling, tab_decay, tab_compare = st.tabs(
    ["Rolling walk-forward", "Where it broke", "Compare strategies"], on_change="rerun",
)

# --------------------------------------------------------------------------
# Rolling walk-forward
# --------------------------------------------------------------------------
if tab_rolling.open:
    with tab_rolling:
        st.markdown(
            "One split gives you one out-of-sample number, and one number cannot tell "
            "*this works* from *this got lucky once*. Consecutive out-of-sample blocks give "
            "you a distribution."
        )
        with st.container(border=True):
            controls = st.columns([2, 1, 1, 1])
            strategy_name = controls[0].selectbox("Strategy", list(ALL_STRATEGIES), key="v_strategy")
            train_days = controls[1].number_input("Train days", 252, 2520, 756, step=126, key="v_train")
            test_days = controls[2].number_input("Test days", 21, 504, 126, step=21, key="v_test")
            cost_bps = controls[3].number_input("Cost (bps)", 0.0, 50.0, 5.0, step=1.0, key="v_cost")

        # The adaptive wrappers take the sidebar's regime model; the base
        # strategies take no regime argument at all.
        params = {"regimes": regimes} if strategy_name in ADAPTIVE_STRATEGIES else {}

        try:
            rolling = rolling_walk_forward(
                df, ALL_STRATEGIES[strategy_name], train_days=int(train_days),
                test_days=int(test_days), cost_bps=cost_bps, **params,
            )
        except ValueError as exc:
            st.warning(str(exc), icon=":material/warning:")
            st.stop()

        summary = st.columns(4)
        summary[0].metric("Folds", rolling["n_folds"])
        summary[1].metric("Folds positive", f"{rolling['pct_folds_positive']:.0%}",
                          help="Consistency. More informative than the average — 10 of 12 positive beats a high mean off two huge folds.")
        summary[2].metric("Median fold Sharpe", f"{rolling['median_sharpe']:.2f}")
        summary[3].metric("Worst fold", f"{rolling['worst_fold_sharpe']:.2f}",
                          help="The fold you would actually have had to live through.")

        chart = fold_chart(rolling["folds"])
        if chart is not None:
            st.altair_chart(chart)

        if rolling["pct_folds_positive"] < 0.5:
            caveat(
                f"Only {rolling['pct_folds_positive']:.0%} of out-of-sample folds were positive. "
                f"Whatever the full-period backtest showed, this strategy did not work most of the "
                f"time it was tested."
            )
        elif rolling["sharpe_std"] > 1.0:
            caveat(
                f"Fold Sharpes have a standard deviation of {rolling['sharpe_std']:.2f} — the "
                f"result swings wildly between periods. That spread is your real uncertainty about "
                f"the strategy, and it is much larger than a single split suggests.",
                level="info",
            )

        st.markdown("**Stitched out-of-sample record**")
        st.line_chart(
            pd.DataFrame({"Out-of-sample equity": rolling["oos_equity"]}),
            height=240, color="#2a78d6",
        )
        st.caption(rolling["fitted_note"])

        with st.expander("Fold-by-fold detail", icon=":material/table_rows:"):
            st.dataframe(
                rolling["folds"], hide_index=True, key="v_folds",
                column_config={
                    "fold": st.column_config.NumberColumn("Fold"),
                    "test_start": st.column_config.DateColumn("From"),
                    "test_end": st.column_config.DateColumn("To"),
                    "total_return": st.column_config.NumberColumn("Return", format="percent"),
                    "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                    "max_drawdown": st.column_config.NumberColumn("Max DD", format="percent"),
                    "exposure": st.column_config.NumberColumn("Exposure", format="percent"),
                    "num_trades": st.column_config.NumberColumn("Trades"),
                },
                column_order=["fold", "test_start", "test_end", "total_return",
                              "sharpe_ratio", "max_drawdown", "exposure", "num_trades"],
            )

# --------------------------------------------------------------------------
# Regime-attributed decay
# --------------------------------------------------------------------------
if tab_decay.open:
    with tab_decay:
        st.markdown(
            "When performance decays out-of-sample there are two very different stories, and "
            "the regime mix usually settles which one you're in."
        )
        decay_strategy = st.selectbox("Strategy", list(ALL_STRATEGIES), key="v_decay_strategy")
        decay_params = {"regimes": regimes} if decay_strategy in ADAPTIVE_STRATEGIES else {}

        evaluation = evaluate_with_regimes(
            df, ALL_STRATEGIES[decay_strategy], regimes, cost_bps=5.0,
            strategy_params=decay_params,
        )

        headline = st.columns(3)
        headline[0].metric("In-sample Sharpe", f"{evaluation['in_sample']['sharpe_ratio']:.2f}")
        headline[1].metric("Out-of-sample Sharpe", f"{evaluation['out_sample']['sharpe_ratio']:.2f}")
        headline[2].metric(
            "Decay",
            f"{evaluation['in_sample']['sharpe_ratio'] - evaluation['out_sample']['sharpe_ratio']:+.2f}",
        )

        st.markdown("**Regime mix — read this first**")
        mix = evaluation["regime_mix"].copy()
        mix.index = [evaluation["names"].get(i, str(i)) for i in mix.index]
        mix["change"] = mix["out_sample"] - mix["in_sample"]
        st.dataframe(
            mix, key="v_mix",
            column_config={
                "in_sample": st.column_config.NumberColumn("In-sample share", format="percent"),
                "out_sample": st.column_config.NumberColumn("Out-of-sample share", format="percent"),
                "change": st.column_config.NumberColumn("Change", format="percent"),
            },
        )
        biggest = mix["change"].abs().max() if not mix.empty else 0
        if biggest > 0.15:
            st.info(
                f"The regime mix shifted by up to {biggest:.0%} between the two periods. Before "
                f"concluding the strategy broke, check whether its per-regime numbers held up — "
                f"if they did, the market changed, not the strategy.",
                icon=":material/lightbulb:",
            )

        mix_left, mix_right = st.columns(2)
        with mix_left:
            st.markdown("**In-sample, by regime**")
            st.dataframe(evaluation["in_sample_by_regime"].drop(columns=["regime"]), hide_index=True,
                         column_config=PERFORMANCE_CONFIG, key="v_is_regime")
        with mix_right:
            st.markdown("**Out-of-sample, by regime**")
            st.dataframe(evaluation["out_sample_by_regime"].drop(columns=["regime"]), hide_index=True,
                         column_config=PERFORMANCE_CONFIG, key="v_oos_regime")

        quant_note("risk_by_regime")

# --------------------------------------------------------------------------
# Fair comparison
# --------------------------------------------------------------------------
if tab_compare.open:
    with tab_compare:
        quant_note("fair_comparison", expanded=True)
        compare_cost = st.slider(
            "Transaction cost (bps)", 0.0, 25.0, 5.0, step=1.0, key="v_compare_cost",
            help="Drag this from 0 upward and watch the ranking reorder. The post-cost ranking is the real one.",
        )
        with st.spinner("Running every strategy on identical data..."):
            table = compare_strategies(
                df,
                {name: (fn, {"regimes": regimes} if name in ADAPTIVE_STRATEGIES else {})
                 for name, fn in ALL_STRATEGIES.items()},
                cost_bps=compare_cost,
            )

        chart = comparison_chart(table)
        if chart is not None:
            st.altair_chart(chart)
            st.caption(
                "Points below the diagonal decayed out-of-sample; points far below it were fitted. "
                "Points in the bottom-left quadrant never worked in the first place."
            )

        st.dataframe(
            table.drop(columns=["error"]), hide_index=True,
            column_config=COMPARISON_CONFIG, key="v_comparison",
        )

        failed = table[table["error"].notna()]
        if not failed.empty:
            with st.expander("Strategies that errored", icon=":material/error:"):
                st.dataframe(failed[["strategy", "error"]], hide_index=True, key="v_errors")

        st.warning(
            "**Do not read this table by picking the top row.** If you compare eleven strategies "
            "and report the best one's out-of-sample Sharpe, that number is no longer "
            "out-of-sample — you used the holdout period to make the choice. With eleven "
            "candidates on random data, the winner's Sharpe looks respectable by construction. "
            "The honest report is the whole table.",
            icon=":material/warning:",
        )
