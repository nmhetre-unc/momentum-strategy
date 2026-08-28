"""
Rolling walk-forward, regime-attributed decay, and a fair comparison of
everything.

This page carries the teaching layer for validation. Its whole purpose is
to make one habit automatic: **assume your backtest is wrong until you
have tried to break it**, and treat the gap between in-sample and
out-of-sample as the number that matters rather than either level. Every
control, chart and table from the plain version is unchanged.

Note on structure: the heavy work sits inside `if tab.open:` guards, so
only the selected tab computes. The onboarding above the tabs is
deliberately cheap -- it renders on every run.
"""

import numpy as np
import pandas as pd
import streamlit as st

from adaptive import ADAPTIVE_STRATEGIES, ALL_STRATEGIES
from regime_dashboard import (caveat, chart_caption, common_mistakes, comparison_chart,
    COMPARISON_CONFIG, explainer, fold_chart, how_to_read, next_steps, page_intro,
    PERFORMANCE_CONFIG, quant_note, require_regimes, table_caption
)
from walk_forward import compare_strategies, evaluate_with_regimes, rolling_walk_forward

df, regimes = require_regimes()

page_intro("validation")
common_mistakes("validation")

# --------------------------------------------------------------------------
# Onboarding: what validation is and why one number isn't enough
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        "#### New here? Validation in ninety seconds\n"
        "**Validation** is the discipline of checking whether a strategy's performance "
        "survives on data that played no part in building it. It is the difference between "
        "research and curve-fitting with extra steps."
    )

    intro_left, intro_right = st.columns(2)
    with intro_left:
        st.markdown(
            "**Why a fixed-history backtest isn't evidence**\n\n"
            "You have parameters. You have one finite price history. *Some* setting of those "
            "parameters was necessarily the best on that history — so finding it and reporting "
            "its performance reports the maximum of a search, not the expected performance of "
            "a process.\n\n"
            "This holds even for a perfectly correct backtest, and even if you never ran an "
            "optimizer. Picking 50 and 200 because they're conventional is still a choice "
            "informed by decades of other people searching the same data."
        )
    with intro_right:
        st.markdown(
            "**In-sample, out-of-sample, and the gap**\n\n"
            "**In-sample (IS)** is the period your strategy could have been influenced by — "
            "parameters chosen, models fitted, charts looked at. **Out-of-sample (OOS)** is "
            "data that played no part in any of that.\n\n"
            "The split is always **chronological**, never random: neighbouring days share "
            "volatility regimes and overlapping rolling windows, so a randomly held-out "
            "Tuesday between two training days isn't held out at all."
        )

    st.markdown("**The gap is the metric — not either level**")
    st.markdown(
        "| Pattern | Reading |\n|---|---|\n"
        "| IS 1.2 → OOS 1.0 | Small gap. The process generalizes. **This is the good outcome.** |\n"
        "| IS 1.8 → OOS 0.2 | Collapse. The in-sample number was mostly fitting. |\n"
        "| IS 1.8 → OOS −0.4 | Sign flip. What was learned is actively wrong on new data. |\n"
        "| IS 0.3 → OOS 0.4 | No gap, but nothing to generalize either. Honest and unremarkable. |\n"
        "\n"
        "A **small gap at a modest level** beats a **large level with a large gap**, every time. "
        "The first describes a process you can repeat; the second describes one lucky period."
    )

    st.info(
        "**Why one split is not enough, and rolling walk-forward is the standard.** A single "
        "70/30 split gives you exactly one out-of-sample number — and one number cannot tell "
        "*this works* from *this was tested on a friendly stretch of market*. Rolling "
        "walk-forward slides the window through history and produces ten or fifteen "
        "consecutive holdouts instead. That gives you a **distribution**: the share of folds "
        "positive, the spread between them, and — the part people skip — the *sequence*, which "
        "is the only way to see a strategy that worked until 2018 and never again.",
        icon=":material/repeat:",
    )

with st.expander("How to read this page", icon=":material/map:"):
    st.markdown(
        """
**The single split** (Where it broke tab) — history cut chronologically at 70%. Everything
before is in-sample, everything after is out-of-sample.

- **IS vs OOS metrics** — compare them as a pair. Neither number means much alone.
- **Sharpe decay** — in-sample minus out-of-sample. Positive means performance fell. Under
  about 0.5 is normal; a collapse or sign flip means the in-sample figure was fitted.
- **Drawdown changes** — read these too. A strategy that keeps its return but doubles its
  drawdown out-of-sample has still degraded, and Sharpe alone can hide that.
- **Trade count** — the caveat people skip. Two trades out-of-sample is two coin flips, and
  no Sharpe computed on it means anything in either direction.

**Rolling walk-forward** (first tab) — many consecutive out-of-sample blocks instead of one.

- **Folds** — how many test windows fitted in your history. `Train days` sets the lookback,
  `Test days` the size of each holdout block.
- **Folds positive** — the share above zero. More informative than the average: ten of
  twelve positive beats a high mean driven by two enormous folds.
- **Fold Sharpe distribution** — each bar is one out-of-sample window. Individual bars are
  noisy (a 126-day fold has an error bar near ±1.4); the *pattern* is the signal.
- **Stitched OOS equity** — only the out-of-sample days, concatenated into one record. The
  closest thing here to a paper trading log. Read its shape, not its endpoint.
- **Fold-by-fold detail** — the same folds as a table, in time order. Read the sequence:
  positive early and negative late is regime drift, not noise.

**Strategy comparison** (third tab) — every strategy on identical data, dates, costs and
split, plotted as in-sample against out-of-sample Sharpe.

- Points **below the diagonal** decayed; points far below were fitted; points in the
  **bottom-left** never worked at all.
- The cost slider is not decoration — drag it and watch the ranking reorder. The post-cost
  ranking is the real one.
"""
    )

quant_note("walkforward_reason")
quant_note("walk_forward", expanded=True)
explainer(
    "Validation, as training for a race",
    "walk-forward is running practice laps; the IS→OOS gap is how well your training "
    "transfers to race day; each fold is a different set of weather conditions you have to "
    "survive; and the stitched curve is your actual track record.",
    """
**Practice laps.** You don't judge a runner by their best training session on their favourite
course. You judge them by whether the training transfers. Walk-forward is the same idea:
build on one stretch of history, test on another you never touched.

**The transfer gap.** A runner who posts brilliant training times and mediocre race times has
trained for the training, not the race. A strategy with IS Sharpe 1.8 and OOS 0.2 did the
same thing. And note which number matters — nobody gets a medal for practice.

**Folds as weather.** One race in perfect conditions tells you little. You want to see the
runner in heat, in rain, uphill. Each walk-forward fold is a different market environment,
and what you're checking is whether they finish all of them respectably — not whether they
won one spectacularly.

**The track record.** The stitched out-of-sample curve strings together only the races,
never the practice. It's the closest thing here to what you would actually have experienced,
and its flat stretches are periods you'd have had to sit through without knowing they'd end.

**And the way to ruin all of it:** looking at the race result, going back to change your
training, and then reporting the race time as though it were still an independent test. See
the note on silent fitting below — it is the one bias here with no technical fix.
""",
)
quant_note("silent_fitting")

# Keyed so the selected tab survives a rerun (changing a control in one tab
# no longer bounces you back to the first), and so the two lazily-computed
# tabs are addressable from tests.
tab_rolling, tab_decay, tab_compare = st.tabs(
    ["Rolling walk-forward", "Where it broke", "Compare strategies"],
    on_change="rerun", key="v_tabs",
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
            chart_caption(
                "Out-of-sample Sharpe for each consecutive walk-forward fold.",
    "Each bar is one test window the strategy had never seen.",
    "most bars above zero, and no downward trend from left to right — the sequence matters as much as the spread.",
            )

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

        folds = rolling["folds"]
        negative_folds = int((folds["sharpe_ratio"] <= 0).sum())
        fold_error = np.sqrt(252 / max(int(test_days), 1))

        if 0.5 <= rolling["pct_folds_positive"] < 0.65 and negative_folds >= 3:
            caveat(
                f"**{negative_folds} of {rolling['n_folds']} folds were negative.** The strategy "
                f"was above water in a bare majority of windows. Read that as a coin flip with a "
                f"small tilt rather than as a working strategy — and note that a fold this long "
                f"carries a Sharpe error bar of roughly ±{fold_error:.1f} on its own.",
                level="info",
            )

        # Is there a trend across the fold sequence? That's regime drift, and
        # it's invisible in the average. Correlate fold Sharpe against fold order.
        if rolling["n_folds"] >= 6:
            order = np.arange(len(folds))
            trend = np.corrcoef(order, folds["sharpe_ratio"].to_numpy())[0, 1]
            if not np.isnan(trend) and trend < -0.5:
                first_half = folds["sharpe_ratio"].iloc[: len(folds) // 2].mean()
                second_half = folds["sharpe_ratio"].iloc[len(folds) // 2:].mean()
                caveat(
                    f"**Performance is trending down across the fold sequence** (correlation "
                    f"{trend:.2f} between fold order and Sharpe; first half averaged "
                    f"{first_half:.2f}, second half {second_half:.2f}). That is regime drift, not "
                    f"noise — the strategy worked early and stopped. Averaging across the folds "
                    f"produces a number describing a market that no longer exists."
                )

        st.markdown("**Stitched out-of-sample record**")
        st.line_chart(
            pd.DataFrame({"Out-of-sample equity": rolling["oos_equity"]}),
            height=240, color="#2a78d6",
        )
        st.caption(rolling["fitted_note"])

        oos_equity = rolling["oos_equity"]
        if len(oos_equity) > 1 and oos_equity.iloc[-1] <= 1.0:
            caveat(
                f"**The stitched out-of-sample record ends at {oos_equity.iloc[-1]:.2f}x** — flat "
                f"or below where it started. Stringing together only the days this strategy was "
                f"genuinely being tested, it did not make money. That is the closest thing here to "
                f"a paper trading record, and it is the number to quote."
            )

        quant_note("fold_uncertainty")
        how_to_read(
            f"""
- **Read the share positive before the average.** {rolling['pct_folds_positive']:.0%} of folds
  were above zero here. Ten of twelve positive is a better bet than a high mean driven by two
  spectacular folds.
- **Individual bars are almost uninformative.** A {int(test_days)}-day fold carries a Sharpe
  error bar of about ±{fold_error:.1f}. The pattern across bars is the signal, never any one bar.
- **Read the bars left to right, not as a set.** Consistently positive early and negative
  late is regime drift. That is a completely different finding from an even scatter, and the
  average hides it.
- **The worst fold is the one you'd have lived through.** Ask whether you would still have
  been running the strategy afterwards. If not, the later folds were never available to you.
- **The stitched curve's shape matters more than its endpoint.** Long flat stretches are
  periods you would have had to sit through without knowing they would end.
"""
        )

        with st.expander("Fold-by-fold detail", icon=":material/table_rows:"):
            table_caption(
                "Every walk-forward fold in time order.",
                "Scan the trades column first, then check whether negative folds cluster in one period.",
            )
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
            st.markdown(
                "**How to interpret this:** scan the `Trades` column first — folds with two or "
                "three trades cannot support any conclusion, whatever their Sharpe says. Then "
                "read `From` down the page and ask whether the negative folds cluster in a "
                "particular period. Clustered failures are a regime story; scattered ones are "
                "noise."
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

        is_stats, oos_stats = evaluation["in_sample"], evaluation["out_sample"]
        decay = is_stats["sharpe_ratio"] - oos_stats["sharpe_ratio"]

        if decay > 0.5:
            caveat(
                f"**Sharpe fell {decay:.2f} out-of-sample** "
                f"({is_stats['sharpe_ratio']:.2f} → {oos_stats['sharpe_ratio']:.2f}). A gap this "
                f"size means the in-sample number was substantially a description of that "
                f"specific period rather than of the strategy. The size of the gap is the "
                f"finding, not the level of either number."
            )
        if oos_stats["sharpe_ratio"] < 0.3:
            caveat(
                f"**Out-of-sample Sharpe of {oos_stats['sharpe_ratio']:.2f}.** Below about 0.3 "
                f"there is no meaningful risk-adjusted return here — whatever the in-sample "
                f"figure showed, this strategy did not work on data it hadn't seen.",
                level="info",
            )
        if oos_stats["num_trades"] < 10:
            caveat(
                f"**Only {oos_stats['num_trades']} trades out-of-sample.** Every metric in this "
                f"section describes that many independent bets. No conclusion — positive or "
                f"negative — is supportable on that evidence. Widen the date range or use a "
                f"faster parameter set.",
                level="info",
            )
        if oos_stats["max_drawdown"] < is_stats["max_drawdown"] - 0.10:
            caveat(
                f"**Drawdown deepened out-of-sample**, from {is_stats['max_drawdown']:.1%} to "
                f"{oos_stats['max_drawdown']:.1%}. A strategy can keep its return and still "
                f"degrade — read drawdown alongside Sharpe rather than instead of it.",
                level="info",
            )

        quant_note("is_vs_oos")
        how_to_read(
            """
- **Read the gap, not the levels.** Under about 0.5 of decay is normal. A collapse means the
  in-sample number was fitted; a sign flip means what was learned is actively wrong now.
- **Check the trade count before anything else.** Under ten out-of-sample trades and the
  comparison cannot support a conclusion in either direction.
- **Then check drawdown.** Return can hold up while risk doubles. That is still degradation.
- **Then read the regime mix below** — it usually settles which of the two stories you're in.
"""
        )

        st.markdown("**Regime mix — read this first**")
        mix = evaluation["regime_mix"].copy()
        mix.index = [evaluation["names"].get(i, str(i)) for i in mix.index]
        mix["change"] = mix["out_sample"] - mix["in_sample"]
        table_caption(
            "How the regime mix shifted between the in-sample and out-of-sample periods.",
            "A large shift means the market changed — check the per-regime numbers before blaming the strategy.",
        )
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
            table_caption(
                "In-sample performance, split by regime.",
                "Compare each row against its twin in the out-of-sample table beside it.",
            )
            st.dataframe(evaluation["in_sample_by_regime"].drop(columns=["regime"]), hide_index=True,
                         column_config=PERFORMANCE_CONFIG, key="v_is_regime")
        with mix_right:
            st.markdown("**Out-of-sample, by regime**")
            table_caption(
                "Out-of-sample performance, split by regime.",
                "If these numbers held and only the shares moved, the strategy is intact and the market changed.",
            )
            st.dataframe(evaluation["out_sample_by_regime"].drop(columns=["regime"]), hide_index=True,
                         column_config=PERFORMANCE_CONFIG, key="v_oos_regime")

        how_to_read(
            """
Two stories produce the same falling Sharpe, and these tables separate them.

- **Story A — the mix changed.** Per-regime performance is roughly intact, but the
  out-of-sample period contained more of the regime this strategy dislikes. The strategy is
  fine; your expectations were built on a biased sample of history.
- **Story B — the strategy decayed.** Per-regime performance itself deteriorated. It now
  loses money in the regime it used to profit from. No amount of regime timing fixes that.

Compare the same regime's row across the two tables. If the numbers held and only the shares
moved, you're in A. If the numbers fell within regimes, you're in B — and B is the one that
means the strategy is finished.
"""
        )

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

        table_caption(
            "Every strategy on identical data, dates, costs and split.",
            "Read the whole table — picking the best out-of-sample row makes that number in-sample.",
        )
        st.dataframe(
            table.drop(columns=["error"]), hide_index=True,
            column_config=COMPARISON_CONFIG, key="v_comparison",
        )

        failed = table[table["error"].notna()]
        if not failed.empty:
            with st.expander("Strategies that errored", icon=":material/error:"):
                table_caption(
                    "Strategies that raised an error rather than producing a result.",
                    "Usually a date range too short for that strategy's warm-up period.",
                )
                st.dataframe(failed[["strategy", "error"]], hide_index=True, key="v_errors")

        scored = table.dropna(subset=["is_sharpe", "oos_sharpe"])
        if len(scored) >= 3:
            best_is = scored.loc[scored["is_sharpe"].idxmax(), "strategy"]
            best_oos = scored.loc[scored["oos_sharpe"].idxmax(), "strategy"]
            rank_correlation = scored["is_sharpe"].corr(scored["oos_sharpe"], method="spearman")

            if best_is != best_oos or (not pd.isna(rank_correlation) and rank_correlation < 0.3):
                caveat(
                    f"**The ranking does not survive the split.** The best in-sample strategy is "
                    f"**{best_is}**; the best out-of-sample one is **{best_oos}**. Rank "
                    f"correlation between the two columns is {rank_correlation:.2f} — near zero "
                    f"means in-sample performance carried almost no information about "
                    f"out-of-sample performance. Which is exactly why picking a strategy on "
                    f"in-sample results does not work."
                )
            if (scored["oos_sharpe"] <= 0).sum() >= len(scored) / 2:
                caveat(
                    f"**{int((scored['oos_sharpe'] <= 0).sum())} of {len(scored)} strategies have a "
                    f"non-positive out-of-sample Sharpe** at {compare_cost:.0f}bps. That is the "
                    f"honest base rate for this kind of work, and it is the context in which any "
                    f"single good result should be read.",
                    level="info",
                )

        quant_note("survivorship_bias")
        how_to_read(
            """
- **Read the whole table, never the top row.** Picking the best out-of-sample Sharpe from
  eleven candidates means you used the holdout to make the choice — that number is now
  in-sample. With eleven candidates on random data, the winner looks respectable by
  construction.
- **The diagonal is the reference line.** On it means no decay. Far below means the
  in-sample figure was fitted. Bottom-left means it never worked at all.
- **Compare the `Decay` column across strategies, not just the OOS level.** A strategy at
  0.5 with no decay is a better process than one at 0.7 that fell from 1.9.
- **Drag the cost slider from 0 to 15.** Watch which strategies collapse. High-turnover
  signals look best at zero cost and degrade fastest — the post-cost ranking is the only
  real one.
- **Check exposure before crediting anything.** A strategy invested 30% of the time isn't
  comparable to a fully invested one on raw return.
"""
        )

        st.warning(
            "**Do not read this table by picking the top row.** If you compare eleven strategies "
            "and report the best one's out-of-sample Sharpe, that number is no longer "
            "out-of-sample — you used the holdout period to make the choice. With eleven "
            "candidates on random data, the winner's Sharpe looks respectable by construction. "
            "The honest report is the whole table.",
            icon=":material/warning:",
        )

# --------------------------------------------------------------------------
# Where to go next

# --------------------------------------------------------------------------
# Where to go next
# --------------------------------------------------------------------------
next_steps("validation")
