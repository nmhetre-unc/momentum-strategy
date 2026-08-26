"""
Usage:
    python main.py --ticker SPY --strategy sma_crossover
    python main.py --ticker AAPL --strategy momentum --walk-forward
    python main.py --ticker MSFT --strategy mean_reversion --plot
    python main.py --ticker SPY --strategy ml_direction --walk-forward --model-report

    python main.py --ticker SPY --strategy sma_crossover --regimes
    python main.py --ticker SPY --strategy adaptive_ensemble --regimes --cost-bps 5
    python main.py --ticker SPY --strategy sma_crossover --rolling
    python main.py --ticker SPY --exercises

Output is deliberately plain ASCII so it renders correctly in a Windows
console as well as a POSIX terminal.
"""

import argparse

from adaptive import ADAPTIVE_STRATEGIES, ALL_STRATEGIES, describe_choices
from analytics import full_report, performance_by_regime
from backtest import run_backtest
from data_loader import fetch_ohlcv
from regime import REGIME_METHODS, detect_regimes, detect_regimes_walk_forward, regime_stability, regime_summary
from visualize import plot_equity_curve, plot_drawdown
from walk_forward import evaluate_out_of_sample, rolling_walk_forward


def print_report(label: str, stats: dict):
    print(f"\n{label}")
    print(f"  Total return:          {stats['total_return']:.2%}")
    print(f"  CAGR:                  {stats['cagr']:.2%}")
    print(f"  Annualized volatility: {stats['annualized_volatility']:.2%}")
    print(f"  Sharpe ratio:          {stats['sharpe_ratio']:.2f}")
    print(f"  Sortino ratio:         {stats['sortino_ratio']:.2f}")
    print(f"  Max drawdown:          {stats['max_drawdown']:.2%}")
    print(f"  Number of trades:      {stats['num_trades']}")
    print(f"  Win rate:              {stats['win_rate']:.2%}")
    print(f"  Exposure:              {stats['exposure']:.1%}")
    print(f"  Turnover:              {stats['turnover']:.1f}x/yr")


def main():
    parser = argparse.ArgumentParser(description="Run a backtest for a given ticker and strategy.")
    parser.add_argument("--ticker", help="Ticker symbol, e.g. SPY")
    parser.add_argument("--strategy", choices=ALL_STRATEGIES.keys())
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--walk-forward", action="store_true",
                         help="Split into in-sample/out-of-sample periods and evaluate both")
    parser.add_argument("--plot", action="store_true",
                         help="Save equity curve and drawdown charts as PNG files")
    parser.add_argument("--model-report", action="store_true",
                         help="For --strategy ml_direction: print train/test accuracy and feature importance")
    parser.add_argument("--cost-bps", type=float, default=0.0,
                         help="Transaction cost in basis points per unit of position change (try 5)")
    parser.add_argument("--regimes", action="store_true",
                         help="Detect market regimes and break performance down by regime")
    parser.add_argument("--regime-method", default="hmm", choices=REGIME_METHODS)
    parser.add_argument("--n-regimes", type=int, default=3)
    parser.add_argument("--regime-walk-forward", action="store_true",
                         help="Refit the regime model on an expanding window (the honest, non-leaky version)")
    parser.add_argument("--rolling", action="store_true",
                         help="Rolling walk-forward: many consecutive out-of-sample windows instead of one split")
    parser.add_argument("--exercises", action="store_true",
                         help="Run every automated exercise check against this ticker and print the results")
    args = parser.parse_args()

    if not args.ticker:
        parser.error("--ticker is required")
    df = fetch_ohlcv(args.ticker, args.start, args.end)

    # ---- Exercises mode: no strategy needed ----
    if args.exercises:
        from exercises import run_all

        print(f"Exercise checks for {args.ticker} ({args.start} to {args.end})\n")
        for _, row in run_all(df, args.ticker).iterrows():
            marker = "OK " if row["observed"] else ("-- " if row["observed"] is False else "ERR")
            print(f"[{marker}] {row['exercise']}")
            print(f"        {row['message']}\n")
        if not args.strategy:
            return

    if not args.strategy:
        parser.error("--strategy is required (or use --exercises on its own)")

    strategy_fn = ALL_STRATEGIES[args.strategy]

    # ---- Regime detection, if asked for ----
    regimes = None
    if args.regimes or args.strategy in ADAPTIVE_STRATEGIES:
        detect = detect_regimes_walk_forward if args.regime_walk_forward else detect_regimes
        kwargs = {"method": args.regime_method, "n_regimes": args.n_regimes}
        if not args.regime_walk_forward:
            kwargs["fit_frac"] = 0.6
        regimes = detect(df, **kwargs)

    strategy_params = {"regimes": regimes} if args.strategy in ADAPTIVE_STRATEGIES else {}
    signal = strategy_fn(df, **strategy_params)
    result = run_backtest(df, signal, cost_bps=args.cost_bps,
                          regimes=None if regimes is None else regimes.labels)

    header = f"Strategy: {args.strategy} | Ticker: {args.ticker} | {args.start} to {args.end}"
    if args.cost_bps:
        header += f" | costs {args.cost_bps:.0f}bps"
    print(header)
    print_report("Full period", full_report(result))

    # ---- Regime breakdown ----
    if regimes is not None:
        stability = regime_stability(regimes.labels)
        print(f"\n--- Regimes ({args.regime_method}, "
              f"{'walk-forward' if args.regime_walk_forward else 'fit_frac=0.6'}) ---")
        print(f"  Labelled days: {stability['labelled_days']}  episodes: {stability['n_episodes']}  "
              f"avg duration: {stability['avg_duration']:.0f}d  switches/yr: {stability['switches_per_year']:.1f}")
        if stability["avg_duration"] < 15:
            print("  WARNING: average episode under 15 days. These labels are flickering, not")
            print("           describing regimes. Raise the confirmation window or use fewer regimes.")
        if not regimes.causal:
            print("  WARNING: labels are not causal (the model saw the days it is labelling).")
            print("           Fine for describing history; invalid for the numbers below.")

        print("\n  What the asset did in each regime:")
        print(regime_summary(regimes, df)[
            ["name", "days", "ann_return", "ann_volatility", "max_drawdown"]
        ].round(3).to_string(index=False))

        print("\n  What the strategy did in each regime:")
        print(performance_by_regime(result, regimes.labels, regimes.names)[
            ["name", "days", "sharpe_ratio", "total_return", "max_drawdown", "exposure"]
        ].round(3).to_string(index=False))

        print("\n  Transition matrix (P(tomorrow | today)) -- read the diagonal:")
        print(regimes.transition_matrix().round(3).to_string())

        if args.strategy in ADAPTIVE_STRATEGIES and args.strategy in (
            "regime_switch", "adaptive_ensemble", "regime_filtered"
        ):
            described = describe_choices(df, regimes=regimes)
            print(f"\n  Auto-selection learned on data up to {described['learn_end'].date()}:")
            for regime_id, choice in sorted(described["choices"].items()):
                print(f"    {regimes.names.get(regime_id, regime_id):28s} -> {choice or 'no clear evidence'}")

    # ---- Validation ----
    if args.walk_forward:
        wf = evaluate_out_of_sample(df, strategy_fn, **strategy_params)
        print(f"\n--- Walk-forward split at {wf['split_date']} ---")
        print_report("In-sample", wf["in_sample"])
        print_report("Out-of-sample", wf["out_sample"])
        decay = wf["in_sample"]["sharpe_ratio"] - wf["out_sample"]["sharpe_ratio"]
        if decay > 0.5:
            print(f"\n  NOTE: Sharpe fell {decay:.2f} out-of-sample. The size of the gap is the")
            print("        finding, not the level of either number.")

    if args.rolling:
        rolling = rolling_walk_forward(df, strategy_fn, cost_bps=args.cost_bps, **strategy_params)
        print(f"\n--- Rolling walk-forward: {rolling['n_folds']} out-of-sample windows ---")
        print(f"  Folds positive:   {rolling['pct_folds_positive']:.0%}")
        print(f"  Median Sharpe:    {rolling['median_sharpe']:.2f}")
        print(f"  Sharpe std dev:   {rolling['sharpe_std']:.2f}")
        print(f"  Worst fold:       {rolling['worst_fold_sharpe']:.2f}")
        print("\n" + rolling["folds"][
            ["fold", "test_start", "test_end", "total_return", "sharpe_ratio", "max_drawdown"]
        ].round(3).to_string(index=False))
        print(f"\n  {rolling['fitted_note']}")

    if args.plot:
        equity_path = f"{args.ticker}_{args.strategy}_equity.png"
        drawdown_path = f"{args.ticker}_{args.strategy}_drawdown.png"
        plot_equity_curve(result, header, out_path=equity_path)
        plot_drawdown(result, header, out_path=drawdown_path)
        print(f"\nSaved charts: {equity_path}, {drawdown_path}")

    if args.model_report:
        if args.strategy not in ("ml_direction", "ml_regime_conditional"):
            print("\n--model-report only applies to --strategy ml_direction or ml_regime_conditional")
        else:
            from ml_strategy import model_report

            report_kwargs = {}
            if args.strategy == "ml_regime_conditional" and regimes is not None:
                report_kwargs = {"regimes": regimes.labels, "regime_mode": "conditional"}
            report = model_report(df, **report_kwargs)
            print(f"\n--- Model diagnostics (split at {report['split_date']}) ---")
            print(f"  Train accuracy: {report['train_accuracy']:.2%}")
            print(f"  Test accuracy:  {report['test_accuracy']:.2%}")
            print(f"  Base rate:      {report['test_base_rate']:.2%}  "
                  f"<- the bar to clear; below it the model added nothing")
            print(f"  Test confusion matrix [[TN, FP], [FN, TP]]: {report['test_confusion_matrix']}")
            print("  Top features by importance:")
            top_features = sorted(report["feature_importance"].items(), key=lambda kv: -abs(kv[1]))[:5]
            for name, value in top_features:
                print(f"    {name:20s} {value:+.4f}")
            if report.get("by_regime"):
                print("\n  Test accuracy by regime:")
                for row in report["by_regime"]:
                    name = regimes.names.get(row["regime"], row["regime"]) if regimes else row["regime"]
                    edge = row["test_accuracy"] - row["base_rate"]
                    print(f"    {str(name):28s} acc={row['test_accuracy']:.1%}  "
                          f"base={row['base_rate']:.1%}  edge={edge:+.1%}  days={row['test_days']}")


if __name__ == "__main__":
    main()
