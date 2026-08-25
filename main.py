"""
Usage:
    python main.py --ticker SPY --strategy sma_crossover
    python main.py --ticker AAPL --strategy momentum --walk-forward
    python main.py --ticker MSFT --strategy mean_reversion --plot
"""

import argparse

from data_loader import fetch_ohlcv
from strategies import STRATEGIES
from backtest import run_backtest
from analytics import full_report
from walk_forward import evaluate_out_of_sample
from visualize import plot_equity_curve, plot_drawdown


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


def main():
    parser = argparse.ArgumentParser(description="Run a backtest for a given ticker and strategy.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. SPY")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES.keys())
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--walk-forward", action="store_true",
                         help="Split into in-sample/out-of-sample periods and evaluate both")
    parser.add_argument("--plot", action="store_true",
                         help="Save equity curve and drawdown charts as PNG files")
    parser.add_argument("--model-report", action="store_true",
                         help="For --strategy ml_direction: print train/test accuracy and feature importance")
    args = parser.parse_args()

    df = fetch_ohlcv(args.ticker, args.start, args.end)
    strategy_fn = STRATEGIES[args.strategy]
    signal = strategy_fn(df)
    result = run_backtest(df, signal)

    header = f"Strategy: {args.strategy} | Ticker: {args.ticker} | {args.start} to {args.end}"
    print(header)
    print_report("Full period", full_report(result))

    if args.walk_forward:
        wf = evaluate_out_of_sample(df, strategy_fn)
        print(f"\n--- Walk-forward split at {wf['split_date']} ---")
        print_report("In-sample", wf["in_sample"])
        print_report("Out-of-sample", wf["out_sample"])

    if args.plot:
        equity_path = f"{args.ticker}_{args.strategy}_equity.png"
        drawdown_path = f"{args.ticker}_{args.strategy}_drawdown.png"
        plot_equity_curve(result, header, out_path=equity_path)
        plot_drawdown(result, header, out_path=drawdown_path)
        print(f"\nSaved charts: {equity_path}, {drawdown_path}")

    if args.model_report:
        if args.strategy != "ml_direction":
            print("\n--model-report only applies to --strategy ml_direction")
        else:
            from ml_strategy import model_report
            report = model_report(df)
            print(f"\n--- Model diagnostics (split at {report['split_date']}) ---")
            print(f"  Train accuracy: {report['train_accuracy']:.2%}")
            print(f"  Test accuracy:  {report['test_accuracy']:.2%}")
            print(f"  Test confusion matrix [[TN, FP], [FN, TP]]: {report['test_confusion_matrix']}")
            print("  Top features by importance:")
            top_features = sorted(report["feature_importance"].items(), key=lambda kv: -abs(kv[1]))[:5]
            for name, value in top_features:
                print(f"    {name:20s} {value:+.4f}")


if __name__ == "__main__":
    main()
