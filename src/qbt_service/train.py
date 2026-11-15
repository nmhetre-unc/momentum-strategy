"""
Trains a monthly model on a stored price snapshot and registers it as a
new model version.

    python -m qbt_service.train SPY --start 2008-01-01 --end 2025-01-01

Into the Compose stack's database (a one-shot container of the image):

    docker compose run --rm train SPY --start 2008-01-01 --end 2025-01-01

The snapshot is pinned the way a backtest's is: the newest stored series
whose fetch range covers [start, end) (`--end` exclusive, as for ingest),
its bars in that range. The registry row keeps its id. Nothing is
downloaded; ingest first.

The default model is the one C2 found best: LightGBM on next-month
regime with the HMM features. It did not beat its baseline out of
sample (results/ml/c2_results.json), and the registry records that next
to it: training evaluates it with the C2 nested walk-forward on this
snapshot before fitting the served model on every labelled month (see
qbt.ml.train).

Reproducible: fixed seeds and deterministic LightGBM, so the same
snapshot, options and library versions give a byte-identical artifact.
Training that again finds the existing version instead of adding one.
"""

import argparse
import sys
import time
from datetime import date

from qbt.ml.train import FEATURE_SETS, SERVED_MODELS, TrainingConfig, train_for_serving
from qbt_service.db import make_engine
from qbt_service.prices import find_series, load_frame, normalize_ticker
from qbt_service.registry import register
from qbt_service.settings import Settings


def main(argv: list[str] | None = None,
         config_overrides: dict | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qbt_service.train", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker")
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, required=True,
                        help="YYYY-MM-DD, exclusive")
    parser.add_argument("--target", choices=["regime", "direction"], default="regime")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="base+hmm")
    parser.add_argument("--model", choices=list(SERVED_MODELS), default="lightgbm")
    parser.add_argument("--seed", type=int, default=0, help="LightGBM random_state")
    args = parser.parse_args(argv)
    if args.start >= args.end:
        parser.error("--start must be before --end")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    ticker = normalize_ticker(args.ticker)
    config = TrainingConfig(target=args.target, feature_set=args.features, model=args.model,
                            seed=args.seed, **(config_overrides or {}))

    engine = make_engine(Settings.from_env().database_url)
    try:
        with engine.connect() as conn:
            series_id = find_series(conn, ticker, args.start, args.end)
            if series_id is None:
                print(f"No price snapshot for {ticker} covers [{args.start}, {args.end}). "
                      f"Ingest one first: python -m qbt_service.ingest {ticker} "
                      f"--start {args.start} --end {args.end}", file=sys.stderr)
                return 1
            frame = load_frame(conn, series_id, args.start, args.end)
        print(f"{ticker}: training {config.model} on {config.target} ({config.feature_set}) "
              f"from price series {series_id}, {len(frame)} bars", flush=True)
        started = time.perf_counter()
        try:
            trained = train_for_serving(frame, config)
        except ValueError as exc:
            print(f"{ticker}: FAILED: {exc}", file=sys.stderr)
            return 1
        request = {"ticker": ticker, "start": args.start.isoformat(),
                   "end": args.end.isoformat()}
        with engine.begin() as conn:
            registered = register(conn, series_id, trained, request)
    finally:
        engine.dispose()

    oos, bar = trained.metrics["out_of_sample"], trained.metrics["bar"]
    verb = "registered model version" if registered.created else \
        "unchanged: identical to registered model version"
    print(f"{verb} {registered.version} (artifact sha256 {trained.artifact_sha256.hex()}, "
          f"{time.perf_counter() - started:.0f} s)")
    print(f"out of sample: accuracy {oos['accuracy']:.3f} over {oos['n_months']} months "
          f"({oos['first_test_decision']} to {oos['last_test_decision']}); baseline "
          f"{bar['name']} {bar['accuracy']:.3f}; "
          f"{'beats' if bar['beats'] else 'does NOT beat'} its baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
