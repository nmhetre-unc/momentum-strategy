"""
Tests for the monthly prediction package (qbt.ml.features, .labels,
.splits, .baselines, .evaluate) on synthetic prices; no network.

The lookahead canaries follow tests/test_causality.py: build a feature
from a price history, truncate the history at a cutoff, build it again,
and require every value on every day that survives the truncation to be
identical. A feature that used anything after day t would change on day
t when the future is cut off. Each feature gets its own canary, and
test_canary_catches_leaky_features proves the check fails on features
that do leak, so a pass means something.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from qbt.ml.baselines import majority_class, predict_majority, predict_previous
from qbt.ml.evaluate import (
    balanced_accuracy,
    build_dataset,
    evaluate_baselines,
    mcnemar_exact,
    strategy_returns,
    trading_summary,
    wilson_interval,
)
from qbt.ml.features import (
    BASE_FEATURES,
    HMM_FEATURES,
    daily_features,
    month_end_dates,
    monthly_features,
)
from qbt.ml.labels import (
    direction_labels,
    forward_returns,
    label_end_times,
    previous_labels,
    regime_labels,
)
from qbt.ml.splits import (
    EMBARGO_MONTHS,
    Fold,
    purge_and_embargo,
    purged_kfold_splits,
    walk_forward_splits,
)
from qbt.regime import UNKNOWN

N_DAYS = 1500
# Mid-month cutoffs, so truncation also leaves an incomplete final month.
CUTOFFS = ("2018-11-14", "2019-09-17")
HMM_KWARGS = {"initial_train": 252, "refit_every": 42}


def _prices(n: int = N_DAYS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Alternating calm and turbulent stretches, so regimes actually vary.
    vol = np.where((np.arange(n) // 200) % 2 == 0, 0.007, 0.02)
    returns = rng.normal(0.0004, vol)
    index = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"Close": 100 * np.cumprod(1 + returns)}, index=index)


def _mismatches(full: pd.DataFrame, cut: pd.DataFrame) -> dict[str, int]:
    """Per column, how many surviving days differ (NaN equals NaN)."""
    a = full.loc[cut.index, cut.columns]
    same = (a == cut) | (a.isna() & cut.isna())
    return {c: int((~same[c]).sum()) for c in cut.columns if (~same[c]).any()}


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return _prices()


@pytest.fixture(scope="module")
def daily_full_and_cut(prices):
    full = daily_features(prices, hmm_kwargs=HMM_KWARGS)
    cuts = [daily_features(prices.loc[:c], hmm_kwargs=HMM_KWARGS) for c in CUTOFFS]
    return full, cuts


# ---- lookahead canaries, one per feature ----

@pytest.mark.parametrize("column", BASE_FEATURES + HMM_FEATURES)
def test_feature_canary(daily_full_and_cut, column):
    full, cuts = daily_full_and_cut
    for cutoff, cut in zip(CUTOFFS, cuts, strict=True):
        assert column in cut.columns
        # The canary is only meaningful if the feature has values to compare.
        assert cut[column].notna().sum() >= 100, f"{column} is mostly warm-up before {cutoff}"
        bad = _mismatches(full[[column]], cut[[column]])
        assert not bad, f"{column} changed on {bad} days when data after {cutoff} was removed"


def test_monthly_feature_table_canary(prices):
    full = monthly_features(prices, include_hmm=False)
    for cutoff in CUTOFFS:
        cut = monthly_features(prices.loc[:cutoff], include_hmm=False)
        assert cut.index.isin(full.index).all()
        assert cut.index[-1] < pd.Timestamp(cutoff)  # the cut-off month is incomplete
        assert not _mismatches(full, cut)


def test_canary_catches_leaky_features(prices):
    def leaky(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        return pd.DataFrame({
            "next_day_return": close.shift(-1) / close - 1,
            "full_sample_zscore": (close - close.mean()) / close.std(),
            "centered_sma": close.rolling(21, center=True).mean(),
        })

    full = leaky(prices)
    cut = leaky(prices.loc[:CUTOFFS[0]])
    assert set(_mismatches(full, cut)) == {"next_day_return", "full_sample_zscore",
                                           "centered_sma"}


def test_features_are_nan_during_warm_up_not_filled(prices):
    feats = daily_features(prices, include_hmm=False)
    assert feats["ret_12m"].iloc[:252].isna().all()
    assert feats["ret_12m"].iloc[252:].notna().all()
    assert feats["mom_12_1"].iloc[:252].isna().all()
    assert feats["drawdown_12m"].iloc[:251].isna().all()
    assert (feats["drawdown_12m"].dropna() <= 0).all()


def test_mom_12_1_skips_the_latest_month(prices):
    feats = daily_features(prices, include_hmm=False)
    close = prices["Close"]
    t = 400
    assert feats["mom_12_1"].iloc[t] == pytest.approx(close.iloc[t - 21] / close.iloc[t - 252] - 1)


# ---- decision dates ----

def test_month_end_dates_keeps_complete_months_only():
    index = pd.bdate_range("2024-01-01", "2024-03-13")
    assert list(month_end_dates(index)) == [pd.Timestamp("2024-01-31"),
                                            pd.Timestamp("2024-02-29")]
    complete = pd.bdate_range("2024-01-01", "2024-02-29")
    assert month_end_dates(complete)[-1] == pd.Timestamp("2024-02-29")


def test_month_end_dates_uses_last_trading_day_and_drops_ambiguous_final_month():
    # 29 March 2024 was Good Friday; the market's last March session was the 28th.
    index = pd.bdate_range("2024-02-01", "2024-04-30").drop(pd.Timestamp("2024-03-29"))
    assert pd.Timestamp("2024-03-28") in month_end_dates(index)
    # Ending on the 28th it can't be told from a truncated month, so it's dropped.
    ends = month_end_dates(index[index <= "2024-03-28"])
    assert list(ends) == [pd.Timestamp("2024-02-29")]


# ---- labels ----

def test_direction_label_is_sign_of_next_month_and_last_is_nan(prices):
    dates = month_end_dates(prices.index)
    labels = direction_labels(prices, dates)
    close = prices["Close"]
    for t, t_next in zip(dates[:-1], dates[1:], strict=True):
        assert labels[t] == float(close[t_next] > close[t])
    assert np.isnan(labels.iloc[-1])
    assert label_end_times(dates).iloc[:-1].tolist() == list(dates[1:])
    assert forward_returns(prices, dates).iloc[0] == pytest.approx(
        close[dates[1]] / close[dates[0]] - 1)


def test_labels_need_data_only_through_their_end_time(prices):
    dates = month_end_dates(prices.index)
    t, t_end = dates[20], dates[21]
    upto_end = prices.loc[:t_end]
    assert direction_labels(upto_end, month_end_dates(upto_end.index))[t] == \
        direction_labels(prices, dates)[t]
    assert regime_labels(upto_end, month_end_dates(upto_end.index))[t] == \
        regime_labels(prices, dates)[t]


def test_regime_label_is_the_modal_regime_of_the_next_month():
    index = pd.bdate_range("2024-01-01", "2024-04-30")
    dates = month_end_dates(index)  # Jan, Feb, Mar, Apr month-ends
    daily = pd.Series(0, index=index)
    feb = (index > dates[0]) & (index <= dates[1])
    mar = (index > dates[1]) & (index <= dates[2])
    apr = (index > dates[2]) & (index <= dates[3])
    daily[feb] = 3
    daily.iloc[np.flatnonzero(feb)[:3]] = 1              # mostly 3
    mar_pos = np.flatnonzero(mar)
    daily.iloc[mar_pos[: len(mar_pos) // 2]] = 2         # about half 2, half 0 ...
    daily.iloc[mar_pos[len(mar_pos) // 2:]] = 0
    daily.iloc[mar_pos[-1]] = 2                          # ... and 2 wins on count
    daily[apr] = UNKNOWN                                 # all warm-up: no label
    labels = regime_labels(pd.DataFrame({"Close": 1.0}, index=index), dates,
                           daily_regimes=daily)
    assert labels.iloc[0] == 3
    assert labels.iloc[1] == 2
    assert np.isnan(labels.iloc[2])                      # April: only UNKNOWN days
    assert np.isnan(labels.iloc[3])                      # no next month

    even = pd.Series(0, index=index)
    mar_days = np.flatnonzero(mar)[:20]
    even.iloc[mar_days[:10]] = 1
    even.iloc[mar_days[10:]] = 2                         # 10 vs 10 tie, 2 seen later
    even.iloc[np.flatnonzero(mar)[20:]] = UNKNOWN
    tied = regime_labels(pd.DataFrame({"Close": 1.0}, index=index), dates, daily_regimes=even)
    assert tied.iloc[1] == 2


def test_previous_label_is_last_months_label():
    labels = pd.Series([1.0, 0.0, 1.0, np.nan])
    assert previous_labels(labels).tolist()[1:3] == [1.0, 0.0]


# ---- splits ----

def _monthly_times(n: int) -> tuple[pd.Series, pd.Series]:
    ends = pd.date_range("2000-01-31", periods=n + 1, freq="ME")
    t0 = pd.Series(ends[:-1])
    t1 = pd.Series(ends[1:])
    return t0, t1


def test_walk_forward_trains_only_on_labels_known_before_the_test_block():
    t0, t1 = _monthly_times(150)
    folds = walk_forward_splits(t0, t1, min_train=60, test_size=24, embargo=EMBARGO_MONTHS)
    assert [len(f.test) for f in folds] == [24, 24, 24, 17]
    tested = np.concatenate([f.test for f in folds])
    assert (np.diff(tested) == 1).all() and tested[-1] == 149
    for fold in folds:
        assert len(fold.train) >= 60
        assert fold.train.max() < fold.test[0]
        # Every training label ended strictly before the first test prediction.
        assert t1[fold.train].max() < t0[fold.test[0]]
        assert fold.purged.tolist() == [fold.test[0] - 1]
        assert len(fold.embargoed) == 0          # nothing after the test block to embargo
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes) and sizes[0] == 60


def test_purged_kfold_purges_overlaps_and_embargoes_the_following_samples():
    t0, t1 = _monthly_times(100)
    folds = purged_kfold_splits(t0, t1, n_splits=5, embargo=12)
    for fold in folds:
        span_start, span_end = t0[fold.test[0]], t1[fold.test].max()
        assert not ((t0[fold.train] <= span_end) & (t1[fold.train] >= span_start)).any()
        after = np.arange(fold.test[-1] + 2, min(fold.test[-1] + 13, 100))
        assert set(after) <= set(fold.embargoed)
        assert not set(after) & set(fold.train)
        assert set(fold.train) | set(fold.test) | set(fold.purged) | set(fold.embargoed) == \
            set(range(100))


def test_plain_kfold_would_train_on_the_test_period():
    # Unpurged contiguous k-fold: the training set is simply "everything
    # else". It contains labels overlapping the test span, and samples
    # just after the test block whose 12-month look-back covers it.
    t0, t1 = _monthly_times(100)
    test = np.arange(40, 60)
    plain_train = np.setdiff1d(np.arange(100), test)
    span_start, span_end = t0[test[0]], t1[test].max()
    overlapping = (t0[plain_train] <= span_end) & (t1[plain_train] >= span_start)
    looks_back_into_test = (plain_train > test[-1]) & (plain_train <= test[-1] + 12)
    assert overlapping.any() and looks_back_into_test.any()

    fixed = purge_and_embargo(t0, t1, plain_train, test, embargo=12)
    assert not ((t0[fixed.train] <= span_end) & (t1[fixed.train] >= span_start)).any()
    assert not ((fixed.train > test[-1]) & (fixed.train <= test[-1] + 12)).any()


def test_split_validation():
    t0, t1 = _monthly_times(30)
    with pytest.raises(ValueError, match="strictly increasing"):
        purge_and_embargo(t0[::-1].reset_index(drop=True), t1, np.arange(5), np.arange(5, 8))
    with pytest.raises(ValueError, match="contiguous"):
        purge_and_embargo(t0, t1, np.arange(5), np.array([6, 8]))
    with pytest.raises(ValueError, match="at or after"):
        purge_and_embargo(t1, t0, np.arange(5), np.arange(5, 8))
    with pytest.raises(ValueError, match="can't give"):
        walk_forward_splits(t0, t1, min_train=30)


# ---- baselines and evaluation ----

def test_majority_class_uses_only_training_labels():
    t0, t1 = _monthly_times(10)
    y = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    fold = purge_and_embargo(t0, t1, np.arange(5), np.arange(6, 10))
    assert fold.train.tolist() == [0, 1, 2, 3, 4]
    assert predict_majority(y, fold).tolist() == [1.0] * 4
    assert majority_class(pd.Series([0.0, 1.0])) == 0.0   # ties go to the smaller label


def test_previous_label_baseline_reads_the_test_rows():
    previous = pd.Series([np.nan, 1.0, 0.0, 1.0])
    fold = Fold(0, np.array([0]), np.array([2, 3]), np.array([1]), np.array([], dtype=int))
    assert predict_previous(previous, fold).tolist() == [0.0, 1.0]


def test_strategy_returns_charge_costs_on_every_position_change():
    fwd = pd.Series([0.02, -0.01, 0.03])
    hold = strategy_returns(pd.Series([1.0, 1.0, 1.0]), fwd, cost_bps=10)
    assert hold.tolist() == pytest.approx([0.019, -0.01, 0.03])
    flip = strategy_returns(pd.Series([1.0, 0.0, 1.0]), fwd, cost_bps=10)
    assert flip.tolist() == pytest.approx([0.019, -0.001, 0.029])
    assert (strategy_returns(pd.Series([0.0, 0.0, 0.0]), fwd) == 0).all()
    summary = trading_summary(hold, pd.Series([1.0, 1.0, 1.0]))
    assert summary["total_return"] == pytest.approx(1.019 * 0.99 * 1.03 - 1)
    assert summary["exposure"] == 1.0


def test_statistics():
    low, high = wilson_interval(50, 100)
    assert (low, high) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    assert mcnemar_exact(np.array([True, False]), np.array([True, False])) == 1.0
    ten = np.ones(10, dtype=bool)
    assert mcnemar_exact(ten, ~ten) == pytest.approx(2 * 0.5 ** 10)
    assert balanced_accuracy(np.array([1, 1, 1, 0]), np.array([1, 1, 1, 1])) == 0.5


@pytest.mark.parametrize("target", ["direction", "regime"])
def test_dataset_is_point_in_time(prices, target):
    full = build_dataset(prices, target=target, include_hmm=False)
    assert not full.X.isna().any().any() and not full.y.isna().any()
    grid = month_end_dates(prices.index)
    position = grid.get_indexer(full.X.index)
    assert (grid[position + 1] == full.t1.to_numpy()).all()
    cut = build_dataset(prices.loc[:CUTOFFS[1]], target=target, include_hmm=False)
    assert not _mismatches(full.X, cut.X)
    assert cut.previous.equals(full.previous.loc[cut.previous.index])
    assert cut.y.equals(full.y.loc[cut.y.index])


@pytest.mark.parametrize("target", ["direction", "regime"])
def test_evaluate_baselines_scores_each_month_once_out_of_sample(prices, target):
    ds = build_dataset(prices, target=target, include_hmm=False)
    folds = walk_forward_splits(ds.t0, ds.t1, min_train=24, test_size=12)
    report = evaluate_baselines(ds, folds)

    preds = report.predictions
    first_test = folds[0].test[0]
    assert preds.index.equals(ds.y.index[first_test:])
    assert (preds["y"] == ds.y.iloc[first_test:]).all()
    assert report.folds["n_test"].sum() == len(preds)
    assert (report.folds["n_train"] >= 24).all()
    for fold, row in zip(folds, report.folds.itertuples(), strict=True):
        assert preds.loc[preds["fold"] == fold.number, "majority_class"].nunique() == 1
        assert row.acc_previous_label == pytest.approx(
            (preds.loc[preds["fold"] == fold.number, "previous_label"]
             == preds.loc[preds["fold"] == fold.number, "y"]).mean())
    summary = report.summary
    assert (summary["n"] == len(preds)).all()
    assert (summary["ci_low"] <= summary["accuracy"]).all()
    assert (summary["accuracy"] <= summary["ci_high"]).all()
    assert 0 <= report.mcnemar_p <= 1
    if target == "direction":
        assert report.trading is not None
        assert report.trading.loc["buy_and_hold", "exposure"] == 1.0
    else:
        assert report.trading is None and "buy_and_hold" not in preds


def test_majority_baseline_cannot_see_test_labels(prices):
    ds = build_dataset(prices, target="direction", include_hmm=False)
    folds = walk_forward_splits(ds.t0, ds.t1, min_train=24, test_size=12)
    before = evaluate_baselines(ds, folds).predictions["majority_class"]
    flipped_y = ds.y.copy()
    last_test = folds[-1].test
    flipped_y.iloc[last_test] = 1 - flipped_y.iloc[last_test]
    flipped = dataclasses.replace(ds, y=flipped_y)
    after = evaluate_baselines(flipped, folds).predictions["majority_class"]
    # The last fold's test labels feed no training set, so no prediction moves.
    assert before.equals(after)


def test_build_dataset_rejects_unknown_target(prices):
    with pytest.raises(ValueError, match="target"):
        build_dataset(prices, target="volatility", include_hmm=False)
