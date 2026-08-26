"""
Market regime detection.

A "regime" is a persistent market environment -- calm uptrend, choppy
range, high-volatility selloff -- that changes the odds facing every
strategy in strategies.py. Trend-following works in trending regimes and
donates money in choppy ones. Mean reversion is the mirror image. None
of that shows up in a single full-period Sharpe ratio, which averages
the good regimes and the bad ones into one number that describes
neither.

This module labels each day with a regime, using one of five methods
that trade off transparency against flexibility:

    rules      -- explicit volatility/trend thresholds. No fitting, no
                  leakage, no mystery. Start here.
    kmeans     -- clusters days in regime-feature space.
    gmm        -- soft clustering; gives you P(regime) per day.
    hmm        -- Gaussian hidden Markov model. The standard choice,
                  because it's the only one that models PERSISTENCE:
                  regimes are sticky, and a model that knows that
                  produces far fewer one-day head-fakes.
    supervised -- defines regimes from FORWARD returns/vol on the
                  training window, then trains a classifier to recognize
                  them from today's features. Read the warning below it.

THE LOOKAHEAD TRAP, which is the whole reason this module is written the
way it is: fitting a clustering model on the full history and then
backtesting on that history is lookahead bias, full stop. The cluster
centers encode the future. Your 2015 "low volatility regime" label was
computed partly from 2020. The backtest will look wonderful and the
strategy will not work. Two defenses are built in:

    fit_frac < 1.0             fit on the first slice only
    detect_regimes_walk_forward()  refit on an expanding window

Both are demonstrated side by side in the dashboard, because the size of
the gap between the honest and dishonest versions is the lesson.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

from analytics import annualized_volatility, max_drawdown, TRADING_DAYS_PER_YEAR
from regime_features import build_regime_features, reduce_dimensions, standardize_features

REGIME_METHODS = ("rules", "kmeans", "gmm", "hmm", "supervised")

# Label used for rows that don't have enough history to be classified.
# Downstream code treats it as "no opinion" and stays flat.
UNKNOWN = -1

REGIME_METHOD_DOCS = {
    "rules": (
        "Hand-written thresholds on volatility percentile and trend. Nothing is fitted, so "
        "nothing can leak and nothing can overfit. Its weakness is also its strength: it only "
        "sees what you told it to look at. Use it as the baseline every fitted method has to beat."
    ),
    "kmeans": (
        "Groups days into k clusters in regime-feature space. Fast and easy to explain, but it "
        "assumes regimes are round blobs of equal size and it has no idea that regimes persist — "
        "so raw k-means labels flicker day to day. Smoothing matters a lot here."
    ),
    "gmm": (
        "Gaussian mixture: soft clustering that gives you P(regime) per day rather than a hard "
        "label. Handles elongated, unequal clusters that k-means mangles. Still has no notion of "
        "persistence."
    ),
    "hmm": (
        "Gaussian hidden Markov model. Learns both what each regime looks like AND how likely it "
        "is to persist, via a transition matrix. This is the standard tool for regime detection "
        "in practice. Decoding with 'filter' uses only past data (honest); 'viterbi' uses the "
        "whole sequence (cleaner-looking labels, but not available in real time)."
    ),
    "supervised": (
        "Defines regimes from FUTURE 21-day return and volatility inside the training window, then "
        "trains a classifier to recognize them from today's features. Legitimate — at inference it "
        "only sees the past — but it is one careless line away from leakage, which is exactly why "
        "it's here. Always run it with fit_frac < 1.0."
    ),
}

SMOOTHING_DOCS = {
    "none": "Raw model output. Expect the labels to flicker — one-day regime changes that no trader would act on.",
    "min_duration": "A new regime must repeat for N consecutive days before it's accepted. Causal, and it costs you N days of lag on every real transition. That lag is the honest price of not being whipsawed.",
    "ema_prob": "Exponentially smooth the model's probabilities, then take the argmax. Causal, softer than min_duration, needs a model that outputs probabilities (gmm/hmm/supervised).",
    "median": "Rolling majority vote over a trailing window. Causal. Simple and effective, but blunt.",
}


# --------------------------------------------------------------------------
# A small Gaussian HMM
# --------------------------------------------------------------------------
class GaussianHMM:
    """
    Diagonal-covariance Gaussian HMM fitted with Baum-Welch (EM).

    Written out longhand rather than pulled from hmmlearn for two
    reasons: it keeps the dependency list to what's already in
    requirements.txt, and the forward pass is the part interns most need
    to actually see. `filter()` uses information up to time t only --
    that is what you could have known live. `smooth()` and `viterbi()`
    condition on the whole sequence and are therefore historical
    reconstructions, not tradeable signals.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100, tol: float = 1e-6,
                 var_floor: float = 1e-3, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.var_floor = var_floor
        self.random_state = random_state
        self.converged_ = False
        self.n_iter_ = 0
        self.loglikelihood_ = -np.inf

    # -- emissions -------------------------------------------------------
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """log P(x_t | state k) for every (t, k), diagonal covariance."""
        var = self.variances_
        diff = X[:, None, :] - self.means_[None, :, :]
        return -0.5 * (np.log(2 * np.pi * var).sum(axis=1)[None, :] + ((diff ** 2) / var).sum(axis=2))

    @staticmethod
    def _scaled_emission(log_b: np.ndarray):
        """
        exp() of the log-emissions with the per-row max divided out.
        Prevents underflow; the offset cancels everywhere except the
        log-likelihood, where we add it back.
        """
        offset = log_b.max(axis=1, keepdims=True)
        return np.exp(log_b - offset), offset.ravel()

    # -- inference -------------------------------------------------------
    def _forward(self, b: np.ndarray):
        """Scaled forward pass. Returns (filtered probabilities, scaling factors)."""
        n_obs = b.shape[0]
        alpha = np.zeros((n_obs, self.n_states))
        scale = np.zeros(n_obs)

        a = self.startprob_ * b[0]
        scale[0] = a.sum() or 1e-300
        alpha[0] = a / scale[0]

        for t in range(1, n_obs):
            a = (alpha[t - 1] @ self.transmat_) * b[t]
            scale[t] = a.sum() or 1e-300
            alpha[t] = a / scale[t]

        return alpha, scale

    def _backward(self, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
        n_obs = b.shape[0]
        beta = np.zeros((n_obs, self.n_states))
        beta[-1] = 1.0
        for t in range(n_obs - 2, -1, -1):
            beta[t] = (self.transmat_ @ (b[t + 1] * beta[t + 1])) / scale[t + 1]
        return beta

    # -- fitting ---------------------------------------------------------
    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n_obs, n_features = X.shape

        # Initialize from a GMM: EM is only as good as where it starts,
        # and random starts on financial data routinely land in a local
        # optimum where every state looks the same.
        init = GaussianMixture(
            n_components=self.n_states, covariance_type="diag",
            random_state=self.random_state, reg_covar=1e-4, n_init=3,
        ).fit(X)

        self.means_ = init.means_.copy()
        self.variances_ = np.maximum(init.covariances_.copy(), self.var_floor)
        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)

        # Start strongly diagonal: assume regimes persist, then let the
        # data argue otherwise. Starting uniform tends to converge to a
        # memoryless model that's just a slower GMM.
        stay = 0.95
        self.transmat_ = np.full((self.n_states, self.n_states),
                                 (1 - stay) / max(self.n_states - 1, 1))
        np.fill_diagonal(self.transmat_, stay)

        prev_loglik = -np.inf
        for iteration in range(self.n_iter):
            log_b = self._log_emission(X)
            b, offset = self._scaled_emission(log_b)

            alpha, scale = self._forward(b)
            beta = self._backward(b, scale)

            loglik = np.log(scale).sum() + offset.sum()
            if not np.isfinite(loglik):
                break

            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma = gamma / np.where(gamma_sum > 0, gamma_sum, 1.0)

            # xi[t, i, j] = P(state_t = i, state_{t+1} = j | X)
            xi_num = (
                alpha[:-1, :, None]
                * self.transmat_[None, :, :]
                * (b[1:] * beta[1:])[:, None, :]
                / scale[1:, None, None]
            )
            xi_sum = xi_num.sum(axis=0)

            # M-step, with guards: a state that has collapsed to zero
            # responsibility keeps its old parameters rather than
            # producing NaNs that silently propagate.
            self.startprob_ = gamma[0] / gamma[0].sum()

            row_sums = xi_sum.sum(axis=1, keepdims=True)
            new_transmat = np.divide(xi_sum, row_sums, out=self.transmat_.copy(), where=row_sums > 0)
            self.transmat_ = new_transmat

            weights = gamma.sum(axis=0)
            for k in range(self.n_states):
                if weights[k] <= 1e-8:
                    continue
                mu = (gamma[:, k, None] * X).sum(axis=0) / weights[k]
                var = (gamma[:, k, None] * (X - mu) ** 2).sum(axis=0) / weights[k]
                self.means_[k] = mu
                self.variances_[k] = np.maximum(var, self.var_floor)

            self.n_iter_ = iteration + 1
            self.loglikelihood_ = loglik
            if abs(loglik - prev_loglik) < self.tol * max(abs(prev_loglik), 1.0):
                self.converged_ = True
                break
            prev_loglik = loglik

        return self

    # -- decoding --------------------------------------------------------
    def filter(self, X: np.ndarray) -> np.ndarray:
        """P(state_t | x_1..x_t) -- causal. This is the live-tradeable one."""
        b, _ = self._scaled_emission(self._log_emission(np.asarray(X, dtype=float)))
        alpha, _ = self._forward(b)
        return alpha

    def smooth(self, X: np.ndarray) -> np.ndarray:
        """P(state_t | all data) -- NOT causal. Hindsight reconstruction."""
        b, _ = self._scaled_emission(self._log_emission(np.asarray(X, dtype=float)))
        alpha, scale = self._forward(b)
        gamma = alpha * self._backward(b, scale)
        return gamma / gamma.sum(axis=1, keepdims=True)

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Most likely single state PATH given all data -- also not causal."""
        log_b = self._log_emission(np.asarray(X, dtype=float))
        n_obs = log_b.shape[0]
        with np.errstate(divide="ignore"):
            log_start = np.log(self.startprob_)
            log_trans = np.log(self.transmat_)

        delta = np.zeros((n_obs, self.n_states))
        psi = np.zeros((n_obs, self.n_states), dtype=int)
        delta[0] = log_start + log_b[0]
        for t in range(1, n_obs):
            scores = delta[t - 1][:, None] + log_trans
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores.max(axis=0) + log_b[t]

        path = np.zeros(n_obs, dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(n_obs - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]
        return path


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class RegimeResult:
    """Everything a downstream consumer needs to reason about the labels."""

    labels: pd.Series            # int per day; UNKNOWN (-1) during warm-up
    names: dict                  # {regime_id: human-readable name}
    features: pd.DataFrame       # raw (unstandardized) regime features
    method: str
    n_regimes: int
    causal: bool                 # False if the fit saw data it's labelling
    probabilities: pd.DataFrame = None
    model: object = None
    fit_end: object = None       # last date the model was allowed to see
    meta: dict = field(default_factory=dict)

    def named_labels(self) -> pd.Series:
        """The label series with names substituted, for plotting and tables."""
        mapping = dict(self.names)
        mapping[UNKNOWN] = "Warm-up"
        return self.labels.map(mapping)

    def valid(self) -> pd.Series:
        """Boolean mask of days that actually got a regime."""
        return self.labels != UNKNOWN

    def transition_matrix(self) -> pd.DataFrame:
        return transition_matrix(self.labels, self.names)

    def episodes(self) -> pd.DataFrame:
        return regime_episodes(self.labels, self.names)

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        return regime_summary(self, df)


# --------------------------------------------------------------------------
# Labelling helpers
# --------------------------------------------------------------------------
def _quadrant_name(trend: float, vol_pct: float) -> str:
    """Turns a cluster centroid into something a human can argue with."""
    if vol_pct >= 0.66:
        vol_band = "high"
    elif vol_pct >= 0.33:
        vol_band = "mid"
    else:
        vol_band = "low"

    if trend > 0.02:
        trend_band = "up"
    elif trend < -0.02:
        trend_band = "down"
    else:
        trend_band = "flat"

    return {
        ("low", "up"): "Calm Uptrend",
        ("low", "flat"): "Quiet Range",
        ("low", "down"): "Slow Bleed",
        ("mid", "up"): "Steady Uptrend",
        ("mid", "flat"): "Choppy",
        ("mid", "down"): "Grinding Down",
        ("high", "up"): "Volatile Rally",
        ("high", "flat"): "Turbulent",
        ("high", "down"): "Crisis / Selloff",
    }[(vol_band, trend_band)]


def _order_and_name(labels: pd.Series, features: pd.DataFrame) -> tuple:
    """
    Cluster IDs come out of every fitting algorithm in arbitrary order,
    and they change between refits. That would make "regime 2" mean
    something different in 2018 than in 2022, which quietly destroys any
    regime-conditioned strategy.

    Fix: always renumber so regime 0 is the calmest and the highest ID is
    the most violent, then name each one from its centroid. Now IDs are
    stable and comparable across methods, refits, and tickers.

    Returns (renumbered_labels, names, remap) -- the remap is needed to
    reorder any probability columns to match.
    """
    valid = labels != UNKNOWN
    if not valid.any():
        return labels, {}, {}

    present = sorted(labels[valid].unique())
    # Ordered on absolute realized volatility rather than its expanding
    # percentile: the percentile is a rank against history, so an early
    # moderately-choppy stretch can out-rank a later genuinely violent one
    # and "regime 0 is the calmest" would stop being true in level terms.
    vol_by_label = {
        k: features.loc[valid & (labels == k), "vol_20d"].mean()
        for k in present
    }
    # NaN centroids (a cluster with no complete feature rows) sort last.
    ordering = sorted(present, key=lambda k: (np.isnan(vol_by_label[k]), vol_by_label[k]))
    remap = {old: new for new, old in enumerate(ordering)}

    new_labels = labels.map(lambda v: remap.get(v, UNKNOWN)).astype(int)

    bases, vols = {}, {}
    for new_id in range(len(ordering)):
        rows = valid & (new_labels == new_id)
        trend = features.loc[rows, "trend_60d"].mean()
        vol_pct = features.loc[rows, "vol_percentile"].mean()
        bases[new_id] = _quadrant_name(
            0.0 if pd.isna(trend) else trend,
            0.5 if pd.isna(vol_pct) else vol_pct,
        )
        vols[new_id] = features.loc[rows, "vol_20d"].mean()

    # Two clusters can land in the same quadrant -- typically two flavours
    # of selloff that differ mainly in severity. Disambiguating with their
    # actual volatility is more useful than tacking on "(2)".
    counts = pd.Series(list(bases.values())).value_counts()
    names = {}
    for new_id, base in bases.items():
        if counts.get(base, 0) > 1 and not pd.isna(vols[new_id]):
            names[new_id] = f"{base} ({vols[new_id]:.0%} vol)"
        else:
            names[new_id] = base

    return new_labels, names, remap


def smooth_labels(labels: pd.Series, method: str = "min_duration", min_duration: int = 5,
                  probabilities: pd.DataFrame = None, ema_span: int = 5,
                  window: int = 5) -> pd.Series:
    """
    Raw model labels flicker. A regime that lasts one day is not a regime,
    it's noise, and trading every flicker converts a decent signal into a
    transaction-cost machine.

    Every method here is CAUSAL -- it uses past labels only. A centered
    rolling filter would look far tidier on the chart and would be
    lookahead bias.
    """
    if method == "none":
        return labels

    valid = labels != UNKNOWN
    if not valid.any():
        return labels

    if method == "ema_prob":
        if probabilities is None:
            raise ValueError("smooth='ema_prob' requires a model that outputs probabilities (gmm/hmm/supervised).")
        smoothed_probs = probabilities.ewm(span=ema_span, adjust=False).mean()
        out = labels.copy()
        out.loc[valid] = smoothed_probs.loc[valid].to_numpy().argmax(axis=1)
        return out.astype(int)

    if method == "median":
        out = labels.copy()
        rolled = (
            labels[valid]
            .rolling(window, min_periods=1)
            .apply(lambda w: pd.Series(w).mode().iat[0], raw=False)
        )
        out.loc[valid] = rolled.astype(int)
        return out.astype(int)

    if method != "min_duration":
        raise ValueError(f"Unknown smoothing method: {method!r}")

    # Confirmation filter: a candidate regime has to show up for
    # `min_duration` consecutive days before we accept the switch.
    values = labels[valid].to_numpy()
    out_values = np.empty_like(values)
    current = values[0]
    pending, streak = None, 0
    for i, raw in enumerate(values):
        if raw == current:
            pending, streak = None, 0
        else:
            if raw == pending:
                streak += 1
            else:
                pending, streak = raw, 1
            if streak >= min_duration:
                current, pending, streak = raw, None, 0
        out_values[i] = current

    out = labels.copy()
    out.loc[valid] = out_values
    return out.astype(int)


# --------------------------------------------------------------------------
# The individual detection methods
# --------------------------------------------------------------------------
def _rules_labels(features: pd.DataFrame, valid_index: pd.Index) -> tuple:
    """
    Fully transparent baseline: two thresholds, four quadrants, zero
    fitted parameters. Anything a fitted model does has to beat this, and
    interns are often surprised by how rarely it does.
    """
    sub = features.loc[valid_index]
    high_vol = sub["vol_percentile"] >= 0.70
    uptrend = sub["trend_60d"] > 0

    ids = pd.Series(UNKNOWN, index=features.index, dtype=int)
    ids.loc[valid_index[(~high_vol & uptrend).to_numpy()]] = 0
    ids.loc[valid_index[(~high_vol & ~uptrend).to_numpy()]] = 1
    ids.loc[valid_index[(high_vol & uptrend).to_numpy()]] = 2
    ids.loc[valid_index[(high_vol & ~uptrend).to_numpy()]] = 3

    names = {0: "Calm Uptrend", 1: "Quiet Range", 2: "Volatile Rally", 3: "Crisis / Selloff"}
    return ids, names, None, None


def _supervised_labels(df: pd.DataFrame, X: pd.DataFrame, fit_index: pd.Index,
                       horizon: int, random_state: int) -> tuple:
    """
    Regimes defined by what actually happened over the NEXT `horizon`
    days, then learned as a mapping from today's features.

    The forward look is confined to constructing training targets inside
    the training window -- the same arrangement features.build_labels()
    uses for the direction model, and legitimate for the same reason. It
    stops being legitimate the moment fit_frac=1.0, which is why the
    dashboard defaults this method to 0.6.
    """
    close = df["Close"]
    returns = close.pct_change()
    forward_return = close.shift(-horizon) / close - 1
    forward_vol = returns.rolling(horizon).std().shift(-horizon) * np.sqrt(TRADING_DAYS_PER_YEAR)

    train_vol = forward_vol.loc[fit_index].dropna()
    if train_vol.empty:
        raise ValueError("Not enough data to build supervised regime targets.")
    vol_cut = train_vol.median()

    target = pd.Series(UNKNOWN, index=df.index, dtype=int)
    calm, up = forward_vol <= vol_cut, forward_return > 0
    target[calm & up] = 0
    target[calm & ~up] = 1
    target[~calm & up] = 2
    target[~calm & ~up] = 3
    target[forward_vol.isna() | forward_return.isna()] = UNKNOWN

    train_rows = fit_index[(target.loc[fit_index] != UNKNOWN).to_numpy()]
    if len(train_rows) < 50:
        raise ValueError("Fewer than 50 usable training rows for the supervised regime model.")

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=random_state
    )
    model.fit(X.loc[train_rows], target.loc[train_rows])

    proba = pd.DataFrame(
        model.predict_proba(X), index=X.index,
        columns=[int(c) for c in model.classes_],
    ).reindex(columns=[0, 1, 2, 3], fill_value=0.0)

    ids = pd.Series(UNKNOWN, index=X.index, dtype=int)
    ids.loc[X.index] = proba.to_numpy().argmax(axis=1)
    return ids, None, proba, model


def _fit_predict(method: str, X: pd.DataFrame, fit_index: pd.Index, n_regimes: int,
                 decode: str, random_state: int) -> tuple:
    """Fits the chosen unsupervised model on `fit_index`, labels all of X."""
    fit_X = X.loc[fit_index]

    if method == "kmeans":
        model = KMeans(n_clusters=n_regimes, n_init=10, random_state=random_state).fit(fit_X)
        raw = model.predict(X)
        # k-means has no probabilities; distance-based soft scores keep
        # the ema_prob smoother usable rather than erroring out.
        distances = model.transform(X)
        weights = 1.0 / (distances + 1e-9)
        proba = pd.DataFrame(weights / weights.sum(axis=1, keepdims=True), index=X.index)

    elif method == "gmm":
        model = GaussianMixture(
            n_components=n_regimes, covariance_type="full",
            random_state=random_state, reg_covar=1e-4, n_init=3,
        ).fit(fit_X)
        proba = pd.DataFrame(model.predict_proba(X), index=X.index)
        raw = proba.to_numpy().argmax(axis=1)

    elif method == "hmm":
        model = GaussianHMM(n_states=n_regimes, random_state=random_state).fit(fit_X.to_numpy())
        if decode == "viterbi":
            raw = model.viterbi(X.to_numpy())
            proba = pd.DataFrame(model.smooth(X.to_numpy()), index=X.index)
        elif decode == "smooth":
            proba = pd.DataFrame(model.smooth(X.to_numpy()), index=X.index)
            raw = proba.to_numpy().argmax(axis=1)
        else:  # "filter" -- the only causal option, and the default
            proba = pd.DataFrame(model.filter(X.to_numpy()), index=X.index)
            raw = proba.to_numpy().argmax(axis=1)

    else:
        raise ValueError(f"Unknown regime method: {method!r}. Choose from {REGIME_METHODS}.")

    return pd.Series(raw, index=X.index, dtype=int), proba, model


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def detect_regimes(
    df: pd.DataFrame,
    method: str = "hmm",
    n_regimes: int = 3,
    fit_frac: float = 1.0,
    standardize: str = "expanding",
    use_pca: bool = False,
    n_components: int = 3,
    smooth: str = "min_duration",
    min_duration: int = 5,
    ema_span: int = 5,
    decode: str = "filter",
    horizon: int = 21,
    random_state: int = 42,
    features: pd.DataFrame = None,
) -> RegimeResult:
    """
    Labels every day in `df` with a market regime.

    fit_frac controls the honesty of the result. At 1.0 the model is fit
    on all the data it then labels, so the labels embed knowledge of the
    future -- fine for describing history, NOT fine for feeding a
    backtest. Anything below 1.0 fits on the leading slice only, and
    `causal` on the result records which you asked for.

    `rules` is always causal because it fits nothing.
    """
    if method not in REGIME_METHODS:
        raise ValueError(f"Unknown regime method: {method!r}. Choose from {REGIME_METHODS}.")

    raw_features = build_regime_features(df) if features is None else features
    scaled = standardize_features(raw_features, method=standardize)

    if use_pca:
        scaled, _pca = reduce_dimensions(scaled.dropna(), n_components=n_components)
        scaled = scaled.reindex(raw_features.index)

    X = scaled.dropna()
    if len(X) < 50:
        raise ValueError(
            f"Only {len(X)} rows survive the regime-feature warm-up (need 50+). "
            "Fetch a longer history — the long-window features need about a year to spin up."
        )

    split_idx = max(int(len(X) * fit_frac), 30)
    fit_index = X.index[:split_idx]
    fit_end = fit_index[-1]

    if method == "rules":
        ids, names, proba, model = _rules_labels(raw_features, X.index)
        n_regimes, causal, fit_end = 4, True, None
    elif method == "supervised":
        ids, names, proba, model = _supervised_labels(df, X, fit_index, horizon, random_state)
        ids = ids.reindex(raw_features.index, fill_value=UNKNOWN)
        proba = proba.reindex(raw_features.index)
        n_regimes, causal = 4, fit_frac < 1.0
    else:
        ids, proba, model = _fit_predict(method, X, fit_index, n_regimes, decode, random_state)
        ids = ids.reindex(raw_features.index, fill_value=UNKNOWN)
        proba = proba.reindex(raw_features.index)
        names = None
        causal = fit_frac < 1.0 and decode == "filter"

    if names is None:
        ids, names, remap = _order_and_name(ids, raw_features)
        if proba is not None:
            # Columns must follow the labels through the renumbering,
            # otherwise ema_prob smoothing would read column 2 as regime 2
            # when the model meant something else entirely.
            keep = [old for old in remap if old in proba.columns]
            proba = proba[keep]
            proba.columns = [remap[old] for old in keep]
            proba = proba.reindex(columns=sorted(proba.columns))

    smoothed = smooth_labels(
        ids, method=smooth, min_duration=min_duration,
        probabilities=proba, ema_span=ema_span, window=min_duration,
    )

    return RegimeResult(
        labels=smoothed.reindex(df.index, fill_value=UNKNOWN).astype(int),
        names=names,
        features=raw_features,
        method=method,
        n_regimes=n_regimes,
        causal=causal,
        probabilities=None if proba is None else proba.reindex(df.index),
        model=model,
        fit_end=fit_end,
        meta={
            "fit_frac": fit_frac, "smooth": smooth, "min_duration": min_duration,
            "decode": decode, "standardize": standardize, "use_pca": use_pca,
            "raw_labels": ids.reindex(df.index, fill_value=UNKNOWN).astype(int),
        },
    )


def detect_regimes_walk_forward(
    df: pd.DataFrame,
    method: str = "hmm",
    n_regimes: int = 3,
    initial_train: int = 504,
    refit_every: int = 63,
    standardize: str = "expanding",
    smooth: str = "min_duration",
    min_duration: int = 5,
    random_state: int = 42,
    features: pd.DataFrame = None,
) -> RegimeResult:
    """
    The honest way to label a history you intend to trade on.

    Fit on everything up to date D, label the next `refit_every` days,
    roll forward, repeat. No label is ever produced by a model that saw
    the day it's labelling. The first `initial_train` rows get UNKNOWN --
    you genuinely did not have a regime model then, and pretending
    otherwise is the whole bias we're avoiding.

    Compare this against detect_regimes(fit_frac=1.0) on the same data.
    The full-sample version will look sharper and cleaner. That
    difference is the lookahead bias, drawn to scale.
    """
    if method in ("rules", "supervised"):
        # `rules` fits nothing, so refitting is a no-op; `supervised` has
        # its own train/predict split built in via fit_frac.
        return detect_regimes(
            df, method=method, n_regimes=n_regimes, fit_frac=0.6,
            standardize=standardize, smooth=smooth, min_duration=min_duration,
            random_state=random_state, features=features,
        )

    raw_features = build_regime_features(df) if features is None else features
    scaled = standardize_features(raw_features, method=standardize)
    X = scaled.dropna()

    if len(X) <= initial_train + refit_every:
        raise ValueError(
            f"Walk-forward regime detection needs more than {initial_train + refit_every} clean rows; "
            f"got {len(X)}. Either widen the date range or lower initial_train."
        )

    ids = pd.Series(UNKNOWN, index=raw_features.index, dtype=int)
    proba_rows, refit_dates, model = {}, [], None

    start = initial_train
    while start < len(X):
        stop = min(start + refit_every, len(X))
        train_index, test_index = X.index[:start], X.index[start:stop]
        refit_dates.append(train_index[-1])

        chunk_ids, chunk_proba, model = _fit_predict(
            method, X.loc[train_index.union(test_index)], train_index,
            n_regimes, "filter", random_state,
        )

        # Renumber against the TRAINING rows only, so regime 0 keeps
        # meaning "calmest" across every refit without consulting the
        # out-of-sample days we're about to label.
        vol_by_label = {
            k: raw_features.loc[train_index[(chunk_ids.loc[train_index] == k).to_numpy()], "vol_20d"].mean()
            for k in sorted(chunk_ids.loc[train_index].unique())
        }
        ordering = sorted(vol_by_label, key=lambda k: (np.isnan(vol_by_label[k]), vol_by_label[k]))
        remap = {old: new for new, old in enumerate(ordering)}

        ids.loc[test_index] = [remap.get(v, UNKNOWN) for v in chunk_ids.loc[test_index]]
        if chunk_proba is not None:
            reordered = chunk_proba.loc[test_index, ordering]
            reordered.columns = range(len(ordering))
            proba_rows[start] = reordered
        start = stop

    probabilities = pd.concat(proba_rows.values()).reindex(df.index) if proba_rows else None
    labelled = ids[ids != UNKNOWN]
    _, names, _remap = _order_and_name(ids, raw_features)

    smoothed = smooth_labels(
        ids, method=smooth, min_duration=min_duration,
        probabilities=probabilities, window=min_duration,
    )

    return RegimeResult(
        labels=smoothed.reindex(df.index, fill_value=UNKNOWN).astype(int),
        names=names,
        features=raw_features,
        method=method,
        n_regimes=n_regimes,
        causal=True,
        probabilities=probabilities,
        model=model,
        fit_end=None,
        meta={
            "walk_forward": True, "initial_train": initial_train,
            "refit_every": refit_every, "n_refits": len(refit_dates),
            "refit_dates": refit_dates, "first_label": None if labelled.empty else labelled.index[0],
        },
    )


# --------------------------------------------------------------------------
# Transition and stability analysis
# --------------------------------------------------------------------------
def transition_matrix(labels: pd.Series, names: dict = None, normalize: bool = True) -> pd.DataFrame:
    """
    P(tomorrow's regime | today's regime).

    Read the diagonal first: those are the persistence probabilities, and
    they should be high (0.9+ on daily data). A diagonal near 1/k means
    the model isn't finding regimes, it's finding noise, and no amount of
    downstream cleverness will fix that. 1/(1 - p_ii) is the expected
    duration of regime i in days.
    """
    valid = labels[labels != UNKNOWN]
    pairs = pd.DataFrame({"from": valid, "to": valid.shift(-1)}).dropna()
    if pairs.empty:
        return pd.DataFrame()

    counts = pd.crosstab(pairs["from"], pairs["to"].astype(int))
    ids = sorted(set(counts.index) | set(counts.columns))
    counts = counts.reindex(index=ids, columns=ids, fill_value=0)

    matrix = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) if normalize else counts
    if names:
        label_names = [names.get(i, str(i)) for i in ids]
        matrix.index, matrix.columns = label_names, label_names
    return matrix


def regime_episodes(labels: pd.Series, names: dict = None) -> pd.DataFrame:
    """
    Every contiguous run of a single regime, with start, end and length.

    Useful sanity check: if your "regimes" average four days, you have
    not detected regimes. Real ones last weeks to months.
    """
    valid = labels[labels != UNKNOWN]
    if valid.empty:
        return pd.DataFrame(columns=["regime", "name", "start", "end", "days"])

    block = (valid != valid.shift()).cumsum()
    rows = []
    for _, group in valid.groupby(block):
        regime_id = int(group.iloc[0])
        rows.append({
            "regime": regime_id,
            "name": (names or {}).get(regime_id, str(regime_id)),
            "start": group.index[0],
            "end": group.index[-1],
            "days": len(group),
        })
    return pd.DataFrame(rows)


def regime_stability(labels: pd.Series) -> dict:
    """Headline numbers on how twitchy the labelling is."""
    episodes = regime_episodes(labels)
    valid_days = int((labels != UNKNOWN).sum())
    if episodes.empty or valid_days == 0:
        return {"n_episodes": 0, "avg_duration": 0.0, "median_duration": 0.0,
                "switches_per_year": 0.0, "shortest": 0, "labelled_days": valid_days}

    return {
        "n_episodes": len(episodes),
        "avg_duration": float(episodes["days"].mean()),
        "median_duration": float(episodes["days"].median()),
        "switches_per_year": (len(episodes) - 1) / (valid_days / TRADING_DAYS_PER_YEAR),
        "shortest": int(episodes["days"].min()),
        "labelled_days": valid_days,
    }


def regime_summary(result: RegimeResult, df: pd.DataFrame) -> pd.DataFrame:
    """
    What the ASSET did inside each regime (not what a strategy did --
    that's analytics.performance_by_regime).

    This is the table to read before anything else. If your detected
    regimes don't differ in return or volatility, they aren't regimes,
    and conditioning a strategy on them cannot help.
    """
    returns = df["Close"].pct_change()
    episodes = regime_episodes(result.labels, result.names)
    total_valid = int(result.valid().sum())
    rows = []

    for regime_id in sorted(r for r in result.labels.unique() if r != UNKNOWN):
        mask = result.labels == regime_id
        regime_returns = returns[mask].dropna()
        if regime_returns.empty:
            continue
        equity = (1 + regime_returns).cumprod()
        durations = episodes.loc[episodes["regime"] == regime_id, "days"]

        rows.append({
            "regime": regime_id,
            "name": result.names.get(regime_id, str(regime_id)),
            "days": int(mask.sum()),
            "share": mask.sum() / total_valid if total_valid else 0.0,
            "episodes": int(len(durations)),
            "avg_duration": float(durations.mean()) if len(durations) else 0.0,
            "ann_return": (1 + regime_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1,
            "ann_volatility": annualized_volatility(regime_returns),
            "max_drawdown": max_drawdown(equity),
            "pct_up_days": (regime_returns > 0).mean(),
        })

    return pd.DataFrame(rows)
