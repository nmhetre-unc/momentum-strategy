"""
Strategy versions are published like migrations: once a version is
recorded here, its pin never changes. The pin is a fingerprint of the
positions the strategy produces with its default params on a fixed
synthetic price series.

If test_current_versions_reproduce_their_pins fails, the strategy's
behaviour changed. Either that's a bug, or it's intended and the
strategy needs a new version: bump qbt.strategies.STRATEGY_VERSIONS and
ADD a pin for the new version below. Never edit an existing pin. The
service catalog then registers the new version as a new row, so stored
backtests keep the version they ran (see qbt_service.catalog).

A dependency upgrade can change a fingerprint too (ml_direction's comes
from a scikit-learn fit). Treat that the same way: if the positions
changed, it's a new version.

Pins of retired versions stay as the record of what was published; only
the current version can be checked against the code.
"""

import hashlib

import numpy as np
import pandas as pd

from qbt import strategies as strategy_module

PUBLISHED_PINS: dict[tuple[str, int], str] = {
    ("sma_crossover", 1): "08f86d82567a06f9",
    ("momentum", 1): "0f81b8fdef7d06b5",
    ("mean_reversion", 1): "ad2ecccad9294b3c",
    ("ml_direction", 1): "e6ed23fae6f59052",   # pre-fix: predicted its own training period
    ("ml_direction", 2): "acc8214cbb0e9bd8",   # out-of-sample only; see STRATEGY_VERSIONS
}


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(20261115)
    n = 1500
    index = pd.bdate_range("2012-01-02", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n))
    volume = rng.integers(50_000_000, 150_000_000, n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=index)


def fingerprint(name: str) -> str:
    df = _prices()
    signal = strategy_module.STRATEGIES[name](df)
    values = signal.reindex(df.index).astype(float).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def test_every_served_strategy_has_a_version():
    assert set(strategy_module.STRATEGY_VERSIONS) == set(strategy_module.STRATEGIES)


def test_versions_are_published_contiguously_from_1():
    for name, current in strategy_module.STRATEGY_VERSIONS.items():
        published = sorted(v for n, v in PUBLISHED_PINS if n == name)
        assert published == list(range(1, current + 1)), (
            f"{name}: published versions {published}, current {current}")


def test_each_new_version_actually_changed_behaviour():
    for (name, version), pin in PUBLISHED_PINS.items():
        if version > 1:
            assert pin != PUBLISHED_PINS[(name, version - 1)], (
                f"{name} {version} has the same pin as {version - 1}: not a behaviour change")


def test_current_versions_reproduce_their_pins():
    for name, version in strategy_module.STRATEGY_VERSIONS.items():
        assert fingerprint(name) == PUBLISHED_PINS[(name, version)], (
            f"{name} no longer reproduces version {version}: bump its version and add a pin")
