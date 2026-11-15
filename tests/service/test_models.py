"""
The request models generated from PARAM_SPECS: every spec's bounds and
default are enforced exactly, the generated params are exactly what the
strategy functions accept, the cross-field rules hold, the request picks
its model by `strategy`, and the canonical form (what idempotency keys
are checked against) is the same for every way of writing one request.
"""

import inspect
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from qbt.runner import run
from qbt.strategies import PARAM_SPECS, STRATEGIES, default_params
from qbt_service.models import (
    PARAMS_MODELS,
    REQUEST_MODELS,
    canonical_json,
    parse_request,
    request_hash,
)

SPECS = [(name, spec) for name, specs in sorted(PARAM_SPECS.items()) for spec in specs]
SPEC_IDS = [f"{name}.{spec['name']}" for name, spec in SPECS]

BASE = {"ticker": "SPY", "start": "2010-01-01", "end": "2020-01-01"}


def errors(exc: ValidationError) -> list[tuple[tuple, str]]:
    return [(e["loc"], e["type"]) for e in exc.errors()]


# ---- the generated models match the registry and the strategies ----

def test_every_registered_strategy_is_served():
    assert set(PARAMS_MODELS) == set(STRATEGIES) == set(REQUEST_MODELS)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_fields_are_exactly_the_specs(name):
    assert list(PARAMS_MODELS[name].model_fields) == [s["name"] for s in PARAM_SPECS[name]]


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_defaults_are_the_spec_defaults(name):
    assert PARAMS_MODELS[name]().model_dump() == default_params(name)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_field_is_a_strategy_argument(name):
    accepted = set(inspect.signature(STRATEGIES[name]).parameters)
    assert set(PARAMS_MODELS[name].model_fields) <= accepted


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_validated_params_run(name):
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2015-01-01", periods=400)
    df = pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))}, index=index)
    params = parse_request({**BASE, "strategy": name}).params.model_dump()
    assert "sharpe_ratio" in run(df, name, params).metrics


# ---- bounds, types and extra fields ----

@pytest.mark.parametrize(("name", "spec"), SPECS, ids=SPEC_IDS)
def test_bounds_are_inclusive_and_enforced(name, spec):
    model = PARAMS_MODELS[name]
    kind = type(model().model_dump()[spec["name"]])
    # For sma_crossover, a short window of 5 and a long one of 300 keep
    # short < long whichever of the two is being pushed to a bound.
    others = {"short_window": 5, "long_window": 300} if name == "sma_crossover" else {}
    for value in (spec["min"], spec["max"]):
        model(**{**others, spec["name"]: kind(value)})
    step = 1 if kind is int else 1e-9
    with pytest.raises(ValidationError, match="greater than or equal"):
        model(**{**others, spec["name"]: kind(spec["min"] - step)})
    with pytest.raises(ValidationError, match="less than or equal"):
        model(**{**others, spec["name"]: kind(spec["max"] + step)})


def test_schema_carries_the_spec():
    schema = PARAMS_MODELS["momentum"].model_json_schema()
    lookback = schema["properties"]["lookback"]
    assert (lookback["type"], lookback["minimum"], lookback["maximum"], lookback["default"]) == (
        "integer", 5, 60, 20)
    threshold = schema["properties"]["threshold"]
    assert (threshold["type"], threshold["minimum"], threshold["maximum"]) == (
        "number", -0.05, 0.05)
    assert lookback["title"] == "Lookback (days)"
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("value", [20.0, "20", True, None])
def test_int_params_are_strict(value):
    with pytest.raises(ValidationError):
        PARAMS_MODELS["momentum"](lookback=value)


@pytest.mark.parametrize("value", ["0.01", True, math.nan, math.inf, -math.inf])
def test_float_params_are_strict_and_finite(value):
    with pytest.raises(ValidationError):
        PARAMS_MODELS["momentum"](threshold=value)


def test_int_is_accepted_for_a_float_param_and_stored_as_float():
    value = PARAMS_MODELS["momentum"](threshold=0).threshold
    assert value == 0.0 and type(value) is float


def test_unknown_params_are_rejected():
    with pytest.raises(ValidationError) as exc:
        parse_request({**BASE, "strategy": "momentum", "params": {"window": 5}})
    assert errors(exc.value) == [(("momentum", "params", "window"), "extra_forbidden")]


# ---- cross-field rules ----

@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("sma_crossover", {"short_window": 100, "long_window": 100},
         "short_window must be less than long_window"),
        ("sma_crossover", {"short_window": 90, "long_window": 60},
         "short_window must be less than long_window"),
    ],
)
def test_ordered_params(name, params, message):
    with pytest.raises(ValidationError, match=message):
        PARAMS_MODELS[name](**params)


def test_rsi_order_is_currently_implied_by_the_bounds():
    # oversold <= 40 < 60 <= overbought, so the oversold < overbought rule
    # can't fail today; it guards against the ranges being widened. If this
    # test fails, they now overlap: add a case to test_ordered_params.
    specs = {s["name"]: s for s in PARAM_SPECS["mean_reversion"]}
    assert specs["oversold"]["max"] < specs["overbought"]["min"]


def test_ordered_params_accept_valid_orders():
    PARAMS_MODELS["sma_crossover"](short_window=99, long_window=100)
    PARAMS_MODELS["mean_reversion"](oversold=40, overbought=60)


# ---- the request ----

def test_request_defaults_and_normalization():
    request = parse_request({"strategy": "sma_crossover", "ticker": " brk-b ",
                             "start": "2010-01-01", "end": "2020-01-01"})
    assert type(request).__name__ == "SmaCrossoverRequest"
    assert request.ticker == "BRK-B"
    assert (request.start, request.end) == (date(2010, 1, 1), date(2020, 1, 1))
    assert (request.cost_bps, request.n_boot, request.seed) == (5.0, 2000, 0)
    assert request.params.model_dump() == default_params("sma_crossover")


@pytest.mark.parametrize("strategy", ["regime_switch", "adaptive_ensemble", "nope", None])
def test_only_served_strategies_are_accepted(strategy):
    body = {**BASE, "strategy": strategy} if strategy else dict(BASE)
    with pytest.raises(ValidationError) as exc:
        parse_request(body)
    assert errors(exc.value)[0][1] in {"union_tag_invalid", "union_tag_not_found"}


def test_errors_name_only_the_chosen_strategys_fields():
    with pytest.raises(ValidationError) as exc:
        parse_request({**BASE, "strategy": "momentum", "params": {"lookback": 1000}})
    assert errors(exc.value) == [(("momentum", "params", "lookback"), "less_than_equal")]


@pytest.mark.parametrize(
    ("change", "error_type"),
    [
        ({"start": "2020-01-01", "end": "2020-01-01"}, "value_error"),
        ({"start": "2020-01-02", "end": "2020-01-01"}, "value_error"),
        ({"start": "01/02/2020"}, "date_from_datetime_parsing"),
        ({"ticker": ""}, "string_too_short"),
        ({"ticker": "SP Y"}, "value_error"),
        ({"ticker": "A" * 16}, "value_error"),
        ({"cost_bps": -0.5}, "greater_than_equal"),
        ({"cost_bps": 100.5}, "less_than_equal"),
        ({"cost_bps": "5"}, "float_type"),
        ({"cost_bps": math.nan}, "finite_number"),
        ({"n_boot": -1}, "greater_than_equal"),
        ({"n_boot": 5001}, "less_than_equal"),
        ({"n_boot": 100.0}, "int_type"),
        ({"seed": -1}, "greater_than_equal"),
        ({"seed": 2**63}, "less_than_equal"),
        ({"unexpected": 1}, "extra_forbidden"),
    ],
)
def test_request_field_rules(change, error_type):
    with pytest.raises(ValidationError) as exc:
        parse_request({**BASE, "strategy": "momentum", **change})
    assert error_type in {t for _, t in errors(exc.value)}


def test_request_limits_are_inclusive():
    parse_request({**BASE, "strategy": "momentum", "cost_bps": 0, "n_boot": 0, "seed": 0})
    parse_request({**BASE, "strategy": "momentum", "cost_bps": 100, "n_boot": 5000,
                   "seed": 2**63 - 1})


# ---- canonical form ----

def test_equivalent_requests_have_one_canonical_form():
    forms = [
        {**BASE, "strategy": "momentum"},
        {**BASE, "strategy": "momentum", "params": {}},
        {**BASE, "strategy": "momentum", "params": {"lookback": 20, "threshold": 0}},
        {"end": "2020-01-01", "params": {"threshold": 0.0, "lookback": 20}, "ticker": "spy",
         "strategy": "momentum", "start": "2010-01-01", "cost_bps": 5, "n_boot": 2000,
         "seed": 0},
    ]
    canon = {canonical_json(parse_request(f)) for f in forms}
    assert len(canon) == 1
    assert canon.pop() == (
        '{"cost_bps":5.0,"end":"2020-01-01","n_boot":2000,'
        '"params":{"lookback":20,"threshold":0.0},"seed":0,"start":"2010-01-01",'
        '"strategy":"momentum","ticker":"SPY"}'
    )


def test_different_requests_hash_differently():
    base = parse_request({**BASE, "strategy": "momentum"})
    variants = [
        {**BASE, "strategy": "momentum", "params": {"threshold": 0.005}},
        {**BASE, "strategy": "momentum", "seed": 1},
        {**BASE, "strategy": "momentum", "end": "2020-01-02"},
        {**BASE, "strategy": "momentum", "ticker": "QQQ"},
        {**BASE, "strategy": "sma_crossover"},
    ]
    hashes = {request_hash(base)} | {request_hash(parse_request(v)) for v in variants}
    assert len(hashes) == len(variants) + 1
    assert all(len(h) == 32 for h in hashes)
