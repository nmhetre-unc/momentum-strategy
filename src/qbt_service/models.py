"""
Request models for POST /backtests, generated from qbt.strategies.PARAM_SPECS.

PARAM_SPECS already drives the dashboard's sliders, so the API's bounds
and defaults come from the same place and can't drift from them: each
spec becomes a field with ge=min, le=max, the spec's default, and its
label and help text as title and description. A field is an int if
the spec's min, max, default and step are all ints, otherwise a float.
`step` is a slider setting and is not enforced.

Two rules PARAM_SPECS can't express are added here: the short SMA window
must be shorter than the long one, and the RSI oversold level must sit
below the overbought level.

Parameters are validated strictly: an int field rejects 5.0, "5" and
true, and a float field accepts an int (stored as a float, so 0 and 0.0
are the same request) but rejects strings, NaN and infinity.

Only the four base strategies are exposed for now; the adaptive ones
take non-scalar parameters and get hand-written models later.
"""

import hashlib
import json
import re
from datetime import date
from typing import Annotated, Any, ClassVar, Literal, Union, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    TypeAdapter,
    create_model,
    field_validator,
    model_validator,
)

from qbt.strategies import PARAM_SPECS, STRATEGIES, STRATEGY_DOCS
from qbt_service.prices import normalize_ticker

# (lower, upper) pairs that must satisfy params[lower] < params[upper].
# The RSI pair can't fail with today's PARAM_SPECS ranges (oversold <= 40,
# overbought >= 60); the rule is there in case those ranges are widened.
ORDERED_PARAMS: dict[str, list[tuple[str, str]]] = {
    "sma_crossover": [("short_window", "long_window")],
    "mean_reversion": [("oversold", "overbought")],
}

MAX_N_BOOT = 5000
MAX_COST_BPS = 100.0
MAX_SEED = 2**63 - 1
# Yahoo symbols: BRK-B, ^GSPC, EURUSD=X, RDS.A.
TICKER_PATTERN = re.compile(r"[A-Z0-9.\-^=]{1,15}")


class StrategyParams(BaseModel):
    """Base for the generated per-strategy parameter models."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)
    strategy: ClassVar[str] = ""

    @model_validator(mode="after")
    def _check_order(self) -> "StrategyParams":
        for lower, upper in ORDERED_PARAMS.get(self.strategy, []):
            if getattr(self, lower) >= getattr(self, upper):
                raise ValueError(f"{lower} must be less than {upper}")
        return self


def _class_name(strategy: str, suffix: str) -> str:
    return "".join(part.title() for part in strategy.split("_")) + suffix


def _field_type(spec: dict) -> type:
    numbers = [spec["min"], spec["max"], spec["default"], spec.get("step", 1)]
    return int if all(isinstance(v, int) and not isinstance(v, bool) for v in numbers) else float


def _params_model(name: str) -> type[StrategyParams]:
    fields: dict[str, Any] = {}
    # PARAM_SPECS mixes str and number values, so its inferred type is too loose.
    specs = cast(list[dict[str, Any]], PARAM_SPECS[name])
    for spec in specs:
        kind = _field_type(spec)
        fields[spec["name"]] = (kind, Field(
            default=kind(spec["default"]), ge=kind(spec["min"]), le=kind(spec["max"]),
            title=spec["label"], description=spec["help"],
        ))
    model = create_model(_class_name(name, "Params"), __base__=StrategyParams,
                         __module__=__name__, __doc__=STRATEGY_DOCS[name]["what"], **fields)
    model.strategy = name
    return model


SERVED_STRATEGIES = tuple(sorted(STRATEGIES))
PARAMS_MODELS: dict[str, type[StrategyParams]] = {
    name: _params_model(name) for name in SERVED_STRATEGIES
}


class BacktestRequestBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Narrowed by each generated subclass to Literal[name] and its params model.
    strategy: str
    params: StrategyParams
    ticker: str = Field(min_length=1, max_length=20,
                        description="Yahoo Finance symbol, e.g. SPY or BRK-B; upper-cased.")
    start: date = Field(description="First date of the backtest (inclusive).")
    end: date = Field(description="End of the backtest (exclusive, as in yfinance).")
    cost_bps: Annotated[float, Strict()] = Field(
        default=5.0, ge=0.0, le=MAX_COST_BPS, allow_inf_nan=False,
        description="One-way transaction cost per unit of position change, in basis points.",
    )
    n_boot: Annotated[int, Strict()] = Field(
        default=2000, ge=0, le=MAX_N_BOOT,
        description="Stationary-bootstrap replicates for the Sharpe interval; 0 skips it.",
    )
    seed: Annotated[int, Strict()] = Field(default=0, ge=0, le=MAX_SEED,
                                           description="Bootstrap seed.")

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, value: str) -> str:
        ticker = normalize_ticker(value)
        if not TICKER_PATTERN.fullmatch(ticker):
            raise ValueError("ticker may contain only letters, digits and . - ^ = (max 15)")
        return ticker

    @model_validator(mode="after")
    def _dates(self) -> "BacktestRequestBase":
        if self.start >= self.end:
            raise ValueError("start must be before end")
        return self


def _request_model(name: str) -> type[BacktestRequestBase]:
    params_model = PARAMS_MODELS[name]
    # Literal[name] is built from a runtime string, so it isn't a static type
    # form and can't match create_model's TypeForm parameter; it is a real
    # type to pydantic at runtime.
    strategy_type: Any = Literal[name]
    return create_model(
        _class_name(name, "Request"), __base__=BacktestRequestBase, __module__=__name__,
        strategy=(strategy_type, ...),
        params=(params_model, Field(default_factory=params_model)),
    )


REQUEST_MODELS: dict[str, type[BacktestRequestBase]] = {
    name: _request_model(name) for name in SERVED_STRATEGIES
}

# The POST /backtests body: the `strategy` field picks the model, so a
# validation error names only the fields of the strategy that was asked for.
# The Union is built from the registry at import time, which mypy can't follow.
BacktestRequest: Any = Annotated[
    Union[tuple(REQUEST_MODELS.values())],  # noqa: UP007
    Field(discriminator="strategy"),
]
request_adapter: TypeAdapter[BacktestRequestBase] = TypeAdapter(BacktestRequest)


def parse_request(data: Any) -> BacktestRequestBase:
    """Validates a request body (a dict) into its strategy's request model."""
    return request_adapter.validate_python(data)


def canonical_json(model: BaseModel) -> str:
    """
    One string per distinct request: defaults filled in, keys sorted, no
    whitespace. Floats are written with Python's shortest round-trip repr,
    so equal values always serialize the same way.
    """
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def request_hash(model: BaseModel) -> bytes:
    """SHA-256 of canonical_json(); what an Idempotency-Key is checked against."""
    return hashlib.sha256(canonical_json(model).encode()).digest()
