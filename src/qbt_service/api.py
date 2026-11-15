"""
The HTTP API.

    uvicorn --factory qbt_service.api:create_app

A factory rather than a module-level app, so importing this module never
needs DATABASE_URL. At startup the app syncs the strategy catalog to the
database (see qbt_service.catalog) and fails to start if it can't.

Endpoints:

    GET  /strategies        the strategies and their parameter schemas
    POST /backtests         validate, enqueue, 202 with the job id
    GET  /backtests/{id}    status, attempts, timestamps, error, metrics
    GET  /models            the registered model versions
    GET  /models/{version}  one version, with its full evaluation
    POST /predict           score the latest (or a given) version; logged
    GET  /health            200 if the database answers, else 503

Every model entry and every prediction carries the model's out-of-sample
accuracy next to the baseline it was measured against, and whether it
beat it. None of the trained models has (results/ml/c2_results.json);
the API says so rather than leaving it to the docs.

Errors are RFC 9457 problem details (application/problem+json), each
with its own `type`, so a client can tell "this key was used for another
request" from "this body is invalid" although both are 422.

Idempotency follows the IETF Idempotency-Key draft
(draft-ietf-httpapi-idempotency-key-header):

- The key is optional. Without one, every POST creates a job.
- The same key with the same request (compared after validation, so key
  order, omitted defaults, 5 vs 5.0 and ticker case don't matter) returns
  the original job: 202 again, the same id, and `Idempotent-Replayed: true`.
- The same key with a different request gets 422. The draft reserves 409
  for a retry that arrives while the original is still being processed;
  that state never exists here, because the job is created in a single
  INSERT, and a concurrent retry waits on the unique index until the
  first transaction commits and then sees its job. So this API never
  returns 409.
- A malformed key gets 400.

Keys are global (there is no authentication, so there are no clients to
scope them to) and never expire.
"""

import logging
import math
import re
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import date, datetime
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Connection, Engine, func, select
from sqlalchemy.exc import InterfaceError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from qbt.ml.train import CLASS_NAMES, PREDICTS
from qbt_service.catalog import CATALOG, sync_catalog
from qbt_service.db import jobs, make_engine, metrics, predictions, results, strategies
from qbt_service.models import (
    PARAMS_MODELS,
    BacktestRequest,
    BacktestRequestBase,
    request_hash,
)
from qbt_service.prices import find_series
from qbt_service.queue import IdempotencyConflict, enqueue
from qbt_service.registry import (
    IntegrityError,
    ModelCache,
    get_model,
    latest_version,
    list_models,
    serving_input,
)
from qbt_service.settings import Settings

log = logging.getLogger(__name__)

PROBLEM_JSON = "application/problem+json"
RETRY_AFTER_SECONDS = 5
MAX_KEY_LENGTH = 255
# sf-string (RFC 8941 section 3.3.3): printable ASCII in double quotes,
# with only \" and \\ as escapes.
_SF_STRING = re.compile(r'"((?:[\x20\x21\x23-\x5b\x5d-\x7e]|\\["\\])*)"')
_RAW_KEY = re.compile(r"[\x21-\x7e](?:[\x20-\x7e]*[\x21-\x7e])?")

JobStatus = Literal["queued", "running", "succeeded", "failed"]


# ---- problem details ----

class Problem(BaseModel):
    """An RFC 9457 problem details object."""

    type: str = Field(description="Identifies the kind of problem; compare this, not `title`.")
    title: str = Field(description="Short summary of the kind of problem.")
    status: int = Field(description="The HTTP status code.")
    detail: str = Field(description="What went wrong with this request.")


class ValidationErrorItem(BaseModel):
    loc: list[str | int] = Field(description="Where: ['body', ...] or ['header', ...] etc.")
    msg: str
    type: str


class ValidationProblem(Problem):
    """A 422 problem. Only `validation-error` problems carry `errors`."""

    errors: list[ValidationErrorItem] | None = Field(
        default=None,
        description="One entry per invalid value; only when `type` is "
                    "/problems/validation-error.",
    )


class ProblemError(Exception):
    def __init__(self, status: int, slug: str, title: str, detail: str,
                 headers: dict[str, str] | None = None):
        super().__init__(detail)
        self.status, self.slug, self.title, self.detail = status, slug, title, detail
        self.headers = headers or {}


def _problem_type(slug: str) -> str:
    return f"/problems/{slug}"


def problem_response(status: int, slug: str, title: str, detail: str,
                     headers: dict[str, str] | None = None, **extensions: Any) -> JSONResponse:
    body = {"type": _problem_type(slug), "title": title, "status": status, "detail": detail}
    return JSONResponse(body | extensions, status_code=status, headers=headers,
                        media_type=PROBLEM_JSON)


class DatabaseUnavailable(ProblemError):
    def __init__(self) -> None:
        super().__init__(503, "database-unavailable", "Database unavailable",
                         "The database could not be reached. Nothing was changed unless a "
                         "retry says otherwise: retry with the same Idempotency-Key.",
                         headers={"Retry-After": str(RETRY_AFTER_SECONDS)})


@contextmanager
def _transaction(engine: Engine) -> Iterator[Connection]:
    """engine.begin(), with a lost or refused connection turned into a 503."""
    try:
        with engine.begin() as conn:
            yield conn
    except (OperationalError, InterfaceError) as exc:
        log.warning("database unavailable: %s", exc.orig if exc.orig is not None else exc)
        raise DatabaseUnavailable() from exc


# ---- the Idempotency-Key header ----

def parse_idempotency_key(value: str | None) -> str | None:
    """
    The key from an Idempotency-Key header, or None if there is none. The
    draft defines the header as a Structured Field String, i.e. quoted
    ("abc"); many clients send it bare (abc), so that is accepted as well
    and means the same key. ValueError for anything else.
    """
    if value is None:
        return None
    value = value.strip(" \t")
    if value.startswith('"'):
        match = _SF_STRING.fullmatch(value)
        if match is None:
            raise ValueError("a quoted key must be a valid Structured Field String "
                             "(printable ASCII; only \\\" and \\\\ escapes)")
        key = re.sub(r"\\(.)", r"\1", match.group(1))
    else:
        if not _RAW_KEY.fullmatch(value):
            raise ValueError("an unquoted key must be printable ASCII")
        key = value
    if not 1 <= len(key) <= MAX_KEY_LENGTH:
        raise ValueError(f"the key must be 1 to {MAX_KEY_LENGTH} characters")
    return key


# ---- response models ----

class StrategyInfo(BaseModel):
    name: str
    family: str
    description: str
    params_schema: dict[str, Any] = Field(
        description="JSON Schema of `params` for POST /backtests with this strategy."
    )
    defaults: dict[str, Any] = Field(description="The values used for omitted params.")
    version: int = Field(
        description="Behaviour version. It goes up whenever the strategy would compute "
                    "different positions for the same request; a stored job keeps the "
                    "version it ran (`strategy_version` on GET /backtests/{id})."
    )
    schema_hash: str = Field(
        description="SHA-256 of params_schema; changes whenever the schema does."
    )


class StrategyList(BaseModel):
    strategies: list[StrategyInfo]


class Health(BaseModel):
    status: Literal["ok"]


class BacktestAccepted(BaseModel):
    id: uuid.UUID = Field(description="The job id. Poll GET /backtests/{id}.")
    status: JobStatus = Field(
        description="The job's status now: `queued` for a new job; on a replay, wherever "
                    "the original job has got to."
    )
    href: str = Field(description="Where to poll; the same as the Location header.")


class ResultInfo(BaseModel):
    attempt: int = Field(description="The attempt that produced the result.")
    qbt_version: str
    numpy_version: str
    bootstrap_impl: str | None = Field(
        description="`cpp` or `numpy` (bit-identical); null when n_boot is 0."
    )
    duration_ms: int = Field(description="Time spent in the backtest itself.")
    n_days: int = Field(description="Trading days in the backtest.")
    created_at: datetime


class BacktestJob(BaseModel):
    id: uuid.UUID
    status: JobStatus = Field(
        description="`queued` (waiting, or waiting to retry after a failed attempt), "
                    "`running`, `succeeded` or `failed`. The last two are final."
    )
    attempts: int = Field(description="Attempts started so far (0 while first queued).")
    max_attempts: int = Field(description="After this many attempts the job fails.")
    created_at: datetime
    run_after: datetime = Field(
        description="Earliest time a queued job may start; later than created_at while a "
                    "retry is backing off."
    )
    started_at: datetime | None = Field(description="When the first attempt started.")
    heartbeat_at: datetime | None = Field(
        description="The running worker's last heartbeat (that of the latest attempt)."
    )
    finished_at: datetime | None = Field(description="Set once succeeded or failed.")
    last_error: str | None = Field(
        description="Why the latest attempt failed, if it did (kept while a retry is "
                    "queued; cleared on success)."
    )
    request: dict[str, Any] = Field(
        description="The request as validated and stored, with every default filled in."
    )
    price_series_id: int = Field(description="The price snapshot the backtest runs on.")
    strategy_version: int = Field(
        description="The strategy's behaviour version this job was submitted to (and, once "
                    "it has run, ran with). A job whose version is no longer current is "
                    "failed rather than run with newer code; resubmit it."
    )
    metrics: dict[str, float | None] | None = Field(
        description="full_report() metrics, once succeeded; otherwise null. A value that "
                    "is NaN (e.g. a Sharpe ratio with no variance) is null."
    )
    result: ResultInfo | None = Field(description="About the run, once succeeded.")


class ClassInfo(BaseModel):
    label: float = Field(description="The class as the model encodes it.")
    name: str


class OutOfSample(BaseModel):
    accuracy: float = Field(description="Share of out-of-sample months predicted correctly.")
    accuracy_ci: list[float] = Field(
        description="95% paired stationary-bootstrap interval (blocks of months).")
    n_months: int
    first_test_decision: date
    last_test_decision: date
    auc: float | None = Field(description="ROC AUC; macro one-vs-rest for regime.")
    log_loss: float
    brier: float = Field(description="Multiclass Brier score (sum over classes).")


class BaselineComparison(BaseModel):
    name: str = Field(description="The baseline this model has to beat (its bar).")
    description: str
    accuracy: float = Field(description="The baseline's accuracy on the same months.")
    difference: float = Field(description="Model accuracy minus baseline accuracy.")
    difference_ci: list[float] = Field(description="95% paired stationary-bootstrap interval.")
    mcnemar_p: float = Field(description="Exact two-sided McNemar test, same months.")
    beats: bool = Field(description="Interval above 0 and McNemar p < 0.05.")


class TrainingWindow(BaseModel):
    first_decision: date
    last_decision: date
    last_label_end: date = Field(description="When the last training label became known.")
    n_months: int


class ServingInput(BaseModel):
    as_of: date = Field(description="The latest complete month end in the snapshot; "
                                    "POST /predict forecasts the month after it.")
    features_sha256: str


class ArtifactInfo(BaseModel):
    format: str
    sha256: str = Field(description="Checked before the model is loaded.")
    bytes: int


class ModelInfo(BaseModel):
    version: int
    href: str
    model_type: str
    target: str
    predicts: str = Field(description="What a prediction is about, relative to as_of.")
    classes: list[ClassInfo]
    feature_set: str
    features: list[str]
    price_series_id: int = Field(description="The pinned price snapshot it was trained on.")
    training_window: TrainingWindow
    seed: int
    hyperparameters: dict[str, Any] = Field(description="The selected configuration.")
    out_of_sample: OutOfSample
    baseline: BaselineComparison
    summary: str = Field(description="The two lines above in words.")
    serving: ServingInput
    artifact: ArtifactInfo
    library_versions: dict[str, str]
    created_at: datetime


class ModelList(BaseModel):
    models: list[ModelInfo]


class ModelDetail(ModelInfo):
    metrics: dict[str, Any] = Field(description="The full evaluation, including every "
                                                "baseline and the per-fold selections.")
    training: dict[str, Any] = Field(description="Hyperparameter search and configuration.")
    serving_features: dict[str, float]


class PredictRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    version: int | None = Field(default=None, ge=1,
                                description="A registered version; omitted: the latest.")


class ClassProbability(ClassInfo):
    probability: float


class Prediction(BaseModel):
    prediction_id: int = Field(description="This prediction's row in the prediction log.")
    created_at: datetime
    model_version: int
    model_href: str
    target: str
    predicts: str
    as_of: date
    prediction: float = Field(description="The most probable class.")
    prediction_name: str
    probabilities: list[ClassProbability]
    features_sha256: str = Field(description="SHA-256 of the exact input scored.")
    out_of_sample: OutOfSample
    baseline: BaselineComparison
    summary: str


def _summary(oos: dict[str, Any], bar: dict[str, Any]) -> str:
    verdict = ("It beats that baseline." if bar["beats"] else
               "It did not beat that baseline: treat its predictions as an engineering "
               "demonstration, not as a forecast better than the baseline.")
    return (f"Out of sample ({oos['n_months']} months, {oos['first_test_decision']} to "
            f"{oos['last_test_decision']}) this model was right {oos['accuracy']:.1%} of the "
            f"time; its baseline, {bar['name']}, was right {bar['accuracy']:.1%} on the same "
            f"months. {verdict}")


def _class_infos(target: str, classes: list[float]) -> list[ClassInfo]:
    return [ClassInfo(label=c, name=CLASS_NAMES[target][c]) for c in classes]


def _model_info(row: Any, detail: bool = False) -> ModelInfo:
    oos, bar = row["metrics"]["out_of_sample"], row["metrics"]["bar"]
    fields: dict[str, Any] = {
        "version": row["version"], "href": f"/models/{row['version']}",
        "model_type": row["model_type"], "target": row["target"],
        "predicts": PREDICTS[row["target"]],
        "classes": _class_infos(row["target"], row["classes"]),
        "feature_set": row["feature_set"], "features": row["features"],
        "price_series_id": row["price_series_id"],
        "training_window": TrainingWindow(first_decision=row["train_start"],
                                       last_decision=row["train_end"],
                                       last_label_end=row["train_label_end"],
                                       n_months=row["n_train"]),
        "seed": row["seed"], "hyperparameters": row["hyperparameters"]["selected"],
        "out_of_sample": OutOfSample(**{k: oos[k] for k in OutOfSample.model_fields}),
        "baseline": BaselineComparison(**{k: bar[k] for k in BaselineComparison.model_fields}),
        "summary": _summary(oos, bar),
        "serving": ServingInput(as_of=row["serving_as_of"],
                             features_sha256=bytes(row["serving_features_sha256"]).hex()),
        "artifact": ArtifactInfo(format=row["artifact_format"],
                              sha256=bytes(row["artifact_sha256"]).hex(),
                              bytes=row["artifact_bytes"]),
        "library_versions": row["library_versions"], "created_at": row["created_at"],
    }
    if not detail:
        return ModelInfo(**fields)
    return ModelDetail(**fields, metrics=row["metrics"],
                       training={"hyperparameters": row["hyperparameters"],
                                 "config": row["training_config"]},
                       serving_features=row["serving_features"])


# ---- OpenAPI examples ----

EXAMPLE_ID = "0b6c3f5e-8d1a-4a43-9c55-2f1e7d3b9a10"
EXAMPLE_REQUEST: dict[str, Any] = {
    "strategy": "sma_crossover", "params": {"short_window": 50, "long_window": 200},
    "ticker": "SPY", "start": "2015-01-01", "end": "2025-01-01",
    "cost_bps": 5.0, "n_boot": 2000, "seed": 0,
}


def _request_examples() -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    for name, model in PARAMS_MODELS.items():
        examples[name] = {
            "summary": f"{name}, default parameters",
            "value": {"strategy": name, "params": model().model_dump(), "ticker": "SPY",
                      "start": "2015-01-01", "end": "2025-01-01"},
        }
    examples["minimal"] = {
        "summary": "Only the required fields (params, cost_bps, n_boot, seed default)",
        "value": {"strategy": "momentum", "ticker": "QQQ", "start": "2012-01-01",
                  "end": "2022-01-01"},
    }
    examples["everything"] = {"summary": "Every field given", "value": EXAMPLE_REQUEST}
    return examples


def _problem_example(status: int, slug: str, title: str, detail: str,
                     **extensions: Any) -> dict[str, Any]:
    return {"type": _problem_type(slug), "title": title, "status": status, "detail": detail,
            **extensions}


def _problem_content(examples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {PROBLEM_JSON: {"examples": {
        name: {"summary": example["title"], "value": example}
        for name, example in examples.items()
    }}}


NOT_FOUND_EXAMPLE = _problem_example(
    404, "job-not-found", "Job not found", f"There is no backtest job {EXAMPLE_ID}.")
UNAVAILABLE_EXAMPLE = _problem_example(
    503, "database-unavailable", "Database unavailable",
    DatabaseUnavailable().detail)
VALIDATION_EXAMPLE = _problem_example(
    422, "validation-error", "Invalid request",
    "The request has 1 invalid value; see `errors`.",
    errors=[{"loc": ["body", "sma_crossover", "params"], "type": "value_error",
             "msg": "Value error, short_window must be less than long_window"}])
KEY_REUSED_EXAMPLE = _problem_example(
    422, "idempotency-key-reused", "Idempotency-Key reused",
    "This Idempotency-Key was already used for a different request. Use a new key "
    "for a new request.")
NO_PRICES_EXAMPLE = _problem_example(
    422, "no-price-data", "No price data",
    "No price snapshot for SPY covers [2015-01-01, 2025-01-01). Ingest one first: "
    "python -m qbt_service.ingest SPY --start 2015-01-01 --end 2025-01-01")
BAD_KEY_EXAMPLE = _problem_example(
    400, "invalid-idempotency-key", "Invalid Idempotency-Key",
    "Idempotency-Key: the key must be 1 to 255 characters.")

MODEL_NOT_FOUND_EXAMPLE = _problem_example(
    404, "model-not-found", "Model not found", "There is no model version 7.")
NO_MODEL_EXAMPLE = _problem_example(
    422, "no-model-registered", "No model registered",
    "No model is registered. Train one: python -m qbt_service.train SPY --start 2008-01-01 "
    "--end 2025-01-01")
UNKNOWN_VERSION_EXAMPLE = _problem_example(
    422, "unknown-model-version", "Unknown model version", "There is no model version 7.")
INTEGRITY_EXAMPLE = _problem_example(
    500, "model-integrity-error", "Model failed its integrity check",
    "model version 1: the stored artifact doesn't match its SHA-256; refusing to load it")

DESCRIPTION_503 = ("The database is unreachable. Retry after `Retry-After` seconds. A POST "
                   "may or may not have created its job; retrying with the same "
                   "Idempotency-Key is safe either way.")

DESCRIPTION_503_HEALTH = "The database is unreachable from this API process."


def _job_example(status: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": EXAMPLE_ID, "status": status, "attempts": 0, "max_attempts": 3,
        "created_at": "2026-11-14T10:00:00.000000Z", "run_after": "2026-11-14T10:00:00.000000Z",
        "started_at": None, "heartbeat_at": None, "finished_at": None, "last_error": None,
        "request": EXAMPLE_REQUEST, "price_series_id": 7, "strategy_version": 1,
        "metrics": None, "result": None,
    }
    started = {"attempts": 1, "started_at": "2026-11-14T10:00:01.200000Z"}
    if status == "running":
        return base | started | {"heartbeat_at": "2026-11-14T10:00:11.200000Z"}
    if status == "succeeded":
        return base | started | {
            "heartbeat_at": "2026-11-14T10:00:01.200000Z",
            "finished_at": "2026-11-14T10:00:02.900000Z",
            "metrics": {"annualized_volatility": 0.121, "cagr": 0.074, "exposure": 0.81,
                        "max_drawdown": -0.198, "num_trades": 9.0, "sharpe_ratio": 0.66,
                        "sharpe_ci_low": 0.05, "sharpe_ci_high": 1.27, "sortino_ratio": 0.91,
                        "total_return": 1.04, "turnover": 0.0036, "win_rate": 0.54},
            "result": {"attempt": 1, "qbt_version": "0.1.0", "numpy_version": "2.3.4",
                       "bootstrap_impl": "cpp", "duration_ms": 1480, "n_days": 2515,
                       "created_at": "2026-11-14T10:00:02.900000Z"},
        }
    if status == "failed":
        return base | started | {
            "heartbeat_at": "2026-11-14T10:00:01.200000Z",
            "finished_at": "2026-11-14T10:00:01.400000Z",
            "last_error": "backtest raised: Traceback (most recent call last): ...",
        }
    return base


# ---- the app ----

def _strategy_list() -> StrategyList:
    return StrategyList(strategies=[
        StrategyInfo(name=e.name, family=e.family, description=e.description,
                     params_schema=e.param_schema, defaults=e.defaults,
                     version=e.version, schema_hash=e.schema_hash.hex())
        for e in CATALOG.values()
    ])


def _finite(value: float | None) -> float | None:
    return None if value is None or math.isnan(value) or math.isinf(value) else value


def _validation_problem(exc: RequestValidationError) -> JSONResponse:
    errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""),
               "type": e.get("type", "")} for e in exc.errors()]
    count = len(errors)
    return problem_response(
        422, "validation-error", "Invalid request",
        f"The request has {count} invalid value{'s' if count != 1 else ''}; see `errors`.",
        errors=errors)


def _use_problem_media_type(schema: dict[str, Any]) -> None:
    """
    FastAPI files every response model under application/json. Move the
    problem models to application/problem+json, which is what is sent.
    """
    problem_refs = {f"#/components/schemas/{name}" for name in ("Problem", "ValidationProblem")}
    for operations in schema["paths"].values():
        for operation in operations.values():
            for response in operation.get("responses", {}).values():
                content = response.get("content", {})
                plain = content.get("application/json")
                if plain and plain.get("schema", {}).get("$ref") in problem_refs:
                    content.setdefault(PROBLEM_JSON, {})["schema"] = content.pop(
                        "application/json")["schema"]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = Settings.from_env() if settings is None else settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url, settings.statement_timeout_seconds)
        with engine.begin() as conn:
            app.state.strategy_ids = sync_catalog(conn)
        app.state.engine = engine
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(
        title="qbt backtest service", version="0.1.0", lifespan=lifespan,
        description=(
            "Submit backtests of the qbt strategies and poll for their results. "
            "Jobs run on workers in the background, at least once and "
            "deterministically. Errors are RFC 9457 problem details."
        ),
    )
    strategy_list = _strategy_list()  # fixed for the life of the process
    model_cache = ModelCache()

    @app.exception_handler(ProblemError)
    async def problem_error(request: Request, exc: ProblemError) -> JSONResponse:
        return problem_response(exc.status, exc.slug, exc.title, exc.detail,
                                headers=exc.headers or None)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _validation_problem(exc)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        title = {404: "Not Found", 405: "Method Not Allowed"}.get(exc.status_code, "Error")
        return problem_response(exc.status_code, "http-error", title, str(exc.detail),
                                headers=getattr(exc, "headers", None))

    @app.get("/strategies", response_model=StrategyList, tags=["strategies"])
    def list_strategies() -> StrategyList:
        """The strategies POST /backtests accepts, with their parameter schemas."""
        return strategy_list

    @app.get(
        "/health", response_model=Health, tags=["health"], summary="Health check",
        responses={503: {"model": Problem, "description": DESCRIPTION_503_HEALTH,
                         "headers": {"Retry-After": {"description": "Seconds to wait.",
                                                     "schema": {"type": "integer"}}},
                         "content": _problem_content(
                             {"database_unavailable": UNAVAILABLE_EXAMPLE})}},
    )
    def health() -> Health:
        """
        200 if this process can run a query on the database (SELECT 1),
        else 503. The container healthcheck (python -m qbt_service.health
        api) calls this.
        """
        with _transaction(app.state.engine) as conn:
            conn.execute(select(1))
        return Health(status="ok")

    @app.post(
        "/backtests", status_code=202, response_model=BacktestAccepted, tags=["backtests"],
        summary="Submit a backtest",
        responses={
            202: {
                "description": "Accepted: the job is queued (or, on a replay, the original "
                               "job is returned).",
                "headers": {
                    "Location": {"description": "/backtests/{id}", "schema": {"type": "string"}},
                    "Idempotent-Replayed": {
                        "description": "`true` when an earlier request with this "
                                       "Idempotency-Key created the job.",
                        "schema": {"type": "string", "enum": ["true"]},
                    },
                },
                "content": {"application/json": {"examples": {
                    "accepted": {"summary": "A new job", "value": {
                        "id": EXAMPLE_ID, "status": "queued",
                        "href": f"/backtests/{EXAMPLE_ID}"}},
                    "replayed": {"summary": "A replay, after the job finished", "value": {
                        "id": EXAMPLE_ID, "status": "succeeded",
                        "href": f"/backtests/{EXAMPLE_ID}"}},
                }}},
            },
            400: {"model": Problem, "description": "The Idempotency-Key header is malformed.",
                  "content": _problem_content({"invalid_key": BAD_KEY_EXAMPLE})},
            422: {"model": ValidationProblem,
                  "description": "The body is invalid, no price snapshot covers the range, "
                                 "or the Idempotency-Key was used for a different request. "
                                 "`type` says which; only `validation-error` has `errors`.",
                  "content": _problem_content({
                      "validation_error": VALIDATION_EXAMPLE,
                      "idempotency_key_reused": KEY_REUSED_EXAMPLE,
                      "no_price_data": NO_PRICES_EXAMPLE,
                  })},
            503: {"model": Problem, "description": DESCRIPTION_503,
                  "headers": {"Retry-After": {"description": "Seconds to wait.",
                                              "schema": {"type": "integer"}}},
                  "content": _problem_content({"database_unavailable": UNAVAILABLE_EXAMPLE})},
        },
    )
    def submit_backtest(
        body: Annotated[BacktestRequest, Body(openapi_examples=_request_examples())],
        response: Response,
        idempotency_key: Annotated[str | None, Header(
            alias="Idempotency-Key",
            description="Makes the POST safe to retry: the same key with the same request "
                        "returns the original job instead of creating another. A Structured "
                        'Field String ("...", as in the IETF draft) or the bare value; '
                        f"1 to {MAX_KEY_LENGTH} printable ASCII characters. Optional.",
            examples=['"8e03978e-40d5-43e8-bc93-6894a57f9324"'],
        )] = None,
    ) -> BacktestAccepted:
        """
        Validates the request, picks the newest price snapshot that covers
        [start, end), and queues a job. Returns 202 at once; poll
        GET /backtests/{id} for the outcome.

        `params` is checked against the strategy's schema (GET /strategies),
        and omitted fields take their defaults. The stored request has
        every default filled in, and two requests that validate to the
        same values are the same request for the Idempotency-Key.

        Idempotency (IETF draft-ietf-httpapi-idempotency-key-header): with
        an `Idempotency-Key`, a retry of the same request returns the same
        job with `Idempotent-Replayed: true`; the same key with a different
        request gets 422 (`/problems/idempotency-key-reused`). 409, which
        the draft uses for a retry while the original is still being
        processed, can't happen: the job is created atomically, and a
        concurrent retry waits for it and then replays it.
        """
        try:
            key = parse_idempotency_key(idempotency_key)
        except ValueError as exc:
            raise ProblemError(400, "invalid-idempotency-key", "Invalid Idempotency-Key",
                               f"Idempotency-Key: {exc}.") from exc

        request: BacktestRequestBase = body
        with _transaction(app.state.engine) as conn:
            series_id = find_series(conn, request.ticker, request.start, request.end)
            if series_id is None:
                raise ProblemError(
                    422, "no-price-data", "No price data",
                    f"No price snapshot for {request.ticker} covers [{request.start}, "
                    f"{request.end}). Ingest one first: python -m qbt_service.ingest "
                    f"{request.ticker} --start {request.start} --end {request.end}")
            try:
                enqueued = enqueue(
                    conn, strategy_id=app.state.strategy_ids[request.strategy],
                    price_series_id=series_id, request=request.model_dump(mode="json"),
                    request_hash=request_hash(request), idempotency_key=key,
                    max_attempts=settings.max_attempts)
            except IdempotencyConflict as exc:
                raise ProblemError(
                    422, "idempotency-key-reused", "Idempotency-Key reused",
                    "This Idempotency-Key was already used for a different request. Use a "
                    "new key for a new request.") from exc
            status: JobStatus = "queued" if enqueued.created else conn.execute(
                select(jobs.c.status).where(jobs.c.id == enqueued.job_id)).scalar_one()

        href = f"/backtests/{enqueued.job_id}"
        response.headers["Location"] = href
        if not enqueued.created:
            response.headers["Idempotent-Replayed"] = "true"
        return BacktestAccepted(id=enqueued.job_id, status=status, href=href)

    @app.get(
        "/backtests/{job_id}", response_model=BacktestJob, tags=["backtests"],
        summary="Get a backtest job",
        responses={
            404: {"model": Problem, "description": "No job has this id.",
                  "content": _problem_content({"not_found": NOT_FOUND_EXAMPLE})},
            422: {"model": ValidationProblem, "description": "The id is not a UUID.",
                  "content": _problem_content({"validation_error": _problem_example(
                      422, "validation-error", "Invalid request",
                      "The request has 1 invalid value; see `errors`.",
                      errors=[{"loc": ["path", "job_id"], "type": "uuid_parsing",
                               "msg": "Input should be a valid UUID, invalid length: "
                                      "expected length 32 for simple format, found 3"}])})},
            503: {"model": Problem, "description": DESCRIPTION_503,
                  "headers": {"Retry-After": {"description": "Seconds to wait.",
                                              "schema": {"type": "integer"}}},
                  "content": _problem_content({"database_unavailable": UNAVAILABLE_EXAMPLE})},
        },
    )
    def get_backtest(job_id: uuid.UUID) -> BacktestJob:
        """
        The job's status, attempts, timestamps and last error, the request
        as stored, and, once it has succeeded, its metrics and run details.

        A job can go back from `running` to `queued`: a failed attempt that
        will be retried, or a worker that died or shut down. `attempts`
        counts attempts started; `last_error` says why the previous one
        ended.
        """
        with _transaction(app.state.engine) as conn:
            row = conn.execute(
                select(jobs.c.id, jobs.c.status, jobs.c.attempts, jobs.c.max_attempts,
                       jobs.c.created_at, jobs.c.run_after, jobs.c.started_at,
                       jobs.c.heartbeat_at, jobs.c.finished_at, jobs.c.last_error,
                       jobs.c.request, jobs.c.price_series_id,
                       strategies.c.version.label("strategy_version"))
                .select_from(jobs.join(strategies, jobs.c.strategy_id == strategies.c.id))
                .where(jobs.c.id == job_id)
            ).mappings().one_or_none()
            if row is None:
                raise ProblemError(404, "job-not-found", "Job not found",
                                   f"There is no backtest job {job_id}.")
            # complete() commits the status, the result and the metrics in
            # one transaction, so a job read as succeeded has both.
            result = values = None
            if row["status"] == "succeeded":
                result = conn.execute(
                    select(results.c.attempt, results.c.qbt_version, results.c.numpy_version,
                           results.c.bootstrap_impl, results.c.duration_ms,
                           func.cardinality(results.c.dates).label("n_days"),
                           results.c.created_at)
                    .where(results.c.job_id == job_id)
                ).mappings().one()
                values = {name: _finite(value) for name, value in conn.execute(
                    select(metrics.c.name, metrics.c.value)
                    .where(metrics.c.job_id == job_id).order_by(metrics.c.name)
                )}

        return BacktestJob(**row, metrics=values,
                           result=ResultInfo(**result) if result is not None else None)

    unavailable_503 = {
        "model": Problem, "description": DESCRIPTION_503,
        "headers": {"Retry-After": {"description": "Seconds to wait.",
                                    "schema": {"type": "integer"}}},
        "content": _problem_content({"database_unavailable": UNAVAILABLE_EXAMPLE})}

    @app.get("/models", response_model=ModelList, tags=["models"],
             summary="List registered models", responses={503: unavailable_503})
    def models_list() -> ModelList:
        """
        Every registered model version, oldest first, each with its
        out-of-sample accuracy next to the baseline it had to beat.
        """
        with _transaction(app.state.engine) as conn:
            rows = list_models(conn)
        return ModelList(models=[_model_info(row) for row in rows])

    @app.get(
        "/models/{version}", response_model=ModelDetail, tags=["models"],
        summary="Get a model version",
        responses={404: {"model": Problem, "description": "No model has this version.",
                         "content": _problem_content({"not_found": MODEL_NOT_FOUND_EXAMPLE})},
                   503: unavailable_503},
    )
    def model_detail(version: int) -> ModelDetail:
        """One version with its full evaluation, hyperparameter search and serving input."""
        with _transaction(app.state.engine) as conn:
            row = get_model(conn, version)
        if row is None:
            raise ProblemError(404, "model-not-found", "Model not found",
                               f"There is no model version {version}.")
        detail = _model_info(row, detail=True)
        assert isinstance(detail, ModelDetail)
        return detail

    @app.post(
        "/predict", response_model=Prediction, tags=["models"], summary="Predict",
        responses={
            422: {"model": ValidationProblem,
                  "description": "The body is invalid, no model is registered, or the "
                                 "version doesn't exist. `type` says which.",
                  "content": _problem_content({"no_model": NO_MODEL_EXAMPLE,
                                               "unknown_version": UNKNOWN_VERSION_EXAMPLE})},
            500: {"model": Problem,
                  "description": "The stored model or its input failed its SHA-256 check. "
                                 "Nothing was scored or logged.",
                  "content": _problem_content({"integrity": INTEGRITY_EXAMPLE})},
            503: unavailable_503,
        },
    )
    def predict(body: Annotated[PredictRequest | None, Body()] = None) -> Prediction:
        """
        Scores a model version (the latest unless `version` is given) on the
        latest complete month in its pinned price snapshot (`as_of`), i.e.
        forecasts the month after as_of. That month's label isn't in the
        snapshot, so the model never trained on it. Every prediction is
        logged, and every response carries the model's out-of-sample
        accuracy and the baseline it was measured against.
        """
        request = body or PredictRequest()
        with _transaction(app.state.engine) as conn:
            version = request.version if request.version is not None else latest_version(conn)
            if version is None:
                raise ProblemError(
                    422, "no-model-registered", "No model registered",
                    "No model is registered. Train one: python -m qbt_service.train SPY "
                    "--start 2008-01-01 --end 2025-01-01")
            row = get_model(conn, version)
            if row is None:
                raise ProblemError(422, "unknown-model-version", "Unknown model version",
                                   f"There is no model version {version}.")
            try:
                model = model_cache.load(conn, version)
                as_of, features = serving_input(row)
            except IntegrityError as exc:
                log.error("%s", exc)
                raise ProblemError(500, "model-integrity-error",
                                   "Model failed its integrity check", str(exc)) from exc
            proba = model.predict_proba(features)[0]
            label = model.classes[int(proba.argmax())]
            logged = conn.execute(
                predictions.insert().values(
                    model_version=version, as_of=as_of,
                    features_sha256=row["serving_features_sha256"], prediction=label,
                    probabilities=[float(p) for p in proba])
                .returning(predictions.c.id, predictions.c.created_at)
            ).one()
        info = _model_info(row)
        names = CLASS_NAMES[row["target"]]
        return Prediction(
            prediction_id=logged.id, created_at=logged.created_at, model_version=version,
            model_href=info.href, target=row["target"], predicts=info.predicts, as_of=as_of,
            prediction=label, prediction_name=names[label],
            probabilities=[ClassProbability(label=c, name=names[c], probability=float(p))
                           for c, p in zip(model.classes, proba, strict=True)],
            features_sha256=info.serving.features_sha256,
            out_of_sample=info.out_of_sample, baseline=info.baseline, summary=info.summary)

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            schema = get_openapi(title=app.title, version=app.version,
                                 description=app.description, routes=app.routes)
            _use_problem_media_type(schema)
            # Added here, not in responses=: FastAPI drops None values from
            # examples given there, and a queued job's nulls are the point.
            ok = schema["paths"]["/backtests/{job_id}"]["get"]["responses"]["200"]
            ok["content"]["application/json"]["examples"] = {
                status: {"summary": status.capitalize(), "value": _job_example(status)}
                for status in ("queued", "running", "succeeded", "failed")
            }
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]
    return app
