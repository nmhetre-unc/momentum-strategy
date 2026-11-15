"""
The service's schema as SQLAlchemy Core tables. Alembic compares this
metadata with the database (`alembic check`), but the migrations under
qbt_service/migrations are written by hand: autogenerate doesn't see
CHECK constraints, partial-index predicates or table storage options.

Design notes the code below relies on:

- price_series rows are immutable snapshots. yfinance's adjusted closes
  are rewritten after every dividend or split, so re-ingesting creates a
  new series rather than updating bars, and a job pins the series it was
  submitted against. That keeps a rerun bit-identical to the first run.
- jobs is the queue. Heartbeats update only lease_expires_at and
  heartbeat_at, and no index covers either column, so every heartbeat
  can be a HOT (in-page) update; the migration also sets fillfactor=90
  to leave room for them.
- A running job always has a lease_token. Finishing a job is fenced on
  that token, so a worker that lost its lease can't write a result.
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Double,
    Engine,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    PrimaryKeyConstraint,
    SmallInteger,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

# Deterministic constraint names, so a later migration (or a downgrade)
# can refer to a constraint by name without looking it up.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

JOB_STATUSES = ("queued", "running", "succeeded", "failed")
job_status = Enum(*JOB_STATUSES, name="job_status")

NOW = text("now()")

price_series = Table(
    "price_series", metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("ticker", Text, nullable=False),
    Column("source", Text, nullable=False, server_default=text("'yfinance'")),
    Column("adjusted", Boolean, nullable=False, server_default=text("true")),
    # The range that was requested, [fetch_start, fetch_end) like yfinance's
    # own start/end. A snapshot covers a backtest's [start, end) when this
    # range does; first_date/last_date alone can't say that, because the
    # first bar falls on the first trading day on or after fetch_start.
    Column("fetch_start", Date, nullable=False),
    Column("fetch_end", Date, nullable=False),
    Column("first_date", Date, nullable=False),
    Column("last_date", Date, nullable=False),
    Column("n_bars", Integer, nullable=False),
    # Over the fetch range and every bar (see qbt_service.prices.content_hash).
    Column("content_sha256", LargeBinary, nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    CheckConstraint("n_bars > 0", name="n_bars_positive"),
    CheckConstraint("first_date <= last_date", name="date_order"),
    CheckConstraint("octet_length(content_sha256) = 32", name="sha256_length"),
    CheckConstraint("fetch_start < fetch_end", name="fetch_range_order"),
    CheckConstraint("first_date >= fetch_start AND last_date < fetch_end",
                    name="bars_within_fetch_range"),
    # Re-ingesting unchanged data for the same range finds the existing
    # series instead of adding one.
    UniqueConstraint("ticker", "source", "content_sha256"),
)
# "Newest snapshot of this ticker that covers [start, end]", at submit time.
Index("price_series_latest", price_series.c.ticker, price_series.c.fetched_at.desc())

price_bars = Table(
    "price_bars", metadata,
    Column("series_id", BigInteger,
           ForeignKey("price_series.id", ondelete="RESTRICT"), nullable=False),
    Column("date", Date, nullable=False),
    Column("open", Double),
    Column("high", Double),
    Column("low", Double),
    # double precision, never numeric: values must round-trip to NumPy exactly.
    Column("close", Double, nullable=False),
    Column("volume", BigInteger),
    # Also the only index the worker's price query (one series, a date range) needs.
    PrimaryKeyConstraint("series_id", "date"),
)

strategies = Table(
    "strategies", metadata,
    Column("id", BigInteger, Identity(always=True), primary_key=True),
    Column("name", Text, nullable=False),
    Column("schema_hash", LargeBinary, nullable=False),
    Column("param_schema", JSONB, nullable=False),
    Column("description", Text),
    Column("is_current", Boolean, nullable=False, server_default=text("true")),
    Column("registered_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    # A new row per parameter-schema change: old jobs keep the schema they were validated with.
    UniqueConstraint("name", "schema_hash"),
)
Index("strategies_one_current", strategies.c.name, unique=True,
      postgresql_where=strategies.c.is_current)

jobs = Table(
    "jobs", metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("strategy_id", BigInteger,
           ForeignKey("strategies.id", ondelete="RESTRICT"), nullable=False),
    Column("price_series_id", BigInteger,
           ForeignKey("price_series.id", ondelete="RESTRICT"), nullable=False),
    Column("request", JSONB, nullable=False),        # validated, defaults filled in
    Column("request_hash", LargeBinary, nullable=False),
    Column("idempotency_key", Text),
    Column("status", job_status, nullable=False, server_default=text("'queued'")),
    Column("attempts", SmallInteger, nullable=False, server_default=text("0")),
    Column("max_attempts", SmallInteger, nullable=False, server_default=text("3")),
    Column("run_after", DateTime(timezone=True), nullable=False, server_default=NOW),
    Column("lease_token", UUID(as_uuid=True)),       # fencing token, new on every claim
    Column("lease_owner", Text),                     # worker id, for operators
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("heartbeat_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    Column("last_error", Text),
    CheckConstraint(
        "(status = 'running') = (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)",
        name="running_has_lease",
    ),
    CheckConstraint(
        "(status IN ('succeeded', 'failed')) = (finished_at IS NOT NULL)",
        name="finished_iff_terminal",
    ),
    CheckConstraint("attempts >= 0 AND attempts <= max_attempts", name="attempts_in_range"),
    CheckConstraint("max_attempts >= 1", name="max_attempts_positive"),
    CheckConstraint("octet_length(request_hash) = 32", name="request_hash_length"),
    CheckConstraint("length(idempotency_key) BETWEEN 1 AND 255", name="idempotency_key_length"),
)
# Partial unique: enforces idempotency and settles concurrent POSTs with the same key.
Index("jobs_idempotency_key", jobs.c.idempotency_key, unique=True,
      postgresql_where=jobs.c.idempotency_key.isnot(None))
# Partial: holds only queued rows, so it stays small as finished jobs pile up.
Index("jobs_claimable", jobs.c.created_at, postgresql_where=jobs.c.status == "queued")
# Lets the reaper find the few running jobs. Deliberately not on
# lease_expires_at: that column changes on every heartbeat.
Index("jobs_running", jobs.c.lease_owner, postgresql_where=jobs.c.status == "running")

results = Table(
    "results", metadata,
    # Primary key on job_id: at most one result per job, whatever retries do.
    Column("job_id", UUID(as_uuid=True),
           ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True),
    Column("attempt", SmallInteger, nullable=False),
    Column("qbt_version", Text, nullable=False),
    # Bit-identity across workers only holds within one NumPy release.
    Column("numpy_version", Text, nullable=False),
    # NULL when n_boot == 0 (no bootstrap ran). Informational only: every
    # implementation gives bit-identical numbers.
    Column("bootstrap_impl", Text),
    Column("duration_ms", Integer, nullable=False),
    Column("dates", ARRAY(Date), nullable=False),
    Column("equity", ARRAY(Double), nullable=False),
    Column("position", ARRAY(Double), nullable=False),  # fractional positions exist
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=NOW),
    CheckConstraint(
        "bootstrap_impl IS NULL OR bootstrap_impl IN ('python', 'numpy', 'cpp')",
        name="bootstrap_impl_known",
    ),
    CheckConstraint("duration_ms >= 0", name="duration_nonnegative"),
    CheckConstraint(
        "cardinality(dates) = cardinality(equity) AND cardinality(dates) = cardinality(position)",
        name="series_lengths_match",
    ),
)

metrics = Table(
    "metrics", metadata,
    Column("job_id", UUID(as_uuid=True),
           ForeignKey("results.job_id", ondelete="CASCADE"), nullable=False),
    Column("name", Text, nullable=False),   # full_report() keys
    Column("value", Double),                # NaN allowed; a metric not computed has no row
    PrimaryKeyConstraint("job_id", "name"),
)


def make_engine(url: str, statement_timeout_seconds: float | None = None) -> Engine:
    """
    The service's engine. pool_pre_ping: survive a database restart.
    statement_timeout_seconds, if given, makes the server cancel any query
    that runs longer.
    """
    connect_args = {}
    if statement_timeout_seconds is not None:
        milliseconds = max(1, round(statement_timeout_seconds * 1000))
        connect_args["options"] = f"-c statement_timeout={milliseconds}"
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)
