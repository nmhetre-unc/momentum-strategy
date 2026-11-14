"""Initial schema: price snapshots, strategy catalog, job queue, results, metrics.

Written by hand to match qbt_service.db (see that module for the reasoning
behind each table and index). `alembic check` in CI keeps the two in step
for tables, columns, types, foreign keys and indexes; the tests cover what
it can't see (CHECK constraints, partial-index predicates, storage options).

Revision ID: 0001
Revises:
Create Date: 2026-11-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NOW = sa.text("now()")
JOB_STATUS = pg.ENUM("queued", "running", "succeeded", "failed", name="job_status",
                     create_type=False)


def upgrade() -> None:
    JOB_STATUS.create(op.get_bind())

    op.create_table(
        "price_series",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), nullable=False),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("source", sa.Text, nullable=False, server_default=sa.text("'yfinance'")),
        sa.Column("adjusted", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("first_date", sa.Date, nullable=False),
        sa.Column("last_date", sa.Date, nullable=False),
        sa.Column("n_bars", sa.Integer, nullable=False),
        sa.Column("content_sha256", sa.LargeBinary, nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_price_series"),
        sa.CheckConstraint("n_bars > 0", name="ck_price_series_n_bars_positive"),
        sa.CheckConstraint("first_date <= last_date", name="ck_price_series_date_order"),
        sa.CheckConstraint("octet_length(content_sha256) = 32",
                           name="ck_price_series_sha256_length"),
        sa.UniqueConstraint("ticker", "source", "content_sha256",
                            name="uq_price_series_ticker_source_content_sha256"),
    )
    op.create_index("price_series_latest", "price_series",
                    ["ticker", sa.text("fetched_at DESC")])

    op.create_table(
        "price_bars",
        sa.Column("series_id", sa.BigInteger, nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("open", sa.Double),
        sa.Column("high", sa.Double),
        sa.Column("low", sa.Double),
        sa.Column("close", sa.Double, nullable=False),
        sa.Column("volume", sa.BigInteger),
        sa.PrimaryKeyConstraint("series_id", "date", name="pk_price_bars"),
        sa.ForeignKeyConstraint(["series_id"], ["price_series.id"], ondelete="RESTRICT",
                                name="fk_price_bars_series_id_price_series"),
    )

    op.create_table(
        "strategies",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("schema_hash", sa.LargeBinary, nullable=False),
        sa.Column("param_schema", pg.JSONB, nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("is_current", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_strategies"),
        sa.UniqueConstraint("name", "schema_hash", name="uq_strategies_name_schema_hash"),
    )
    op.create_index("strategies_one_current", "strategies", ["name"], unique=True,
                    postgresql_where=sa.text("is_current"))

    op.create_table(
        "jobs",
        sa.Column("id", pg.UUID(as_uuid=True), nullable=False,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("strategy_id", sa.BigInteger, nullable=False),
        sa.Column("price_series_id", sa.BigInteger, nullable=False),
        sa.Column("request", pg.JSONB, nullable=False),
        sa.Column("request_hash", sa.LargeBinary, nullable=False),
        sa.Column("idempotency_key", sa.Text),
        sa.Column("status", JOB_STATUS, nullable=False, server_default=sa.text("'queued'")),
        sa.Column("attempts", sa.SmallInteger, nullable=False, server_default=sa.text("0")),
        sa.Column("max_attempts", sa.SmallInteger, nullable=False, server_default=sa.text("3")),
        sa.Column("run_after", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("lease_token", pg.UUID(as_uuid=True)),
        sa.Column("lease_owner", sa.Text),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("last_error", sa.Text),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
        sa.ForeignKeyConstraint(["strategy_id"], ["strategies.id"], ondelete="RESTRICT",
                                name="fk_jobs_strategy_id_strategies"),
        sa.ForeignKeyConstraint(["price_series_id"], ["price_series.id"], ondelete="RESTRICT",
                                name="fk_jobs_price_series_id_price_series"),
        sa.CheckConstraint(
            "(status = 'running') = (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="ck_jobs_running_has_lease",
        ),
        sa.CheckConstraint(
            "(status IN ('succeeded', 'failed')) = (finished_at IS NOT NULL)",
            name="ck_jobs_finished_iff_terminal",
        ),
        sa.CheckConstraint("attempts >= 0 AND attempts <= max_attempts",
                           name="ck_jobs_attempts_in_range"),
        sa.CheckConstraint("max_attempts >= 1", name="ck_jobs_max_attempts_positive"),
        sa.CheckConstraint("octet_length(request_hash) = 32", name="ck_jobs_request_hash_length"),
        sa.CheckConstraint("length(idempotency_key) BETWEEN 1 AND 255",
                           name="ck_jobs_idempotency_key_length"),
    )
    # Room on each page for heartbeat updates to stay in-page (HOT), and
    # vacuum the queue table early: every heartbeat leaves a dead tuple.
    op.execute("ALTER TABLE jobs SET (fillfactor = 90, autovacuum_vacuum_scale_factor = 0.02)")
    op.create_index("jobs_idempotency_key", "jobs", ["idempotency_key"], unique=True,
                    postgresql_where=sa.text("idempotency_key IS NOT NULL"))
    op.create_index("jobs_claimable", "jobs", ["created_at"],
                    postgresql_where=sa.text("status = 'queued'"))
    op.create_index("jobs_running", "jobs", ["lease_owner"],
                    postgresql_where=sa.text("status = 'running'"))

    op.create_table(
        "results",
        sa.Column("job_id", pg.UUID(as_uuid=True), nullable=False),
        sa.Column("attempt", sa.SmallInteger, nullable=False),
        sa.Column("qbt_version", sa.Text, nullable=False),
        sa.Column("numpy_version", sa.Text, nullable=False),
        sa.Column("bootstrap_impl", sa.Text),
        sa.Column("duration_ms", sa.Integer, nullable=False),
        sa.Column("dates", pg.ARRAY(sa.Date), nullable=False),
        sa.Column("equity", pg.ARRAY(sa.Double), nullable=False),
        sa.Column("position", pg.ARRAY(sa.Double), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("job_id", name="pk_results"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE",
                                name="fk_results_job_id_jobs"),
        sa.CheckConstraint(
            "bootstrap_impl IS NULL OR bootstrap_impl IN ('python', 'numpy', 'cpp')",
            name="ck_results_bootstrap_impl_known",
        ),
        sa.CheckConstraint("duration_ms >= 0", name="ck_results_duration_nonnegative"),
        sa.CheckConstraint(
            "cardinality(dates) = cardinality(equity) "
            "AND cardinality(dates) = cardinality(position)",
            name="ck_results_series_lengths_match",
        ),
    )

    op.create_table(
        "metrics",
        sa.Column("job_id", pg.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("value", sa.Double),
        sa.PrimaryKeyConstraint("job_id", "name", name="pk_metrics"),
        sa.ForeignKeyConstraint(["job_id"], ["results.job_id"], ondelete="CASCADE",
                                name="fk_metrics_job_id_results"),
    )


def downgrade() -> None:
    for table in ("metrics", "results", "jobs", "strategies", "price_bars", "price_series"):
        op.drop_table(table)  # drops the table's indexes with it
    JOB_STATUS.drop(op.get_bind())
