"""Record each price snapshot's fetch range; fix 0001's CHECK constraint names.

1. A snapshot covers a backtest's [start, end) when its requested range
   [fetch_start, fetch_end) does. first_date/last_date can't answer that:
   the first bar falls on the first trading day on or after fetch_start.
   Existing rows (none are expected yet) are backfilled with the tightest
   range their bars imply, which only ever makes them cover less.

2. 0001 named its CHECK constraints "ck_<table>_<name>" by hand, and the
   metadata naming convention then prefixed them again, giving e.g.
   "ck_jobs_ck_jobs_running_has_lease". They are renamed to the names
   qbt_service.db produces. New names below go through op.f(), which
   marks a name as final so the convention leaves it alone.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MISNAMED_CHECKS = {
    "price_series": ("date_order", "n_bars_positive", "sha256_length"),
    "jobs": ("attempts_in_range", "finished_iff_terminal", "idempotency_key_length",
             "max_attempts_positive", "request_hash_length", "running_has_lease"),
    "results": ("bootstrap_impl_known", "duration_nonnegative", "series_lengths_match"),
}


def _rename_checks(doubled_to_single: bool) -> None:
    for table, names in MISNAMED_CHECKS.items():
        for name in names:
            single = f"ck_{table}_{name}"
            doubled = f"ck_{table}_{single}"
            old, new = (doubled, single) if doubled_to_single else (single, doubled)
            op.execute(f'ALTER TABLE {table} RENAME CONSTRAINT "{old}" TO "{new}"')


def upgrade() -> None:
    _rename_checks(doubled_to_single=True)

    op.add_column("price_series", sa.Column("fetch_start", sa.Date))
    op.add_column("price_series", sa.Column("fetch_end", sa.Date))
    op.execute("UPDATE price_series SET fetch_start = first_date, fetch_end = last_date + 1")
    op.alter_column("price_series", "fetch_start", nullable=False)
    op.alter_column("price_series", "fetch_end", nullable=False)
    op.create_check_constraint(op.f("ck_price_series_fetch_range_order"), "price_series",
                               "fetch_start < fetch_end")
    op.create_check_constraint(op.f("ck_price_series_bars_within_fetch_range"), "price_series",
                               "first_date >= fetch_start AND last_date < fetch_end")


def downgrade() -> None:
    # Dropping the columns drops the CHECK constraints that use them.
    op.drop_column("price_series", "fetch_end")
    op.drop_column("price_series", "fetch_start")
    _rename_checks(doubled_to_single=False)
