"""Version strategies by behaviour as well as by parameter schema.

A strategy's behaviour can change while its parameter schema doesn't
(ml_direction 2 predicts out of sample only, with the same params as
version 1). strategies gets a `version` column (qbt.strategies.
STRATEGY_VERSIONS), and a row is now identified by (name, version,
schema_hash). Existing rows were all registered by code without
versions and are backfilled as version 1, which is what they ran.

Downgrading is refused once two versions of a strategy share a schema
hash: the old (name, schema_hash) key can't hold both, and jobs point at
each row.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("strategies", sa.Column("version", sa.Integer, nullable=False,
                                          server_default=sa.text("1")))
    op.create_check_constraint(op.f("ck_strategies_version_positive"), "strategies",
                               "version >= 1")
    op.drop_constraint("uq_strategies_name_schema_hash", "strategies", type_="unique")
    op.create_unique_constraint(op.f("uq_strategies_name_version_schema_hash"), "strategies",
                                ["name", "version", "schema_hash"])


def downgrade() -> None:
    shared = op.get_bind().execute(sa.text(
        "SELECT name FROM strategies GROUP BY name, schema_hash HAVING count(*) > 1"
    )).scalars().all()
    if shared:
        raise RuntimeError(
            f"can't downgrade below 0003: strategies {sorted(set(shared))} have several "
            "versions with the same parameter schema, which 0002's (name, schema_hash) key "
            "can't represent"
        )
    op.drop_constraint(op.f("uq_strategies_name_version_schema_hash"), "strategies",
                       type_="unique")
    op.create_unique_constraint(op.f("uq_strategies_name_schema_hash"), "strategies",
                                ["name", "schema_hash"])
    op.drop_column("strategies", "version")  # drops its CHECK constraint too
