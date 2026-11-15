"""
The strategy catalog: what GET /strategies serves, and what the
`strategies` table records.

The code is the source of truth. sync_catalog() writes the current
parameter schemas to the table, adding a new row whenever a strategy's
schema changes, so every job keeps a foreign key to the exact schema its
request was validated against. Strategies that are no longer served
keep their rows (old jobs point at them) but stop being current.

The API runs sync_catalog() at startup. Several replicas starting at
once are serialized by a transaction-level advisory lock.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Connection, func, select, update
from sqlalchemy.dialects.postgresql import insert

from qbt.strategies import STRATEGY_DOCS
from qbt_service.db import strategies
from qbt_service.models import PARAMS_MODELS, StrategyParams

# An arbitrary constant identifying "catalog sync" among advisory locks.
SYNC_LOCK_KEY = 0x71B7_CA7A


@dataclass(frozen=True)
class StrategyEntry:
    name: str
    family: str
    description: str
    params_model: type[StrategyParams]
    param_schema: dict[str, Any]
    schema_hash: bytes

    @property
    def defaults(self) -> dict[str, Any]:
        return self.params_model().model_dump()


def _schema_hash(schema: dict[str, Any]) -> bytes:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).digest()


def build_catalog(
    models: dict[str, type[StrategyParams]] | None = None,
) -> dict[str, StrategyEntry]:
    models = PARAMS_MODELS if models is None else models
    catalog = {}
    for name, model in sorted(models.items()):
        schema = model.model_json_schema()
        docs = STRATEGY_DOCS.get(name, {})
        catalog[name] = StrategyEntry(
            name=name, family=docs.get("family", ""), description=docs.get("what", ""),
            params_model=model, param_schema=schema, schema_hash=_schema_hash(schema),
        )
    return catalog


CATALOG = build_catalog()


def sync_catalog(conn: Connection,
                 catalog: dict[str, StrategyEntry] | None = None) -> dict[str, int]:
    """
    Makes the table's current rows match `catalog` and returns
    {name: strategies.id} for them. Runs in the caller's transaction.
    """
    catalog = CATALOG if catalog is None else catalog
    conn.execute(select(func.pg_advisory_xact_lock(SYNC_LOCK_KEY)))

    ids = {}
    for entry in catalog.values():
        conn.execute(
            insert(strategies)
            .values(name=entry.name, schema_hash=entry.schema_hash,
                    param_schema=entry.param_schema, description=entry.description,
                    is_current=False)
            .on_conflict_do_nothing(index_elements=["name", "schema_hash"])
        )
        # Unique indexes are checked row by row, not at the end of a
        # statement, so clear the old current row before setting the new one.
        conn.execute(
            update(strategies)
            .where(strategies.c.name == entry.name, strategies.c.is_current,
                   strategies.c.schema_hash != entry.schema_hash)
            .values(is_current=False)
        )
        ids[entry.name] = conn.execute(
            update(strategies)
            .where(strategies.c.name == entry.name,
                   strategies.c.schema_hash == entry.schema_hash)
            .values(is_current=True, description=entry.description)
            .returning(strategies.c.id)
        ).scalar_one()

    conn.execute(
        update(strategies)
        .where(strategies.c.is_current, strategies.c.name.not_in(list(catalog)))
        .values(is_current=False)
    )
    return ids
