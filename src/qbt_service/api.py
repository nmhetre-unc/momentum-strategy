"""
The HTTP API.

    uvicorn --factory qbt_service.api:create_app

A factory rather than a module-level app, so importing this module never
needs DATABASE_URL. At startup the app syncs the strategy catalog to the
database (see qbt_service.catalog) and fails to start if it can't.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from qbt_service.catalog import CATALOG, sync_catalog
from qbt_service.db import make_engine
from qbt_service.settings import Settings


class StrategyInfo(BaseModel):
    name: str
    family: str
    description: str
    params_schema: dict[str, Any] = Field(
        description="JSON Schema of `params` for POST /backtests with this strategy."
    )
    defaults: dict[str, Any] = Field(description="The values used for omitted params.")
    schema_hash: str = Field(
        description="SHA-256 of params_schema; changes whenever the schema does."
    )


class StrategyList(BaseModel):
    strategies: list[StrategyInfo]


def _strategy_list() -> StrategyList:
    return StrategyList(strategies=[
        StrategyInfo(name=e.name, family=e.family, description=e.description,
                     params_schema=e.param_schema, defaults=e.defaults,
                     schema_hash=e.schema_hash.hex())
        for e in CATALOG.values()
    ])


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = Settings.from_env() if settings is None else settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(settings.database_url)
        with engine.begin() as conn:
            app.state.strategy_ids = sync_catalog(conn)
        app.state.engine = engine
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="qbt backtest service", version="0.1.0", lifespan=lifespan)
    strategy_list = _strategy_list()  # fixed for the life of the process

    @app.get("/strategies", response_model=StrategyList, tags=["strategies"])
    def list_strategies() -> StrategyList:
        """The strategies POST /backtests accepts, with their parameter schemas."""
        return strategy_list

    return app
