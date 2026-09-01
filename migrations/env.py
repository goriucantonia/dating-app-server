"""Alembic environment — wired to the async engine (S1-B4).

The URL comes from DATABASE_URL (the same variable the app reads), so migrations
always run against the database the app will use.
"""

from __future__ import annotations

import asyncio
import os

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

config = context.config
config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])

# Imported after the config lines above by design: alembic must set the URL
# before app.models pulls the engine machinery in. The suppression comment that
# used to sit here named E402, which this project does not enable — so the
# suppression was itself the only lint error in the repo (RUF100). Removed
# 2026-09-01; the reason it documented is kept here, where it belongs.
from app.models import Base

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=NullPool,
    )
    async with engine.connect() as connection:
        await connection.run_sync(_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
