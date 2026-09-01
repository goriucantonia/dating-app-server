"""Async database engine wiring.

One engine per process, created at startup and stored on `app.state`. Sessions
and models arrive in Step 3; Step 1 only needs a connection the health check
can prove.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


async def check_connection(engine: AsyncEngine) -> None:
    """Raises if the database is unreachable — the health check's evidence."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
