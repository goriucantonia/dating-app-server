"""Management entry point for the reconciliation pass (S3-B6).

The same reconcile() that runs at startup, invocable on demand:

    docker compose exec api python scripts/run_reconcile.py
"""

from __future__ import annotations

import asyncio

from app.config import get_settings
from app.db import create_engine
from app.logging_setup import setup_logging
from app.reconcile import reconcile


async def main() -> None:
    setup_logging()
    engine = create_engine(get_settings().database_url)
    counts = await reconcile(engine)
    await engine.dispose()
    print(f"reconcile: {counts}")


if __name__ == "__main__":
    asyncio.run(main())
