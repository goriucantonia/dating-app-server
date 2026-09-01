"""Management entry point for the reconciliation pass (S3-B6, S15-B6).

The same four-step pass that runs at startup, invocable on demand:

    docker compose exec api python scripts/run_reconcile.py [--wait]

`--wait` runs the demo pipeline's AI half (extraction, compilation,
embedding — real model calls) inline and reports it; without it the script
returns as soon as the inline steps are done and the AI half runs in the
background, the way boot does.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from app.ai.registry import build_providers
from app.ai.routing import TaskRouter
from app.config import get_settings, load_ai_config
from app.db import create_engine
from app.logging_setup import setup_logging
from app.reconcile import run_full_pass


async def main(wait: bool) -> None:
    setup_logging()
    settings = get_settings()
    ai_config = load_ai_config(settings.ai_config_path)
    engine = create_engine(settings.database_url)
    router = TaskRouter(build_providers(ai_config), ai_config)
    app = SimpleNamespace(
        state=SimpleNamespace(engine=engine, ai_router=router, ai_config=ai_config)
    )
    # `--wait` awaits the demo pipeline inline (the probe's mode); without it
    # the AI half runs in the background exactly as it does at boot.
    results = await run_full_pass(app, inline_demo_pipeline=wait)
    print(f"reconcile: {results}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main(wait="--wait" in sys.argv[1:]))
