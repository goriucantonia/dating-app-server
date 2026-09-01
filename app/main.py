"""FastAPI application object (S1-B1).

`GET /health` checks its own database connection — liveness alone is not
health. CORS is on for the Flutter-web origin (communication_protocol.md §2);
mobile/desktop don't need it but aren't harmed by it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.ai.registry import build_providers
from app.ai.routing import TaskRouter
from app.config import get_settings, load_ai_config
from app.db import check_connection, create_engine
from app.errors import ApiError, register_error_handlers
from app.logging_setup import log_event, setup_logging
from app.reconcile import run_full_pass
from app.routers import analyses as analyses_router
from app.routers import auth as auth_router
from app.routers import chat as chat_router
from app.routers import me as me_router
from app.routers import persona as persona_router
from app.routers import questions as questions_router
from app.routers import simulation as simulation_router
from app.routers import traits as traits_router

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    ai_config = load_ai_config(settings.ai_config_path)

    # An unfilled route must never be a surprise (S1-B5): name each one, loudly,
    # on every boot. They are filled after the gates, not before.
    unresolved = ai_config.unresolved_routes()
    if unresolved:
        log_event(
            logger, "ai_routes_unresolved", level=logging.WARNING,
            tasks=unresolved,
            note="model slots deliberately unfilled (owner decision 2026-09-01); "
                 "chosen after the quota-fit and fidelity gates",
        )
    log_event(
        logger, "startup",
        resolved_routes={t: f"{r.provider}/{r.model}" for t, r in ai_config.routing.items() if r.resolved},
        embeddings=f"{ai_config.embeddings.provider}/{ai_config.embeddings.model}",
    )

    app.state.settings = settings
    app.state.ai_config = ai_config
    # Fails at startup on incoherent routing config, not mid-date (S2-B5).
    app.state.ai_router = TaskRouter(build_providers(ai_config), ai_config)
    app.state.engine = create_engine(settings.database_url)
    # The four-step reconciliation pass (§12 — no trust bypasses): questions,
    # demo profiles, stuck analyses, embedding-model consistency. Step 2's
    # AI half runs in the background and reports through its own log lines.
    await run_full_pass(app)
    yield
    await app.state.engine.dispose()


app = FastAPI(title="Dating App AI — API", lifespan=lifespan)

register_error_handlers(app)

app.include_router(auth_router.router)
app.include_router(me_router.router)
app.include_router(questions_router.router)
app.include_router(traits_router.router)
app.include_router(persona_router.router)
app.include_router(analyses_router.router)
app.include_router(simulation_router.router)
app.include_router(chat_router.router)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=get_settings().cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(request: Request) -> dict:
    try:
        await check_connection(request.app.state.engine)
    except Exception as exc:  # noqa: BLE001 — any DB failure means unhealthy, by design
        log_event(logger, "health_check", level=logging.ERROR, outcome="db_unreachable", error=str(exc))
        raise ApiError(503, "database_unavailable", "The server can't reach its database right now.")
    log_event(logger, "health_check", outcome="ok")
    return {"status": "ok", "database": "connected"}
