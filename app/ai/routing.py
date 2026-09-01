"""TaskRouter (S2-B5): `task -> (provider, model)` from config.

Fails at startup, not mid-date: every routed provider must exist and every
routed task name must be one of the eight known tasks. A task whose model slot
is still the `free-model-of-choice` placeholder resolves to a typed
RouteUnresolvedError — loudly, never a guessed model (the slots are unfilled
by owner decision, 2026-09-01).
"""

from __future__ import annotations

import logging

from app.ai.base import ROUTED_TASKS, AIError, AIProvider, RouteUnresolvedError
from app.config import AIConfig
from app.logging_setup import log_event

logger = logging.getLogger("app.ai")


class TaskRouter:
    def __init__(self, providers: dict[str, AIProvider], config: AIConfig):
        self._providers = providers
        self._config = config

        unknown_tasks = sorted(set(config.routing) - set(ROUTED_TASKS))
        if unknown_tasks:
            raise AIError(
                f"routing config names unknown tasks {unknown_tasks}; "
                f"the eight routed tasks are {sorted(ROUTED_TASKS)}"
            )
        missing_tasks = sorted(set(ROUTED_TASKS) - set(config.routing))
        if missing_tasks:
            raise AIError(f"routing config is missing tasks {missing_tasks}")
        for task, route in config.routing.items():
            if route.provider not in providers:
                raise AIError(
                    f"task '{task}' routes to provider '{route.provider}' "
                    f"which is not configured; have {sorted(providers)}"
                )
        if config.embeddings.provider not in providers:
            raise AIError(
                f"embeddings pin provider '{config.embeddings.provider}' "
                f"which is not configured; have {sorted(providers)}"
            )

        log_event(
            logger, "router_ready",
            routes={t: f"{r.provider}/{r.model}" for t, r in config.routing.items()},
            embeddings=f"{config.embeddings.provider}/{config.embeddings.model}",
            unresolved=config.unresolved_routes(),
        )

    def resolve(self, task: str) -> tuple[AIProvider, str]:
        route = self._config.routing[task]
        if not route.resolved:
            raise RouteUnresolvedError(
                f"task '{task}' has no model chosen yet (slot is "
                "'free-model-of-choice', deliberately unfilled — owner decision "
                "2026-09-01, filled after the gates)",
                task=task, provider=route.provider,
            )
        return self._providers[route.provider], route.model

    def resolve_embeddings(self) -> tuple[AIProvider, str]:
        pin = self._config.embeddings
        return self._providers[pin.provider], pin.model
