"""Provider registry (S2-B4): builds provider instances from config,
`name -> instance`. Nothing else may instantiate a provider (§16, D-001).

A provider with a missing API key is still BUILT — the app must boot without
keys (they are owner-supplied and arrive on their own schedule). The missing
key is logged loudly here and becomes a typed AIError at call time.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from app.ai.base import AIError, AIProvider
from app.ai.google import GoogleProvider
from app.ai.openrouter import OpenRouterProvider
from app.ai.resilience import RateLimiter
from app.config import AIConfig
from app.logging_setup import log_event

logger = logging.getLogger("app.ai")

_CONSTRUCTORS: dict[str, Callable[..., AIProvider]] = {
    "google": GoogleProvider,
    "openrouter": OpenRouterProvider,
}


def build_providers(
    config: AIConfig, *, getenv: Callable[[str, str], str] = os.environ.get
) -> dict[str, AIProvider]:
    providers: dict[str, AIProvider] = {}
    for name, pconf in config.providers.items():
        constructor = _CONSTRUCTORS.get(name)
        if constructor is None:
            # Config coherence fails at startup, not mid-date (S2-B5).
            raise AIError(
                f"config names provider '{name}' but no implementation exists; "
                f"known providers: {sorted(_CONSTRUCTORS)}"
            )
        api_key = getenv(pconf.api_key_env, "") or ""
        providers[name] = constructor(api_key, limiter=RateLimiter())
        log_event(
            logger, "provider_built",
            level=logging.WARNING if not api_key else logging.INFO,
            provider=name, api_key_env=pconf.api_key_env, api_key_present=bool(api_key),
        )
    return providers
