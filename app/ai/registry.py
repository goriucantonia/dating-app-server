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
from app.ai.litellm_provider import LiteLLMProvider
from app.ai.openrouter import OpenRouterProvider
from app.ai.resilience import RateLimiter
from app.config import AIConfig
from app.logging_setup import log_event

logger = logging.getLogger("app.ai")

_CONSTRUCTORS: dict[str, Callable[..., AIProvider]] = {
    "google": GoogleProvider,
    "openrouter": OpenRouterProvider,
    "litellm": LiteLLMProvider,
}

# Providers that accept an `api_base` — a self-hosted or proxied endpoint.
# google and openrouter each talk to exactly one service, so an `api_base` set
# on either is a config mistake that would otherwise be silently ignored.
_ACCEPTS_API_BASE: frozenset[str] = frozenset({"litellm"})

# Providers for which an empty API key is a NORMAL state rather than a warning.
# LiteLLM resolves a key per upstream from the environment (ANTHROPIC_API_KEY,
# OPENAI_API_KEY, …), so an empty LITELLM_API_KEY means "let LiteLLM do it" —
# logging that at WARNING every boot would train people to ignore the line that
# genuinely means google or openrouter cannot call anything.
_KEY_OPTIONAL: frozenset[str] = frozenset({"litellm"})


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
        if pconf.api_base and name not in _ACCEPTS_API_BASE:
            raise AIError(
                f"provider '{name}' is configured with an api_base, which it does "
                f"not read — only {sorted(_ACCEPTS_API_BASE)} does. Left in place it "
                "would look like it was pointing somewhere it is not."
            )
        api_key = getenv(pconf.api_key_env, "") or ""
        kwargs: dict[str, object] = {"limiter": RateLimiter()}
        if name in _ACCEPTS_API_BASE:
            kwargs["api_base"] = pconf.api_base
        providers[name] = constructor(api_key, **kwargs)

        key_missing_matters = not api_key and name not in _KEY_OPTIONAL
        log_event(
            logger, "provider_built",
            level=logging.WARNING if key_missing_matters else logging.INFO,
            provider=name, api_key_env=pconf.api_key_env, api_key_present=bool(api_key),
            api_base=pconf.api_base,
            # Says which of the two empty-key meanings this is, so the line
            # explains itself without reading this file (§7).
            key_resolution="provider_env" if (not api_key and name in _KEY_OPTIONAL) else "configured",
        )
    return providers
