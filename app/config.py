"""Settings loading (S1-B5).

Two layers, kept separate on purpose:

- `Settings` — environment-driven runtime settings (.env / container env).
- `AIConfig` — the `ai:` block of `ai_interaction.md` §3, loaded verbatim from
  `config/ai.yaml`. The `free-model-of-choice` slots are DELIBERATELY unfilled
  (owner decision, 2026-09-01); `unresolved_routes()` names them so startup can
  log loudly which routes cannot run yet. Do not fill them to make something
  work — two gates stand before the first real analysis.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# The placeholder string used in config/ai.yaml for routes the owner has not
# chosen a model for yet. A route carrying it resolves to "unresolved", never
# to a real call.
UNFILLED_MODEL = "free-model-of-choice"

SERVER_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings read from the environment (see .env.example at the superproject root)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    jwt_secret: str
    google_ai_api_key: str = ""
    openrouter_api_key: str = ""
    # Optional by design: LiteLLM normally resolves a key per upstream from the
    # environment (ANTHROPIC_API_KEY, OPENAI_API_KEY, …). Set this only to force
    # ONE key across every litellm route — a proxy, or a single upstream.
    litellm_api_key: str = ""

    # Local-only phase (decision log #11): Flutter web dev serves from a random
    # localhost port, so dev CORS admits any localhost origin. Both spellings —
    # the flutter dev server binds IPv6-only when given the name `localhost`,
    # which breaks the dwds debug socket on Windows, so it is run on 127.0.0.1
    # and the browser origin is numeric. Revisited with the hosting decision,
    # not before it.
    cors_origin_regex: str = r"http://(localhost|127\.0\.0\.1)(:\d+)?"

    ai_config_path: Path = SERVER_ROOT / "config" / "ai.yaml"


class ProviderConfig(BaseModel):
    api_key_env: str
    # Only the `litellm` provider reads this, and the registry refuses it on any
    # other (config coherence fails at startup, not mid-date — S2-B5). It is what
    # points LiteLLM at a self-hosted endpoint: a LiteLLM proxy, Ollama, vLLM, or
    # an OpenAI-compatible gateway.
    api_base: str | None = None


class ModelRoute(BaseModel):
    provider: str
    model: str

    @property
    def resolved(self) -> bool:
        return self.model != UNFILLED_MODEL


class AIConfig(BaseModel):
    """Mirror of the `ai:` block in `ai_interaction.md` §3 (locked shape)."""

    providers: dict[str, ProviderConfig]
    embeddings: ModelRoute
    routing: dict[str, ModelRoute]

    def unresolved_routes(self) -> list[str]:
        return [task for task, route in self.routing.items() if not route.resolved]


def load_ai_config(path: Path) -> AIConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AIConfig.model_validate(raw["ai"])


@lru_cache
def get_settings() -> Settings:
    return Settings()
