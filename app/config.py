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

    # Local-only phase (decision log #11): Flutter web dev serves from a random
    # localhost port, so dev CORS admits any localhost origin. Revisited with
    # the hosting decision, not before it.
    cors_origin_regex: str = r"http://localhost(:\d+)?"

    ai_config_path: Path = SERVER_ROOT / "config" / "ai.yaml"


class ProviderConfig(BaseModel):
    api_key_env: str


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
