"""probe_ai_smoke.py — Step 2 acceptance-criteria witness helper.

NOT part of the §2 minimum probe set. Exists to witness, on demand:
- AC1: a real generation through BOTH providers, with the §5 log line each;
- AC2: a real embedding call returning a 768-dimension vector from the
  pinned embedding model;
- a happy-path structured call through the Guard (native mode).

The OpenRouter model comes from the routing config if resolved, else from
argv — the `free-model-of-choice` slots are unfilled by design, and this
probe must not be a reason to fill them:

    docker compose exec api python probes/probe_ai_smoke.py [openrouter-model]
"""

from __future__ import annotations

import asyncio
import sys

from app.ai.base import AIError, GenRequest, Message, RouteUnresolvedError, VersionedSchema
from app.ai.registry import build_providers
from app.ai.routing import TaskRouter
from app.config import get_settings, load_ai_config
from app.logging_setup import setup_logging

PING_SCHEMA = VersionedSchema(
    name="probe_ping",
    version=1,
    json_schema={
        "type": "object",
        "required": ["greeting", "number"],
        "properties": {
            "greeting": {"type": "string"},
            "number": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "additionalProperties": False,
    },
)


async def main() -> int:
    setup_logging()
    settings = get_settings()
    ai_config = load_ai_config(settings.ai_config_path)
    router = TaskRouter(build_providers(ai_config), ai_config)
    failures: list[str] = []

    # --- google generation + structured (trait_extraction route) ---
    provider, model = router.resolve("trait_extraction")
    req = GenRequest(
        task="trait_extraction", model=model,
        messages=[Message(role="user", content="Reply with the single word: ready")],
        temperature=0.0, max_tokens=20,
    )
    try:
        result = await provider.generate(req)
        print(f"PASS google generate ({model}): {result.text.strip()[:60]!r}")
    except AIError as exc:
        failures.append(f"google generate: {exc}")
        print(f"FAIL google generate: {exc}")

    try:
        data = await provider.generate_structured(
            GenRequest(
                task="trait_extraction", model=model,
                messages=[Message(role="user", content="Greet me and pick a number 1-10.")],
                temperature=0.0, max_tokens=100,
            ),
            PING_SCHEMA,
        )
        print(f"PASS google structured (native mode): {data}")
    except AIError as exc:
        failures.append(f"google structured: {exc}")
        print(f"FAIL google structured: {exc}")

    # --- pinned embeddings (AC2: 768 dimensions) ---
    emb_provider, emb_model = router.resolve_embeddings()
    try:
        vectors = await emb_provider.embed(["a probe sentence"], emb_model)
        dim = len(vectors[0])
        verdict = "PASS" if dim == 768 else "FAIL"
        print(f"{verdict} embedding ({emb_provider.name}/{emb_model}): dimension {dim}")
        if dim != 768:
            failures.append(f"embedding dimension {dim} != 768")
    except AIError as exc:
        failures.append(f"embedding: {exc}")
        print(f"FAIL embedding: {exc}")

    # --- openrouter generation (provisional model from config or argv) ---
    or_model = sys.argv[1] if len(sys.argv) > 1 else None
    if or_model is None:
        try:
            _, or_model = router.resolve("date_simulation")
        except RouteUnresolvedError:
            pass
    if or_model is None:
        print("SKIP openrouter: no model — slots unfilled by design; pass one as argv[1]")
        failures.append("openrouter path not exercised (no model given)")
    else:
        or_provider = router._providers["openrouter"]  # probe-only reach-in
        try:
            result = await or_provider.generate(
                GenRequest(
                    task="date_simulation", model=or_model,
                    messages=[Message(role="user", content="Reply with the single word: ready")],
                    temperature=0.0, max_tokens=20,
                )
            )
            print(f"PASS openrouter generate ({or_model}): {result.text.strip()[:60]!r}")
        except AIError as exc:
            failures.append(f"openrouter generate: {exc}")
            print(f"FAIL openrouter generate: {exc}")

    if failures:
        print(f"VERDICT: RED — {len(failures)} failure(s): {failures}")
        return 1
    print("VERDICT: GREEN — both providers answered, embedding is 768-dim, "
          "Guard happy path validated. Check the ai_call log lines above each.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
