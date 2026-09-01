"""probe_structured_guard.py (S2-P1, §2)

Proves the Structured Output Guard's give-up path against a REAL model:
force malformed output with a deliberately hostile schema (a `pattern` no
string can ever match), watch the repair attempts, then the typed give-up —
and assert nothing downstream received a silent default.

The schema is forced into the prompt (not native mode) so the hostility lives
entirely in local validation — a provider cannot reject the request shape,
and no model output can ever validate. Three attempts MUST happen, then
StructuredOutputError MUST carry the raw output.

Run inside the api container (the real deployment):

    docker compose exec api python probes/probe_structured_guard.py

Needs GOOGLE_AI_API_KEY in .env (uses the trait_extraction route's model).
"""

from __future__ import annotations

import asyncio
import logging
import sys

from app.ai.base import GenRequest, Message, StructuredOutputError, VersionedSchema
from app.ai.registry import build_providers
from app.ai.routing import TaskRouter
from app.ai.structured import MAX_VALIDATION_ATTEMPTS, guarded_structured_call
from app.config import get_settings, load_ai_config
from app.logging_setup import setup_logging

# Hostile on purpose: the regex (?!) matches nothing, so no output can ever
# satisfy it — validation fails deterministically, whatever the model says.
HOSTILE_SCHEMA = VersionedSchema(
    name="probe_hostile",
    version=1,
    json_schema={
        "type": "object",
        "required": ["verdict"],
        "properties": {"verdict": {"type": "string", "pattern": "(?!x)x"}},
        "additionalProperties": False,
    },
)


class AttemptCounter(logging.Handler):
    """Counts the Guard's malformed/gave_up log lines — the §7 evidence."""

    def __init__(self):
        super().__init__()
        self.outcomes: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        fields = getattr(record, "event_fields", {}) or {}
        if record.getMessage() == "ai_call" and "schema" in fields:
            self.outcomes.append(fields.get("outcome", "?"))


async def main() -> int:
    setup_logging()
    counter = AttemptCounter()
    logging.getLogger("app.ai").addHandler(counter)

    settings = get_settings()
    ai_config = load_ai_config(settings.ai_config_path)
    router = TaskRouter(build_providers(ai_config), ai_config)
    provider, model = router.resolve("trait_extraction")
    print(f"probe: guard give-up via {provider.name}/{model}, "
          f"hostile schema {HOSTILE_SCHEMA.full_name}")

    req = GenRequest(
        task="trait_extraction",  # a real routed task, so the line is realistic
        model=model,
        system_prompt="You are being tested for schema compliance.",
        messages=[Message(role="user", content='Return {"verdict": "ok"}.')],
        temperature=0.0,
        max_tokens=200,
    )

    downstream = "SENTINEL_NEVER_REPLACED"  # a silent default would overwrite this
    try:
        downstream = await guarded_structured_call(
            provider, req, HOSTILE_SCHEMA, force_schema_in_prompt=True
        )
    except StructuredOutputError as exc:
        print(f"observed: StructuredOutputError raised — {exc}")
        print(f"observed: raw output carried ({len(exc.raw_output)} chars): "
              f"{exc.raw_output[:120]!r}")
    else:
        print(f"RED: guard returned {downstream!r} instead of giving up")
        return 1

    malformed = counter.outcomes.count("malformed")
    gave_up = counter.outcomes.count("gave_up")
    print(f"observed: guard outcomes logged = {counter.outcomes}")

    checks = {
        f"{MAX_VALIDATION_ATTEMPTS} malformed attempts logged": malformed == MAX_VALIDATION_ATTEMPTS,
        "exactly one gave_up logged": gave_up == 1,
        "no ok outcome (nothing validated)": "ok" not in counter.outcomes,
        "downstream never received a value": downstream == "SENTINEL_NEVER_REPLACED",
    }
    for label, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {label}")

    if all(checks.values()):
        print("VERDICT: GREEN — repair attempts observed, typed give-up observed, "
              "no silent default reached downstream (§10, §17).")
        return 0
    print("VERDICT: RED — see FAIL lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
