"""The Structured Output Guard — the single choke point (S2-B6, §16).

No module anywhere else parses model JSON. The locked order (§19 — validate
BEFORE any repair prompt):

    1. native mode first (provider's json_schema support), falling back to
       embedding the schema in the prompt where the model has none (§4.1);
    2. validate the raw output against the versioned JSON schema;
    3. on failure, ONE repair prompt carrying the validation error;
    4. max 3 attempts total, then raise StructuredOutputError with the raw
       output. Never a silent default (§10).

Free OpenRouter models are why this exists: they hold JSON contracts less
reliably than Gemini, and the Guard is what makes them safely swappable anyway.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

import jsonschema

from app.ai.base import (
    GenRequest,
    Message,
    NativeStructuredUnsupported,
    StructuredOutputError,
    VersionedSchema,
)
from app.ai.resilience import RateLimiter, execute
from app.logging_setup import log_event

logger = logging.getLogger("app.ai")

# Give-up after 3 validation attempts (§17), committed alongside the loop.
MAX_VALIDATION_ATTEMPTS = 3


class RawStructuredProvider(Protocol):
    """What the Guard needs from a provider: one raw structured attempt.
    Only the Guard calls this; feature modules use AIProvider.generate_structured."""

    name: str
    limiter: RateLimiter
    supports_native_structured: bool

    async def raw_structured(
        self, req: GenRequest, schema: VersionedSchema, *, schema_in_prompt: bool
    ) -> str: ...


def schema_prompt_block(schema: VersionedSchema) -> str:
    """The fallback for models without native json_schema mode (§4.1)."""
    return (
        "\n\nRespond with ONLY a JSON object — no prose, no markdown fences. "
        f"It must validate against this JSON Schema ({schema.full_name}):\n"
        f"{json.dumps(schema.json_schema)}"
    )


def _strip_fences(raw: str) -> str:
    """Models wrap JSON in ```json fences despite instructions; stripping a
    fence is transport cleanup, not validation leniency."""
    text = raw.strip()
    if text.startswith("```"):
        # ```json\n{...}\n```  and  ```json{...}```  and  ```{...}```
        text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text, count=1)
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _validation_failure(raw: str, schema: VersionedSchema) -> str | None:
    """Returns the failure description, or None when the output validates."""
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        return f"not valid JSON: {exc}"
    try:
        jsonschema.validate(data, schema.json_schema)
    except jsonschema.ValidationError as exc:
        return f"JSON does not match schema {schema.full_name}: {exc.message}"
    return None


def _repair_messages(raw: str, failure: str) -> list[Message]:
    return [
        Message(role="assistant", content=raw),
        Message(
            role="user",
            content=(
                "Your previous response failed validation: "
                f"{failure}. Reply again with ONLY a corrected JSON object "
                "that satisfies the schema. No explanation, no markdown."
            ),
        ),
    ]


async def guarded_structured_call(
    provider: RawStructuredProvider,
    req: GenRequest,
    schema: VersionedSchema,
    *,
    force_schema_in_prompt: bool = False,
) -> dict:
    """The one road for structured output. Returns the validated dict or
    raises StructuredOutputError carrying the raw output."""
    schema_in_prompt = force_schema_in_prompt or not provider.supports_native_structured
    messages = list(req.messages)
    raw = ""

    for attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        attempt_req = GenRequest(
            task=req.task, model=req.model, system_prompt=req.system_prompt,
            messages=messages, temperature=req.temperature, max_tokens=req.max_tokens,
        )

        async def _call(r: GenRequest = attempt_req, in_prompt: bool = schema_in_prompt) -> str:
            return await provider.raw_structured(r, schema, schema_in_prompt=in_prompt)

        try:
            raw = await execute(
                _call, task=req.task, provider=provider.name, model=req.model,
                limiter=provider.limiter,
            )
        except NativeStructuredUnsupported:
            # This model has no native mode after all — same attempt, prompt fallback.
            schema_in_prompt = True
            log_event(
                logger, "structured_native_fallback",
                task=req.task, provider=provider.name, model=req.model,
            )
            raw = await execute(
                lambda r=attempt_req: provider.raw_structured(
                    r, schema, schema_in_prompt=True
                ),
                task=req.task, provider=provider.name, model=req.model,
                limiter=provider.limiter,
            )

        # §19: validate BEFORE any repair prompt.
        failure = _validation_failure(raw, schema)
        if failure is None:
            log_event(
                logger, "ai_call", task=req.task, provider=provider.name,
                model=req.model, attempt=attempt, outcome="ok",
                schema=schema.full_name, mode="prompt" if schema_in_prompt else "native",
            )
            return json.loads(_strip_fences(raw))

        log_event(
            logger, "ai_call", level=logging.WARNING,
            task=req.task, provider=provider.name, model=req.model,
            attempt=attempt, outcome="malformed", schema=schema.full_name,
            validation_error=failure,
        )
        if attempt < MAX_VALIDATION_ATTEMPTS:
            messages = messages + _repair_messages(raw, failure)

    # The typed give-up (§17): raw output attached, logged, never a default.
    log_event(
        logger, "ai_call", level=logging.ERROR,
        task=req.task, provider=provider.name, model=req.model,
        attempt=MAX_VALIDATION_ATTEMPTS, outcome="gave_up", schema=schema.full_name,
        raw_output=raw[:2000],
    )
    raise StructuredOutputError(
        f"structured output failed validation after {MAX_VALIDATION_ATTEMPTS} attempts "
        f"for schema {schema.full_name}",
        raw_output=raw, task=req.task, provider=provider.name, model=req.model,
    )
