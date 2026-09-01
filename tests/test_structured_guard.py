"""Unit tests for the Structured Output Guard and TaskRouter mechanics.

These prove the loop logic with a fake provider — no network. The live-model
witness is probe_structured_guard.py (§1/§2: a passing test is evidence about
the test; the probe is the observation).
"""

from __future__ import annotations

import pytest

from app.ai.base import (
    AIError,
    GenRequest,
    Message,
    RouteUnresolvedError,
    StructuredOutputError,
    VersionedSchema,
)
from app.ai.resilience import RateLimiter
from app.ai.routing import TaskRouter
from app.ai.structured import guarded_structured_call
from app.config import AIConfig

SCHEMA = VersionedSchema(
    name="test",
    version=1,
    json_schema={
        "type": "object",
        "required": ["n"],
        "properties": {"n": {"type": "integer"}},
        "additionalProperties": False,
    },
)


class FakeProvider:
    """Feeds scripted raw outputs to the Guard and records what it was asked."""

    name = "fake"
    supports_native_structured = True

    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.limiter = RateLimiter(min_interval_s=0.0)
        self.seen_requests: list[GenRequest] = []

    async def raw_structured(self, req, schema, *, schema_in_prompt):
        self.seen_requests.append(req)
        return self.outputs.pop(0)


def _req() -> GenRequest:
    return GenRequest(
        task="trait_extraction", model="fake-model",
        messages=[Message(role="user", content="go")],
    )


@pytest.mark.asyncio
async def test_valid_first_try_returns_dict():
    provider = FakeProvider(['{"n": 4}'])
    assert await guarded_structured_call(provider, _req(), SCHEMA) == {"n": 4}


@pytest.mark.asyncio
async def test_markdown_fences_are_stripped():
    provider = FakeProvider(['```json\n{"n": 7}\n```'])
    assert await guarded_structured_call(provider, _req(), SCHEMA) == {"n": 7}


@pytest.mark.asyncio
async def test_repair_prompt_carries_validation_error_then_succeeds():
    provider = FakeProvider(['{"n": "not an int"}', '{"n": 2}'])
    assert await guarded_structured_call(provider, _req(), SCHEMA) == {"n": 2}
    # The second attempt saw the raw failure and a repair instruction (§19:
    # validated first, repaired second).
    second = provider.seen_requests[1]
    roles = [m.role for m in second.messages]
    assert roles == ["user", "assistant", "user"]
    assert "not an int" in second.messages[1].content
    assert "failed validation" in second.messages[2].content


@pytest.mark.asyncio
async def test_three_failures_raise_typed_giveup_with_raw_output():
    provider = FakeProvider(["junk one", "junk two", "junk three"])
    with pytest.raises(StructuredOutputError) as exc_info:
        await guarded_structured_call(provider, _req(), SCHEMA)
    # Give-up after exactly 3 attempts (§17), raw output carried (§10) —
    # nothing was returned, so nothing downstream can hold a silent default.
    assert len(provider.seen_requests) == 3
    assert exc_info.value.raw_output == "junk three"
    assert exc_info.value.task == "trait_extraction"


def _config(routing_overrides: dict | None = None) -> AIConfig:
    routing = {
        task: {"provider": "openrouter", "model": "free-model-of-choice"}
        for task in (
            "dispute_followups", "trait_extraction", "persona_digest",
            "scenario_generation", "date_simulation", "judging",
            "chat_reply", "chat_compaction",
        )
    }
    routing.update(routing_overrides or {})
    return AIConfig.model_validate({
        "providers": {"openrouter": {"api_key_env": "X"}},
        "embeddings": {"provider": "openrouter", "model": "emb"},
        "routing": routing,
    })


def test_router_rejects_unknown_provider_at_startup():
    config = _config({"judging": {"provider": "nonexistent", "model": "m"}})
    with pytest.raises(AIError, match="nonexistent"):
        TaskRouter({"openrouter": FakeProvider([])}, config)


def test_router_rejects_missing_task_at_startup():
    config = _config()
    del config.routing["chat_reply"]
    with pytest.raises(AIError, match="chat_reply"):
        TaskRouter({"openrouter": FakeProvider([])}, config)


def test_unfilled_slot_resolves_to_typed_error_never_a_guess():
    router = TaskRouter({"openrouter": FakeProvider([])}, _config())
    with pytest.raises(RouteUnresolvedError, match="deliberately unfilled"):
        router.resolve("date_simulation")


def test_filled_slot_resolves_to_provider_and_model():
    provider = FakeProvider([])
    router = TaskRouter(
        {"openrouter": provider},
        _config({"judging": {"provider": "openrouter", "model": "some-model"}}),
    )
    resolved_provider, model = router.resolve("judging")
    assert resolved_provider is provider
    assert model == "some-model"
