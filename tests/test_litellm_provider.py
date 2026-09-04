"""LiteLLMProvider unit tests — the rules, without the wire.

None of these make a network call or need `litellm` installed: the provider
imports it lazily through `_litellm()`, so a fake module injected into that
cache exercises every path. What is pinned here is the part that is ours —
error classification, the native/prompt decision, the message shape, and the
embedding width refusal — not litellm's behaviour, which is its own project's
to test.
"""

from __future__ import annotations

import pytest

from app.ai import litellm_provider as lp
from app.ai.base import (
    EMBEDDING_DIMENSIONS,
    AIError,
    GenRequest,
    Message,
    NativeStructuredUnsupported,
    RateLimitedError,
    RefusedError,
    TransientAIError,
    VersionedSchema,
)
from app.ai.registry import build_providers
from app.config import AIConfig

SCHEMA = VersionedSchema(
    name="thing",
    version=1,
    json_schema={
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    },
)


# --- a fake litellm ------------------------------------------------------


class _Exceptions:
    """LiteLLM mirrors OpenAI's hierarchy; the subclassing is what the
    specific-before-general ordering in _map_error has to survive."""

    class APIError(Exception):
        pass

    class APIConnectionError(APIError):
        pass

    class Timeout(APIError):
        pass

    class RateLimitError(APIError):
        def __init__(self, message, retry_after=None):
            super().__init__(message)
            self.retry_after = retry_after

    class AuthenticationError(APIError):
        pass

    class PermissionDeniedError(APIError):
        pass

    class NotFoundError(APIError):
        pass

    class ServiceUnavailableError(APIError):
        pass

    class InternalServerError(APIError):
        pass

    class BadRequestError(APIError):
        pass

    class ContentPolicyViolationError(BadRequestError):
        pass

    class ContextWindowExceededError(BadRequestError):
        pass

    class UnsupportedParamsError(BadRequestError):
        pass


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_Choice(content, finish_reason)]


class FakeLiteLLM:
    """Records the kwargs it was called with; returns or raises what it is told."""

    exceptions = _Exceptions
    suppress_debug_info = False

    def __init__(self, *, reply="{}", raises=None, supports_schema=True, embedding=None):
        self.reply = reply
        self.raises = raises
        self.supports_schema = supports_schema
        self.embedding = embedding
        self.completion_calls: list[dict] = []
        self.embedding_calls: list[dict] = []

    async def acompletion(self, **kwargs):
        self.completion_calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return _Response(self.reply)

    async def aembedding(self, **kwargs):
        self.embedding_calls.append(kwargs)
        if callable(self.embedding):
            return self.embedding(**kwargs)
        return self.embedding

    def supports_response_schema(self, model):
        if isinstance(self.supports_schema, Exception):
            raise self.supports_schema
        return self.supports_schema


@pytest.fixture
def fake(monkeypatch):
    """Install a fake litellm into the lazy-import cache."""
    module = FakeLiteLLM()
    monkeypatch.setattr(lp, "_litellm_module", module)
    return module


def provider(fake_module=None, **kw):
    p = lp.LiteLLMProvider(kw.pop("api_key", "k"), **kw)
    # The limiter's default 1s spacing would make the suite sleep.
    p.limiter = lp.RateLimiter(min_interval_s=0.0)
    return p


def req(**kw):
    base = {"task": "judging", "model": "anthropic/claude-sonnet-4-5", "max_tokens": 64}
    base.update(kw)
    return GenRequest(**base)


# --- error mapping -------------------------------------------------------


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_Exceptions.RateLimitError("slow down"), RateLimitedError),
        (_Exceptions.Timeout("timed out"), TransientAIError),
        (_Exceptions.APIConnectionError("reset"), TransientAIError),
        (_Exceptions.ServiceUnavailableError("503"), TransientAIError),
        (_Exceptions.InternalServerError("500"), TransientAIError),
        (_Exceptions.ContentPolicyViolationError("blocked"), RefusedError),
        (_Exceptions.AuthenticationError("no key"), AIError),
        (_Exceptions.NotFoundError("no such model"), AIError),
        (_Exceptions.APIError("something upstream"), TransientAIError),
    ],
)
def test_errors_map_onto_the_typed_hierarchy(fake, exc, expected):
    p = provider()
    mapped = p._map_error(exc, task="judging", model="m")
    assert isinstance(mapped, expected)
    assert mapped.provider == "litellm"
    assert mapped.model == "m"


def test_a_refusal_is_not_classified_as_a_bad_request(fake):
    """ContentPolicyViolationError SUBCLASSES BadRequestError. Checking the base
    first would make a refusal look fatal-but-retryable and lose the distinction
    the resilience layer needs — refusals are never retried, bad requests are
    never retried either but for a different reason and with a different line."""
    p = provider()
    mapped = p._map_error(_Exceptions.ContentPolicyViolationError("nope"), task="t", model="m")
    assert isinstance(mapped, RefusedError)
    assert not isinstance(mapped, TransientAIError)


def test_a_context_window_error_is_fatal_not_transient(fake):
    """The same prompt is the same length next time; retrying spends calls to
    fail identically three times."""
    p = provider()
    mapped = p._map_error(_Exceptions.ContextWindowExceededError("too long"), task="t", model="m")
    assert isinstance(mapped, AIError)
    assert not isinstance(mapped, TransientAIError)


def test_a_rate_limit_carries_the_providers_own_retry_hint(fake):
    p = provider()
    mapped = p._map_error(
        _Exceptions.RateLimitError("wait", retry_after=12), task="t", model="m"
    )
    assert isinstance(mapped, RateLimitedError)
    assert mapped.retry_after == 12.0


def test_an_unparseable_retry_hint_falls_back_to_the_schedule(fake):
    p = provider()
    mapped = p._map_error(
        _Exceptions.RateLimitError("wait", retry_after="soon"), task="t", model="m"
    )
    assert mapped.retry_after is None


def test_an_aggregators_upstream_fault_is_transient_not_fatal(fake):
    """D-008, reaching us through a second aggregator — and this is the exact
    body the FIRST live call through this provider returned. LiteLLM flattens an
    OpenRouter upstream fault onto BadRequestError, which would make it fatal and
    quietly undo D-008 for every task routed through litellm. A retry is a fresh
    routing draw."""
    body = (
        'litellm.BadRequestError: OpenrouterException - {"error":{"message":'
        '"Provider returned error","code":400,"metadata":{"provider_name":"AtlasCloud"}}}'
    )
    mapped = provider()._map_error(_Exceptions.BadRequestError(body), task="t", model="m")
    assert isinstance(mapped, TransientAIError)


def test_an_upstream_fault_mentioning_response_format_does_not_blacklist_the_model(fake):
    """Ordering, mirroring openrouter.py: one bad routing draw must not downgrade
    every later call for a model that does support schemas."""
    p = provider()
    mapped = p._map_error(
        _Exceptions.BadRequestError("Provider returned error: response_format upstream hiccup"),
        task="t", model="m",
    )
    assert isinstance(mapped, TransientAIError)
    assert "m" not in p._no_native_models


def test_a_genuine_bad_request_stays_fatal(fake):
    """The transient classification has to stay narrow, or a real request error
    is retried three times to fail identically."""
    mapped = provider()._map_error(
        _Exceptions.BadRequestError("temperature must be between 0 and 2"), task="t", model="m"
    )
    assert not isinstance(mapped, TransientAIError)


def test_a_response_format_rejection_becomes_the_guards_fallback_signal(fake):
    """And the model is remembered, so the next call skips the doomed round-trip."""
    p = provider()
    mapped = p._map_error(
        _Exceptions.BadRequestError("response_format is not supported"), task="t", model="m"
    )
    assert isinstance(mapped, NativeStructuredUnsupported)
    assert "m" in p._no_native_models


# --- the native / prompt decision ---------------------------------------


def test_an_unknown_model_is_tried_natively_rather_than_downgraded(fake):
    """A wrong 'no' silently downgrades every call for a model that does support
    schemas; a wrong 'yes' costs one round-trip and is then remembered."""
    fake.supports_schema = RuntimeError("model not in litellm's map")
    assert provider()._supports_native_schema("some/new-model") is True


def test_a_model_litellm_says_has_no_schema_support_goes_to_the_prompt(fake):
    fake.supports_schema = False
    assert provider()._supports_native_schema("ollama/tiny") is False


def test_a_remembered_rejection_short_circuits_the_lookup(fake):
    p = provider()
    p._no_native_models.add("m")
    assert p._supports_native_schema("m") is False


async def test_native_mode_sends_the_schema_and_prompt_mode_does_not(fake):
    p = provider()
    await p.raw_structured(req(), SCHEMA, schema_in_prompt=False)
    sent = fake.completion_calls[-1]
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA.json_schema
    assert sent["response_format"]["json_schema"]["name"] == "thing"

    await p.raw_structured(req(), SCHEMA, schema_in_prompt=True)
    sent = fake.completion_calls[-1]
    assert "response_format" not in sent
    assert SCHEMA.full_name in sent["messages"][0]["content"]


async def test_a_known_unsupported_model_tells_the_guard_instead_of_diverging(fake):
    """Silently switching to prompt mode would leave the Guard logging `native`
    for a call that was not."""
    fake.supports_schema = False
    p = provider()
    with pytest.raises(NativeStructuredUnsupported):
        await p.raw_structured(req(), SCHEMA, schema_in_prompt=False)


# --- request shape -------------------------------------------------------


def test_the_system_prompt_leads_and_messages_keep_their_order(fake):
    p = provider()
    r = req(
        system_prompt="you are a judge",
        messages=[Message(role="user", content="one"), Message(role="assistant", content="two")],
    )
    assert p._messages(r) == [
        {"role": "system", "content": "you are a judge"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]


def test_an_empty_system_prompt_adds_no_message(fake):
    assert provider()._messages(req(system_prompt="")) == []


def test_the_configured_key_and_base_ride_on_every_call(fake):
    p = provider(api_key="sk-x", api_base="http://localhost:11434")
    kwargs = p._kwargs(req())
    assert kwargs["api_key"] == "sk-x"
    assert kwargs["api_base"] == "http://localhost:11434"


def test_no_key_is_sent_when_none_is_configured(fake):
    """An empty key means 'let litellm resolve it per upstream'. Sending
    api_key=None explicitly would override that resolution with nothing."""
    kwargs = provider(api_key="")._kwargs(req())
    assert "api_key" not in kwargs
    assert "api_base" not in kwargs


# --- response reading ----------------------------------------------------


async def test_a_content_filter_finish_reason_is_a_refusal(fake):
    async def _filtered(**kwargs):
        return _Response("", finish_reason="content_filter")

    fake.acompletion = _filtered
    with pytest.raises(RefusedError):
        await provider().generate(req())


async def test_empty_content_is_transient_not_an_empty_answer(fake):
    """An empty string used to reach the Guard's `.strip()` and be judged
    malformed three times over — this is a wire failure, and the retry ladder
    is where it belongs (openrouter.py carries the same repair)."""
    fake.reply = ""
    with pytest.raises(TransientAIError):
        await provider().generate(req())


async def test_multimodal_parts_are_joined_into_text(fake):
    async def _parts(**kwargs):
        return _Response([{"type": "text", "text": "he"}, {"type": "text", "text": "llo"}])

    fake.acompletion = _parts
    assert (await provider().generate(req())).text == "hello"


# --- embeddings ----------------------------------------------------------


def _vectors(n, width):
    return type("R", (), {"data": [{"embedding": [0.0] * (width - 1) + [1.0]} for _ in range(n)]})()


async def test_a_wrong_width_vector_is_refused_before_it_reaches_the_database(fake):
    """profile_embeddings is vector(768). Letting a 1536-wide row through turns a
    model-choice mistake into an asyncpg error naming a column."""
    fake.embedding = lambda **kw: _vectors(1, 1536)
    with pytest.raises(AIError) as excinfo:
        await provider().embed(["hi"], "openai/text-embedding-3-small")
    assert "1536" in str(excinfo.value)
    assert str(EMBEDDING_DIMENSIONS) in str(excinfo.value)


async def test_a_correct_width_vector_is_returned_normalized(fake):
    fake.embedding = lambda **kw: _vectors(2, EMBEDDING_DIMENSIONS)
    out = await provider().embed(["a", "b"], "openai/text-embedding-3-small")
    assert len(out) == 2
    for vector in out:
        assert len(vector) == EMBEDDING_DIMENSIONS
        assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-9


async def test_a_short_count_is_transient_rather_than_a_silent_misalignment(fake):
    """Two texts in, one vector out would otherwise pair the wrong person's
    embedding with the wrong row."""
    fake.embedding = lambda **kw: _vectors(1, EMBEDDING_DIMENSIONS)
    with pytest.raises(TransientAIError):
        await provider().embed(["a", "b"], "m")


async def test_dimensions_is_requested_first_and_dropped_only_if_rejected(fake):
    seen = []

    def _embed(**kwargs):
        seen.append(dict(kwargs))
        if "dimensions" in kwargs:
            raise _Exceptions.BadRequestError("dimensions is not supported by this model")
        return _vectors(1, EMBEDDING_DIMENSIONS)

    fake.embedding = _embed
    await provider().embed(["a"], "m")
    assert "dimensions" in seen[0] and seen[0]["dimensions"] == EMBEDDING_DIMENSIONS
    assert "dimensions" not in seen[1]


async def test_an_auth_failure_is_not_retried_as_an_unsupported_parameter(fake):
    """The narrow match matters: a 401 whose body happens to mention dimensions
    must not become a second doomed call."""
    def _embed(**kwargs):
        raise _Exceptions.AuthenticationError("invalid api key")

    fake.embedding = _embed
    with pytest.raises(AIError):
        await provider().embed(["a"], "m")
    assert len(fake.embedding_calls) == 1


@pytest.mark.parametrize(
    "message, expected",
    [
        ("dimensions is not supported", True),
        ("unexpected keyword argument 'dimensions'", True),
        ("rate limit exceeded", False),
        ("invalid api key", False),
        # The trap this guard exists for: the word appears, but nothing says
        # the parameter was the problem.
        ("upstream busy while embedding 768 dimensions", False),
    ],
)
def test_the_unsupported_parameter_match_is_narrow(message, expected):
    assert lp._is_unsupported_parameter(Exception(message), "dimensions") is expected


# --- registry wiring -----------------------------------------------------


def _config(providers: dict) -> AIConfig:
    routing = {
        task: {"provider": "openrouter", "model": "x"}
        for task in (
            "dispute_followups", "trait_extraction", "persona_digest",
            "scenario_generation", "date_simulation", "judging",
            "chat_reply", "chat_compaction",
        )
    }
    return AIConfig.model_validate(
        {
            "providers": providers,
            "embeddings": {"provider": "google", "model": "gemini-embedding-001"},
            "routing": routing,
        }
    )


def test_the_registry_builds_litellm_and_passes_its_api_base():
    config = _config(
        {
            "google": {"api_key_env": "G"},
            "openrouter": {"api_key_env": "O"},
            "litellm": {"api_key_env": "L", "api_base": "http://proxy:4000"},
        }
    )
    built = build_providers(config, getenv=lambda k, d="": {"L": "sk-l"}.get(k, d))
    assert isinstance(built["litellm"], lp.LiteLLMProvider)
    assert built["litellm"]._api_base == "http://proxy:4000"
    assert built["litellm"]._api_key == "sk-l"


def test_an_api_base_on_a_provider_that_ignores_it_fails_at_startup():
    """Config coherence fails at boot, not mid-date (S2-B5). Left in place it
    would read as though google were pointed at a proxy, which it never is."""
    config = _config(
        {
            "google": {"api_key_env": "G", "api_base": "http://nope"},
            "openrouter": {"api_key_env": "O"},
        }
    )
    with pytest.raises(AIError, match="api_base"):
        build_providers(config, getenv=lambda k, d="": d)


def test_an_empty_litellm_key_is_a_normal_state_not_a_missing_one():
    """LiteLLM resolves a key per upstream from the environment. This must build
    without one — and the provider must not send an empty key that would
    override that resolution."""
    config = _config(
        {"google": {"api_key_env": "G"}, "openrouter": {"api_key_env": "O"},
         "litellm": {"api_key_env": "LITELLM_API_KEY"}}
    )
    built = build_providers(config, getenv=lambda k, d="": d)
    assert built["litellm"]._api_key is None


def test_the_protocol_is_satisfied():
    """base.AIProvider is runtime_checkable — this is the locked interface
    (ai_interaction.md §2) that every feature module depends on."""
    from app.ai.base import AIProvider

    assert isinstance(lp.LiteLLMProvider(""), AIProvider)


def test_the_guard_can_drive_it():
    """structured.py's RawStructuredProvider needs these three names; a missing
    one would only surface on the first structured call."""
    p = lp.LiteLLMProvider("")
    assert p.name == "litellm"
    assert isinstance(p.supports_native_structured, bool)
    assert hasattr(p, "limiter") and hasattr(p, "raw_structured")


def test_a_missing_litellm_package_names_the_fix(monkeypatch):
    """A dependency in pyproject.toml is not a dependency in the running image
    (D-012, trap 5). The error has to say that, at the call that wanted it."""
    monkeypatch.setattr(lp, "_litellm_module", None)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _no_litellm(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("No module named 'litellm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(
        __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__,
        "__import__",
        _no_litellm,
    )
    with pytest.raises(AIError, match="docker compose build api"):
        lp._litellm()
