"""AIProvider protocol, value types, and the typed error hierarchy (S2-B1).

Typed errors are the contract with the resilience layer: SDK/HTTP exceptions
must never escape this module. The Date Simulation Module's checkpointing (and
every other caller) reasons about THESE types, nothing vendor-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, get_args, runtime_checkable

# The eight routed tasks (ai_interaction.md §3). A call site passes a task
# name; the router resolves provider + model. "embeddings" is not a routed
# task — it is pinned separately.
TaskName = Literal[
    "dispute_followups",
    "trait_extraction",
    "persona_digest",
    "scenario_generation",
    "date_simulation",
    "judging",
    "chat_reply",
    "chat_compaction",
]
ROUTED_TASKS: tuple[str, ...] = get_args(TaskName)

# The mandatory per-call log line's outcome vocabulary (ai_interaction.md §5).
CallOutcome = Literal["ok", "malformed", "rate_limited", "refused", "gave_up"]

# Must match the `profile_embeddings` vector(768) column — the schema is the
# system truth for dimensionality, and every stored vector must share it or
# similarity comparisons are meaningless (ai_interaction.md §3).
#
# It lives HERE, in the shared contract, rather than in the provider that
# happens to serve embeddings today: two providers can embed, and two
# constants that must agree are two constants that will eventually disagree.
EMBEDDING_DIMENSIONS = 768


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class GenRequest:
    """One generation request. `model` is filled from routing config by the
    caller (via TaskRouter) — never hardcoded at a call site. `task` labels
    the mandatory log line; probes may use a non-routed label."""

    task: str
    model: str
    system_prompt: str = ""
    messages: list[Message] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class GenResult:
    text: str
    finish_reason: str | None = None


@dataclass(frozen=True)
class VersionedSchema:
    """A JSON schema with an identity. Stored artifacts record the version
    they were produced against, so old data stays readable after evolution."""

    name: str
    version: int
    json_schema: dict[str, Any]

    @property
    def full_name(self) -> str:
        return f"{self.name}.v{self.version}"


class AIError(Exception):
    """Base of the hierarchy; carries task/provider/model so any error can be
    logged and explained without a debugger (§7)."""

    def __init__(
        self,
        message: str,
        *,
        task: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ):
        super().__init__(message)
        self.task = task
        self.provider = provider
        self.model = model


class TransientAIError(AIError):
    """Retryable: 5xx, timeouts, connection resets. The resilience layer backs
    off and retries these; anything else fails fast."""


class RateLimitedError(TransientAIError):
    """429. Retryable with backoff; carries the provider's retry hint when given."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kw):
        super().__init__(message, **kw)
        self.retry_after = retry_after


class RefusedError(AIError):
    """The model/provider declined (safety block, content filter). Not retryable."""


class StructuredOutputError(AIError):
    """The Guard's typed give-up (§17): repair attempts exhausted. Carries the
    raw output so the failure is inspectable. Never a silent default (§10)."""

    def __init__(self, message: str, *, raw_output: str, **kw):
        super().__init__(message, **kw)
        self.raw_output = raw_output


class NativeStructuredUnsupported(AIError):
    """Internal signal: this model rejects native json_schema mode; the Guard
    falls back to embedding the schema in the prompt (ai_interaction.md §4.1)."""


class RouteUnresolvedError(AIError):
    """The task's model slot is still `free-model-of-choice` (deliberately
    unfilled, owner decision 2026-09-01). Surfacing this loudly beats guessing
    a model to make something run."""


@runtime_checkable
class AIProvider(Protocol):
    """The locked interface (ai_interaction.md §2). Core logic depends ONLY on
    this protocol — no module imports google.py or openrouter.py directly."""

    name: str

    async def generate(self, req: GenRequest) -> GenResult: ...

    async def generate_structured(self, req: GenRequest, schema: VersionedSchema) -> dict:
        """Returns a dict already validated against `schema`; raises
        StructuredOutputError on give-up. Implementations delegate to the one
        Guard in structured.py — never their own validation loop (§16)."""
        ...

    async def embed(self, texts: list[str], model: str) -> list[list[float]]: ...
