"""LiteLLMProvider — one interface to a hundred model providers, behind ours.

**Why this sits ALONGSIDE `google.py` and `openrouter.py` rather than replacing
them.** LiteLLM is a breadth provider. The two hand-written clients are the ones
this project actually measured, pinned and closed its gates on, and each carries
knowledge a generic client does not: that OpenRouter's `"Provider returned
error"` is transient because one model id is several upstreams (D-008), that
`gemini-embedding-001` returns UNnormalized vectors at non-default
dimensionality. Routing everything through LiteLLM would throw that away to gain
nothing on the two providers already working. So this is the door to everything
ELSE — Anthropic, OpenAI, Azure, Bedrock, Vertex, Groq, Together, Mistral,
DeepSeek, a local Ollama or vLLM — reached by writing a LiteLLM model string
into `config/ai.yaml` and restarting.

**The model string IS the routing.** LiteLLM names a model as
`<upstream>/<model>`, so the existing config shape needs no new concept:

    routing:
      judging: { provider: litellm, model: "anthropic/claude-sonnet-4-5" }
      chat_reply: { provider: litellm, model: "ollama/llama3.1" }

Everything downstream is unchanged. `TaskRouter` resolves `litellm` like any
other provider, the Guard is still the only place model JSON is parsed (§16),
and `traits.extracted_by` records `litellm/<the full string>` so a stored
artifact still says exactly what made it.

**Credentials.** Unlike the other two providers, LiteLLM usually reads a
DIFFERENT key per upstream from the environment (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, …). So an empty `api_key_env` here is NOT an error the way it
is for google or openrouter — it means "let LiteLLM resolve the key itself".
Set `LITELLM_API_KEY` only when you want one key used for every LiteLLM route,
which is the right shape for a LiteLLM proxy or a single upstream. Either way a
missing credential surfaces as a typed `AIError` at call time, never as a guess.

**The file is `litellm_provider.py`, not `litellm.py`** — a deliberate departure
from the `<provider>.py` naming in `ai_interaction.md` §1. A module named
`app/ai/litellm.py` that does `import litellm` works under Python 3's absolute
imports, but it puts two different `litellm` names in every traceback and one
`from . import litellm` away from a genuinely confusing bug. Named in the plan.

Nothing outside `app/ai/` may import this file (§16).
"""

from __future__ import annotations

import logging
from typing import Any

from app.ai.base import (
    EMBEDDING_DIMENSIONS,
    AIError,
    GenRequest,
    GenResult,
    NativeStructuredUnsupported,
    RateLimitedError,
    RefusedError,
    TransientAIError,
    VersionedSchema,
)
from app.ai.resilience import RateLimiter, execute
from app.ai.structured import guarded_structured_call, schema_prompt_block
from app.logging_setup import log_event

logger = logging.getLogger("app.ai")

# Cached across calls: importing litellm costs seconds (it pulls tokenizers and
# a large provider map), and the registry builds every configured provider at
# boot whether or not a task routes to it.
_litellm_module: Any = None


def _litellm() -> Any:
    """Import litellm on first use, not at module import.

    Two reasons, both learned here. Boot: `registry.build_providers` runs inside
    the lifespan, and a multi-second import would sit in front of `/health` for
    everyone, including deployments that route nothing to LiteLLM. And staleness:
    a dependency named in `pyproject.toml` is not a dependency in the RUNNING
    image (D-012, PICKUP trap 5) — an ImportError at module scope would take the
    whole app down at boot with a traceback about a provider nobody uses, where
    this raises a typed error naming the fix, at the call that wanted it.
    """
    global _litellm_module
    if _litellm_module is None:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - exercised by the message test
            raise AIError(
                "the litellm provider is configured but the litellm package is not "
                "installed in the running image. It is declared in pyproject.toml, so "
                "this usually means the image predates it: "
                "`docker compose build api && docker compose up -d api` "
                "(PICKUP trap 5).",
                provider="litellm",
            ) from exc
        # LiteLLM writes its own log stream, and both halves of it break §7's
        # promise that the log is one machine-readable record you can rebuild a
        # run from. `suppress_debug_info` stops the startup banner; it does NOT
        # stop the ANSI-coloured `LiteLLM:INFO - completion() model=…` line it
        # emits per call, which arrives on stdout beside our JSON and once per
        # model call would bury a date's reconstruction. Observed on the first
        # live call through this provider.
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
        # NOT setting `drop_params`. It silently discards parameters an upstream
        # does not support — including `response_format`, which would turn a
        # native structured call into a plain one whose output then fails
        # validation three times in the Guard. A loud failure that the Guard can
        # convert into its prompt fallback is worth more than a quiet downgrade.
        _litellm_module = litellm
    return _litellm_module


def _exception_types(litellm: Any) -> dict[str, type[BaseException]]:
    """Resolve LiteLLM's exception classes by name, tolerating version drift.

    LiteLLM's exception surface moves between releases and not every name exists
    in every version. Looking them up by name with a fallback keeps a `pip
    install -U litellm` from turning into an AttributeError at the first 429.
    """
    exceptions = getattr(litellm, "exceptions", litellm)
    resolved: dict[str, type[BaseException]] = {}
    for attr in (
        "RateLimitError",
        "AuthenticationError",
        "PermissionDeniedError",
        "ContentPolicyViolationError",
        "ContextWindowExceededError",
        "UnsupportedParamsError",
        "BadRequestError",
        "NotFoundError",
        "Timeout",
        "APIConnectionError",
        "ServiceUnavailableError",
        "InternalServerError",
        "APIError",
    ):
        found = getattr(exceptions, attr, None) or getattr(litellm, attr, None)
        if isinstance(found, type) and issubclass(found, BaseException):
            resolved[attr] = found
    return resolved


class LiteLLMProvider:
    name = "litellm"
    # Per model, decided per call: LiteLLM knows which upstreams implement a
    # JSON schema, and the Guard handles the ones that do not.
    supports_native_structured = True

    def __init__(
        self,
        api_key: str,
        *,
        limiter: RateLimiter | None = None,
        api_base: str | None = None,
    ):
        # Empty is legitimate here — see the module docstring on credentials.
        self._api_key = api_key or None
        self._api_base = api_base or None
        self.limiter = limiter or RateLimiter()
        # Models observed rejecting a native schema — skip the doomed round-trip.
        self._no_native_models: set[str] = set()

    # --- error mapping: the boundary promise in base.py ---

    def _map_error(self, exc: BaseException, *, task: str, model: str) -> AIError:
        """Map a LiteLLM exception onto this project's typed hierarchy.

        Ordered specific-to-general because LiteLLM mirrors OpenAI's hierarchy,
        where `ContentPolicyViolationError` and `ContextWindowExceededError` are
        both subclasses of `BadRequestError` — checking the base first would
        classify a refusal as a fatal request error and lose the distinction the
        resilience layer needs.
        """
        kw = {"task": task, "provider": self.name, "model": model}
        types = _exception_types(_litellm())
        text = str(exc)

        def is_a(*names: str) -> bool:
            classes = tuple(types[n] for n in names if n in types)
            return bool(classes) and isinstance(exc, classes)

        if is_a("RateLimitError"):
            # LiteLLM normalises the upstream's hint onto the exception when the
            # upstream sends one; without it the rate-limit schedule applies.
            retry_after = getattr(exc, "retry_after", None)
            try:
                retry_after = float(retry_after) if retry_after is not None else None
            except (TypeError, ValueError):
                retry_after = None
            return RateLimitedError(
                f"litellm rate limit: {text[:300]}", retry_after=retry_after, **kw
            )
        if is_a("ContentPolicyViolationError"):
            return RefusedError(f"litellm content policy refusal: {text[:300]}", **kw)
        if is_a("ContextWindowExceededError"):
            # Not retryable: the same prompt is the same length next time.
            return AIError(f"litellm context window exceeded: {text[:300]}", **kw)
        if _is_upstream_fault(text):
            # D-008, reaching us through a second aggregator. An aggregator
            # serves one model id from SEVERAL upstreams and picks per request,
            # so "the upstream we happened to draw failed" is structurally
            # different from "your request was bad" — and LiteLLM flattens both
            # onto BadRequestError, which would make it FATAL.
            #
            # This is not hypothetical: the first live call made through this
            # provider was an OpenRouter `"Provider returned error"` from
            # AtlasCloud, which openrouter.py has treated as transient since
            # D-008. Classifying it fatal here would have quietly undone that
            # fix for every task routed through litellm — a retry is a fresh
            # routing draw and usually lands somewhere else.
            #
            # Checked BEFORE the response_format case below, exactly as
            # openrouter.py does, so an upstream fault whose body happens to
            # mention response_format cannot blacklist the model for the
            # process on the evidence of one bad draw.
            return TransientAIError(
                f"litellm upstream provider error (retryable): {text[:300]}", **kw
            )
        if is_a("UnsupportedParamsError") or (
            is_a("BadRequestError") and "response_format" in text
        ):
            # This upstream does not implement JSON schema. Remember it so the
            # next call skips straight to the Guard's prompt fallback (§4.1).
            self._no_native_models.add(model)
            return NativeStructuredUnsupported(
                f"model {model} rejects a native response_format schema: {text[:200]}", **kw
            )
        if is_a("AuthenticationError", "PermissionDeniedError"):
            return AIError(
                f"litellm auth failure for {model} — LiteLLM resolves a key per "
                f"upstream from the environment unless one is configured: {text[:200]}",
                **kw,
            )
        if is_a("NotFoundError"):
            return AIError(f"litellm model not found: {model}: {text[:200]}", **kw)
        if is_a("Timeout", "APIConnectionError", "ServiceUnavailableError", "InternalServerError"):
            return TransientAIError(f"litellm transport/server error: {text[:300]}", **kw)
        if is_a("BadRequestError"):
            return AIError(f"litellm bad request: {text[:300]}", **kw)
        if is_a("APIError"):
            # The base class catches upstream faults LiteLLM could not classify.
            # Transient by default: these are overwhelmingly 5xx-shaped, and a
            # retry costs one call where a fatal misclassification costs a whole
            # analysis (the D-008 lesson, applied to a different aggregator).
            return TransientAIError(f"litellm upstream error: {text[:300]}", **kw)
        return AIError(f"litellm raised {type(exc).__name__}: {text[:300]}", **kw)

    # --- request/response shaping ---

    def _messages(self, req: GenRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages += [{"role": m.role, "content": m.content} for m in req.messages]
        return messages

    def _kwargs(self, req: GenRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": self._messages(req),
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        return kwargs

    async def _complete(self, kwargs: dict[str, Any], *, task: str, model: str) -> Any:
        litellm = _litellm()
        try:
            return await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise self._map_error(exc, task=task, model=model) from exc

    def _extract_text(self, resp: Any, *, task: str, model: str) -> tuple[str, str | None]:
        kw = {"task": task, "provider": self.name, "model": model}
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise TransientAIError("litellm returned no choices", **kw)
        choice = choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish == "content_filter":
            raise RefusedError("litellm content filter refusal", **kw)
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, list):
            # The multimodal "parts" form. Only text parts are ours to read —
            # the list itself used to reach the Guard's `.strip()` (openrouter.py
            # carries the same repair).
            content = "".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict)
            )
        if not isinstance(content, str) or not content:
            raise TransientAIError(f"litellm returned empty content (finish={finish})", **kw)
        return content, finish

    def _supports_native_schema(self, model: str) -> bool:
        """Ask LiteLLM whether this model implements a JSON schema.

        Unknown to LiteLLM's map (a proxy route, a self-hosted model, a release
        newer than the installed litellm) is answered OPTIMISTICALLY: try the
        native mode once and let the upstream say no, because a wrong "no" here
        would silently downgrade every call for a model that supports it, where
        a wrong "yes" costs one round-trip and is remembered.
        """
        if model in self._no_native_models:
            return False
        litellm = _litellm()
        checker = getattr(litellm, "supports_response_schema", None)
        if checker is None:
            return True
        try:
            return bool(checker(model=model))
        except Exception:  # noqa: BLE001 - see the optimistic-default note above
            # Any failure to ANSWER the question is not an answer of "no".
            # litellm raises here for models missing from its map, and the
            # blanket catch is the point: every such failure means "we do not
            # know", and the safe reading of "we do not know" is to try once.
            return True

    # --- AIProvider protocol ---

    async def generate(self, req: GenRequest) -> GenResult:
        async def _call() -> GenResult:
            resp = await self._complete(self._kwargs(req), task=req.task, model=req.model)
            text, finish = self._extract_text(resp, task=req.task, model=req.model)
            return GenResult(text=text, finish_reason=finish)

        return await execute(
            _call, task=req.task, provider=self.name, model=req.model, limiter=self.limiter
        )

    async def generate_structured(self, req: GenRequest, schema: VersionedSchema) -> dict:
        # Delegates to the one Guard (§16) — no validation loop lives here.
        return await guarded_structured_call(self, req, schema)

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embeddings through LiteLLM — usable, but read the constraint first.

        `profile_embeddings` is `vector(768)` and every stored vector must come
        from ONE model or cosine similarity is comparing incomparable things
        (ai_interaction.md §3). So switching the embeddings pin to a LiteLLM
        route is a migration plus a re-embed of every user, never a config edit
        on its own — and this method REFUSES a wrong-width vector rather than
        letting it reach the database, where the failure would be an asyncpg
        error naming a column instead of a model.
        """
        kw = {"task": "embeddings", "provider": self.name, "model": model}

        async def _call() -> list[list[float]]:
            litellm = _litellm()
            kwargs: dict[str, Any] = {"model": model, "input": texts}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._api_base:
                kwargs["api_base"] = self._api_base
            # Most embedding APIs that can truncate take `dimensions`; those that
            # cannot reject the parameter outright. Ask WITH it first — a model
            # whose native width is not 768 is only usable here if it truncates —
            # and on a rejection of that one parameter, ask again without it. The
            # width check below is what makes the second attempt safe: it refuses
            # a 3072-wide vector rather than storing one.
            try:
                resp = await litellm.aembedding(**kwargs, dimensions=EMBEDDING_DIMENSIONS)
            except Exception as exc:
                if not _is_unsupported_parameter(exc, "dimensions"):
                    raise self._map_error(exc, task="embeddings", model=model) from exc
                log_event(
                    logger, "embedding_dimensions_unsupported", level=logging.INFO,
                    provider=self.name, model=model, error=str(exc)[:200],
                )
                try:
                    resp = await litellm.aembedding(**kwargs)
                except Exception as inner:
                    raise self._map_error(inner, task="embeddings", model=model) from inner

            rows = getattr(resp, "data", None) or []
            vectors: list[list[float]] = []
            for row in rows:
                raw = row.get("embedding") if isinstance(row, dict) else getattr(row, "embedding", None)
                if raw is None:
                    raise TransientAIError("litellm returned a row with no embedding", **kw)
                vectors.append([float(v) for v in raw])
            if len(vectors) != len(texts):
                raise TransientAIError(
                    f"litellm returned {len(vectors)} embeddings for {len(texts)} inputs", **kw
                )
            for vector in vectors:
                if len(vector) != EMBEDDING_DIMENSIONS:
                    raise AIError(
                        f"model {model} returned {len(vector)}-dimensional vectors but "
                        f"profile_embeddings is vector({EMBEDDING_DIMENSIONS}). Changing "
                        "the embedding model is a migration and a re-embed of every "
                        "user, not a config edit (ai_interaction.md §3).",
                        **kw,
                    )
            return [_l2_normalize(v) for v in vectors]

        return await execute(
            _call, task="embeddings", provider=self.name, model=model, limiter=self.limiter
        )

    # --- Guard hook (only structured.py calls this) ---

    async def raw_structured(
        self, req: GenRequest, schema: VersionedSchema, *, schema_in_prompt: bool
    ) -> str:
        use_prompt = schema_in_prompt or not self._supports_native_schema(req.model)
        if use_prompt and not schema_in_prompt:
            # Known-unsupported: tell the Guard rather than silently diverging
            # from the mode it thinks it is in.
            raise NativeStructuredUnsupported(
                f"model {req.model} has no native json_schema mode",
                task=req.task, provider=self.name, model=req.model,
            )

        if use_prompt:
            adjusted = GenRequest(
                task=req.task,
                model=req.model,
                system_prompt=(req.system_prompt or "") + schema_prompt_block(schema),
                messages=req.messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            kwargs = self._kwargs(adjusted)
        else:
            kwargs = self._kwargs(req)
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.name,
                    "strict": True,
                    "schema": schema.json_schema,
                },
            }

        resp = await self._complete(kwargs, task=req.task, model=req.model)
        text, _ = self._extract_text(resp, task=req.task, model=req.model)
        return text


# Phrases an aggregator uses to say "the upstream I routed you to failed",
# as opposed to "your request was wrong". Kept as data rather than buried in an
# `if` because this list grows every time a new aggregator is routed through
# LiteLLM, and the next person needs to see where to add one.
_UPSTREAM_FAULT_MARKERS: tuple[str, ...] = (
    "provider returned error",   # OpenRouter (D-008)
    "upstream error",
    "no healthy upstream",       # proxy/gateway shapes
    "no deployments available",  # a LiteLLM proxy with every deployment cooling off
)


def _is_upstream_fault(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _UPSTREAM_FAULT_MARKERS)


def _is_unsupported_parameter(exc: BaseException, parameter: str) -> bool:
    """True when the upstream rejected one named parameter, rather than failing.

    Matched on the message because that is the only place the information exists
    — LiteLLM normalises status codes across upstreams but not their wording. It
    is deliberately narrow: the parameter name must appear alongside a word that
    means "not supported", so a rate limit or an auth failure whose body happens
    to mention `dimensions` is not mistaken for one.
    """
    text = str(exc).lower()
    if parameter not in text:
        return False
    return any(
        phrase in text
        for phrase in ("unsupported", "not supported", "unrecognized", "unexpected keyword", "invalid parameter")
    )


def _l2_normalize(vector: list[float]) -> list[float]:
    """Normalized storage keeps cosine and dot-product interchangeable, which is
    what `app/matching.py` relies on. Providers differ on whether they normalize;
    doing it here means the database never has to care which one wrote a row."""
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]
