"""GoogleProvider (S2-B2) — the google-genai SDK behind the AIProvider protocol.

Native structured output via `response_json_schema`; embeddings (the pinned
embedding model lives on this provider). SDK exceptions are mapped to the
typed hierarchy at this boundary — they never escape the module.

Nothing outside app/ai/ may import this file (§16).
"""

from __future__ import annotations

import math

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.ai.base import (
    AIError,
    GenRequest,
    GenResult,
    RateLimitedError,
    RefusedError,
    TransientAIError,
    VersionedSchema,
)
from app.ai.resilience import RateLimiter, execute
from app.ai.structured import guarded_structured_call, schema_prompt_block

# Must match the profile_embeddings vector(768) column — the schema is the
# system truth for dimensionality; gemini-embedding-001's default is 3072.
EMBEDDING_DIMENSIONS = 768


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def _map_error(exc: genai_errors.APIError, *, task: str, model: str) -> AIError:
    kw = {"task": task, "provider": "google", "model": model}
    if exc.code == 429:
        return RateLimitedError(f"google rate limit: {exc.message}", **kw)
    if exc.code is not None and exc.code >= 500:
        return TransientAIError(f"google server error {exc.code}: {exc.message}", **kw)
    return AIError(f"google API error {exc.code}: {exc.message}", **kw)


class GoogleProvider:
    name = "google"
    supports_native_structured = True

    def __init__(self, api_key: str, *, limiter: RateLimiter | None = None):
        self._api_key = api_key
        self.limiter = limiter or RateLimiter()
        self._client: genai.Client | None = None

    def _client_or_raise(self, *, task: str, model: str) -> genai.Client:
        if not self._api_key:
            raise AIError(
                "google provider has no API key (GOOGLE_AI_API_KEY is empty)",
                task=task, provider=self.name, model=model,
            )
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def _contents(req: GenRequest) -> list[genai_types.Content]:
        role_map = {"user": "user", "assistant": "model"}
        return [
            genai_types.Content(
                role=role_map[m.role], parts=[genai_types.Part(text=m.content)]
            )
            for m in req.messages
        ]

    async def _raw_generate(
        self,
        req: GenRequest,
        *,
        response_schema: VersionedSchema | None = None,
        schema_in_prompt: bool = False,
    ) -> GenResult:
        client = self._client_or_raise(task=req.task, model=req.model)
        system = req.system_prompt or None
        if response_schema is not None and schema_in_prompt:
            system = (system or "") + schema_prompt_block(response_schema)
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=req.temperature,
            max_output_tokens=req.max_tokens,
            response_mime_type=(
                "application/json"
                if response_schema is not None and not schema_in_prompt
                else None
            ),
            response_json_schema=(
                response_schema.json_schema
                if response_schema is not None and not schema_in_prompt
                else None
            ),
        )
        try:
            resp = await client.aio.models.generate_content(
                model=req.model, contents=self._contents(req), config=config
            )
        except genai_errors.APIError as exc:
            raise _map_error(exc, task=req.task, model=req.model) from exc

        if resp.prompt_feedback is not None and resp.prompt_feedback.block_reason:
            raise RefusedError(
                f"google safety block: {resp.prompt_feedback.block_reason}",
                task=req.task, provider=self.name, model=req.model,
            )
        finish = None
        if resp.candidates:
            finish = str(resp.candidates[0].finish_reason or "")
            if finish == "FinishReason.SAFETY":
                raise RefusedError(
                    "google safety block on the response",
                    task=req.task, provider=self.name, model=req.model,
                )
        text = resp.text
        if text is None:
            raise TransientAIError(
                f"google returned no text (finish_reason={finish})",
                task=req.task, provider=self.name, model=req.model,
            )
        return GenResult(text=text, finish_reason=finish)

    # --- AIProvider protocol ---

    async def generate(self, req: GenRequest) -> GenResult:
        return await execute(
            lambda: self._raw_generate(req),
            task=req.task, provider=self.name, model=req.model, limiter=self.limiter,
        )

    async def generate_structured(self, req: GenRequest, schema: VersionedSchema) -> dict:
        # Delegates to the one Guard (§16) — no validation loop lives here.
        return await guarded_structured_call(self, req, schema)

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        async def _call() -> list[list[float]]:
            client = self._client_or_raise(task="embeddings", model=model)
            try:
                resp = await client.aio.models.embed_content(
                    model=model,
                    contents=texts,
                    config=genai_types.EmbedContentConfig(
                        output_dimensionality=EMBEDDING_DIMENSIONS
                    ),
                )
            except genai_errors.APIError as exc:
                raise _map_error(exc, task="embeddings", model=model) from exc
            if not resp.embeddings:
                raise TransientAIError(
                    "google returned no embeddings",
                    task="embeddings", provider=self.name, model=model,
                )
            # At non-default dimensionality the API returns UNnormalized vectors;
            # normalized storage keeps cosine and dot-product interchangeable
            # (ai_interaction.md §3, revision 2026-09-01).
            return [_l2_normalize(list(e.values or [])) for e in resp.embeddings]

        return await execute(
            _call, task="embeddings", provider=self.name, model=model, limiter=self.limiter
        )

    # --- Guard hook (only structured.py calls this) ---

    async def raw_structured(
        self, req: GenRequest, schema: VersionedSchema, *, schema_in_prompt: bool
    ) -> str:
        result = await self._raw_generate(
            req, response_schema=schema, schema_in_prompt=schema_in_prompt
        )
        return result.text
