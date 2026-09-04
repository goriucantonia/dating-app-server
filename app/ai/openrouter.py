"""OpenRouterProvider (S2-B3) — the OpenAI-compatible REST surface via httpx.

Native `response_format: json_schema` where the model supports it; where it
does not (a 400 naming response_format), `NativeStructuredUnsupported` tells
the Guard to embed the schema in the prompt instead (§4.1), and the model is
remembered so the next call skips the failed round-trip.

OpenRouter does not serve embeddings — they are pinned to the google provider
(ai_interaction.md §3) — so `embed` here is a typed error, not a fallback.

Nothing outside app/ai/ may import this file (§16).
"""

from __future__ import annotations

import httpx

from app.ai.base import (
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

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    name = "openrouter"
    supports_native_structured = True  # per model; fallback handled per call

    def __init__(self, api_key: str, *, limiter: RateLimiter | None = None):
        self._api_key = api_key
        self.limiter = limiter or RateLimiter()
        self._http: httpx.AsyncClient | None = None
        # Models observed rejecting response_format — skip the doomed attempt next time.
        self._no_native_models: set[str] = set()

    def _http_or_raise(self, *, task: str, model: str) -> httpx.AsyncClient:
        if not self._api_key:
            raise AIError(
                "openrouter provider has no API key (OPENROUTER_API_KEY is empty)",
                task=task, provider=self.name, model=model,
            )
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=_BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-Title": "dating-app-ai",
                },
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0),
            )
        return self._http

    async def _chat(
        self, payload: dict, *, task: str, model: str,
        sent_native_schema: bool = False,
    ) -> dict:
        http = self._http_or_raise(task=task, model=model)
        kw = {"task": task, "provider": self.name, "model": model}
        try:
            resp = await http.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise TransientAIError(f"openrouter transport error: {exc}", **kw) from exc

        if resp.status_code == 429:
            retry_after = None
            if resp.headers.get("retry-after", "").isdigit():
                retry_after = float(resp.headers["retry-after"])
            raise RateLimitedError("openrouter rate limit (429)", retry_after=retry_after, **kw)
        if resp.status_code >= 500:
            raise TransientAIError(f"openrouter server error {resp.status_code}", **kw)
        if resp.status_code == 403:
            raise RefusedError(f"openrouter moderation refusal: {resp.text[:300]}", **kw)
        if resp.status_code != 200 and "Provider returned error" in resp.text:
            # OpenRouter serves one model id from SEVERAL upstream providers and
            # picks per request. This body is OpenRouter telling us the UPSTREAM
            # failed — structurally different from OpenRouter rejecting our
            # request, which carries its own message. Observed: `trait_extraction`
            # 400ing via AtlasCloud repeatedly while the SAME model, same
            # parameters, served other tasks fine seconds apart, and while plain
            # calls to that same upstream succeeded.
            #
            # So it is TRANSIENT, and transient is what the retry ladder is for:
            # a retry is a fresh routing draw and usually lands somewhere else.
            # Treating it as fatal made a working pipeline look broken.
            raise TransientAIError(
                f"openrouter upstream provider error (retryable): {resp.text[:200]}",
                **kw,
            )
        if resp.status_code == 400 and "response_format" in resp.text:
            # Explicit: this model really does reject native schemas. Remember
            # it so we stop paying for the doomed attempt. Checked AFTER the
            # "Provider returned error" case above, so an upstream fault
            # whose body happens to mention response_format cannot
            # blacklist the model for the process (audit 2026-09-02).
            self._no_native_models.add(model)
            raise NativeStructuredUnsupported(
                f"model {model} rejects response_format json_schema", **kw
            )
        if resp.status_code == 400 and sent_native_schema:
            # A bare 400 while we had a schema attached. OpenRouter routes one
            # model across several upstream providers, and they do not all
            # implement structured output to the same standard — observed with
            # `trait_extraction.v1` (the largest schema) 400ing on one upstream
            # while the SAME model served smaller schemas fine minutes apart.
            # Deliberately NOT added to _no_native_models: this is an upstream
            # roulette result, not a property of the model, so blacklisting it
            # for the process would downgrade every later call on the evidence
            # of one bad draw. Fall back for this call only.
            raise NativeStructuredUnsupported(
                f"openrouter 400 while sending a native schema "
                f"(upstream provider likely rejected it): {resp.text[:200]}",
                **kw,
            )
        if resp.status_code != 200:
            raise AIError(
                f"openrouter error {resp.status_code}: {resp.text[:300]}", **kw
            )

        try:
            data = resp.json()
        except ValueError as exc:
            # A 200 with an empty or HTML body (an aggregator interstitial).
            # Transient: a retry is a fresh routing draw (D-008).
            raise TransientAIError(
                f"openrouter returned a non-JSON 200 body: {resp.text[:200]!r}", **kw
            ) from exc
        if not isinstance(data, dict):
            raise TransientAIError(
                f"openrouter returned a JSON {type(data).__name__}, not an object", **kw
            )
        # OpenRouter can tunnel an upstream error inside a 200 body.
        if "error" in data:
            err = data["error"]
            code = err.get("code")
            if isinstance(code, str) and code.isdigit():
                code = int(code)  # "429" is 429
            if code == 429:
                raise RateLimitedError(f"openrouter upstream rate limit: {err.get('message')}", **kw)
            if isinstance(code, int) and code >= 500:
                raise TransientAIError(f"openrouter upstream error: {err.get('message')}", **kw)
            raise AIError(f"openrouter upstream error {code}: {err.get('message')}", **kw)
        return data

    def _payload(self, req: GenRequest) -> dict:
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages += [{"role": m.role, "content": m.content} for m in req.messages]
        return {
            "model": req.model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }

    @staticmethod
    def _extract_text(data: dict, *, kw: dict) -> tuple[str, str | None]:
        choices = data.get("choices") or []
        if not choices:
            raise TransientAIError("openrouter returned no choices", **kw)
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish == "content_filter":
            raise RefusedError("openrouter content filter refusal", **kw)
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, list):
            # Multimodal "parts" form. Only the text parts are ours to read;
            # returning the list used to reach the Guard's `.strip()`.
            content = "".join(
                str(p.get("text", "")) for p in content if isinstance(p, dict)
            )
        if not isinstance(content, str) or not content:
            raise TransientAIError(f"openrouter returned empty content (finish={finish})", **kw)
        return content, finish

    # --- AIProvider protocol ---

    async def generate(self, req: GenRequest) -> GenResult:
        kw = {"task": req.task, "provider": self.name, "model": req.model}

        async def _call() -> GenResult:
            data = await self._chat(self._payload(req), task=req.task, model=req.model)
            text, finish = self._extract_text(data, kw=kw)
            return GenResult(text=text, finish_reason=finish)

        return await execute(
            _call, task=req.task, provider=self.name, model=req.model, limiter=self.limiter
        )

    async def generate_structured(self, req: GenRequest, schema: VersionedSchema) -> dict:
        # Delegates to the one Guard (§16) — no validation loop lives here.
        return await guarded_structured_call(self, req, schema)

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        raise AIError(
            "openrouter does not serve embeddings; the embedding model is pinned "
            "to the google provider (ai_interaction.md §3)",
            task="embeddings", provider=self.name, model=model,
        )

    # --- Guard hook (only structured.py calls this) ---

    async def raw_structured(
        self, req: GenRequest, schema: VersionedSchema, *, schema_in_prompt: bool
    ) -> str:
        kw = {"task": req.task, "provider": self.name, "model": req.model}
        use_prompt = schema_in_prompt or req.model in self._no_native_models
        if use_prompt and req.model in self._no_native_models and not schema_in_prompt:
            # Remembered as no-native: tell the Guard rather than silently diverge.
            raise NativeStructuredUnsupported(
                f"model {req.model} previously rejected response_format", **kw
            )
        adjusted = req
        payload = self._payload(req)
        if use_prompt:
            system = (req.system_prompt or "") + schema_prompt_block(schema)
            adjusted = GenRequest(
                task=req.task, model=req.model, system_prompt=system,
                messages=req.messages, temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            payload = self._payload(adjusted)
        else:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.name,
                    "strict": True,
                    "schema": schema.json_schema,
                },
            }
            # OpenRouter serves one model id from SEVERAL upstream providers and
            # picks per request. They do not all implement structured output,
            # and one that doesn't answers a schema-carrying request with a bare
            # `400 bad request` — which looks like our bug and is not. Observed:
            # `trait_extraction.v1` 400ing via AtlasCloud while the SAME model
            # served `persona_digest.v1` fine seconds apart.
            #
            # `require_parameters` tells OpenRouter to route only to providers
            # that support every parameter we sent, which is the difference
            # between "this model supports schemas" and "the provider we happened
            # to be routed to does".
            payload["provider"] = {"require_parameters": True}
        data = await self._chat(
            payload, task=req.task, model=req.model,
            sent_native_schema=not use_prompt,
        )
        text, _ = self._extract_text(data, kw=kw)
        return text
