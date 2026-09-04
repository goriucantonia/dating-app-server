# Module Plan — AI Interaction Module (Foundational Service)

Status: planning locked 2026-09-01. Moved out of `module_1_data_collection.md` into its own document — this is not a feature module but the foundational service every other module calls. Consumers: Data Collection (trait extraction, dispute follow-ups), Trait Persona (behavior digest, calibration replies), Candidate Matching (embeddings), Date Simulation (scenarios, agent turns, judging), Chat (persona replies, compaction).

**Requirement (owner, 2026-09-01): the implementation supports both Google AI Studio and free models via OpenRouter from day one**, behind one clean interface with a provider-switching mechanism, so models can be swapped without touching core logic. *(This revised the earlier "one provider active, switching deferred" decision — the new information was the owner's explicit requirement to run free OpenRouter models now and paid Google models later. Recorded in the `project_description.md` decision log.)*

---

## 1. Layout

```
app/ai/
  base.py         # AIProvider protocol, request/result dataclasses, typed errors
  google.py       # GoogleProvider  — google-genai SDK; native structured output; embeddings
  openrouter.py   # OpenRouterProvider — OpenAI-compatible REST; free models
  litellm_provider.py  # LiteLLMProvider — one interface to everything else
  registry.py     # builds provider instances from config; name -> instance
  routing.py      # task -> (provider, model) resolution
  structured.py   # Structured Output Guard — the single choke point (principle 16)
  resilience.py   # retry, backoff, per-provider rate limiting, give-up conditions
```

**Revision 2026-09-04 — a third provider, `litellm_provider.py`.** The owner asked for models to be callable through the LiteLLM interface. It is added ALONGSIDE the two hand-written clients, not in place of them: those two are the ones this project measured, pinned and closed its gates on, and each encodes knowledge a generic client does not (OpenRouter's `"Provider returned error"` being transient, D-008; `gemini-embedding-001` returning unnormalized vectors at non-default dimensionality). LiteLLM is the door to everything else — Anthropic, OpenAI, Azure, Bedrock, Vertex, Groq, Together, Mistral, DeepSeek, Ollama, vLLM — reached by writing its model string into the routing config.

**Named departure from this layout block:** the file is `litellm_provider.py`, not `litellm.py`. A module named `app/ai/litellm.py` that does `import litellm` works under Python 3's absolute imports, but it puts two different `litellm` names in every traceback and sits one `from . import litellm` away from a confusing bug.

## 2. The interface

```python
class AIProvider(Protocol):
    name: str

    async def generate(self, req: GenRequest) -> GenResult: ...
    # req: system_prompt, messages, model, temperature, max_tokens

    async def generate_structured(self, req: GenRequest, schema: VersionedSchema) -> dict: ...
    # returns a dict already validated against schema; raises StructuredOutputError on give-up

    async def embed(self, texts: list[str], model: str) -> list[list[float]]: ...
```

Core logic (persona compilation, date simulation, judging, dispute follow-ups) depends **only** on this protocol. No module imports `google.py` or `openrouter.py` directly — swapping a provider is a config edit. Every call site passes a `task` name; the module resolves provider + model itself.

## 3. Per-task routing (the switching mechanism)

One config file maps each task to a provider + model. Changing a model = editing one line, no code:

```yaml
ai:
  providers:
    google:     { api_key_env: GOOGLE_AI_API_KEY }
    openrouter: { api_key_env: OPENROUTER_API_KEY }
    litellm:    { api_key_env: LITELLM_API_KEY }   # optional; see below

  embeddings:                       # pinned SEPARATELY from chat models:
    provider: google                # every stored vector must come from the same model
    model: gemini-embedding-001     # or similarity comparisons are meaningless.
                                    # Changing this = migration + re-embed all users.
```

*(Revision 2026-09-01, per principle 23: the pin was `text-embedding-004`, but Google's API now returns 404 for it — the model was withdrawn. New pin: `gemini-embedding-001`, the current stable embedding model, with **768 output dimensions requested explicitly** (its default is 3072) so `vector(768)` and every stored vector stay consistent. Vectors are L2-normalized by the provider wrapper — at non-default dimensionality the API returns unnormalized vectors, and normalized storage keeps cosine and dot-product interchangeable. The swap happened before any vector was ever stored, so no migration or re-embed was needed.)*

```yaml
  routing:
    dispute_followups:   { provider: openrouter, model: "free-model-of-choice" }
    trait_extraction:    { provider: google,     model: "gemini-flash" }
    persona_digest:      { provider: google,     model: "gemini-flash" }
    scenario_generation: { provider: openrouter, model: "free-model-of-choice" }
    date_simulation:     { provider: openrouter, model: "free-model-of-choice" }
    judging:             { provider: google,     model: "gemini-flash" }
    chat_reply:          { provider: openrouter, model: "free-model-of-choice" }
    chat_compaction:     { provider: openrouter, model: "free-model-of-choice" }
```

### The `litellm` provider (added 2026-09-04)

**The model string IS the routing.** LiteLLM names a model `<upstream>/<model>`, so no new config concept was needed:

```yaml
  routing:
    judging:         { provider: litellm, model: "anthropic/claude-sonnet-4-5" }
    chat_reply:      { provider: litellm, model: "groq/llama-3.3-70b-versatile" }
    date_simulation: { provider: litellm, model: "ollama/llama3.1" }
```

**Its API key is optional, and that is the difference from the other two.** LiteLLM resolves a key PER UPSTREAM from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, …), so an empty `LITELLM_API_KEY` is a normal state rather than a missing credential — the registry logs it at INFO with `key_resolution: provider_env` instead of the WARNING the other two get, because a warning that fires every boot is a warning people learn to skip. Fill it only to force one key across every litellm route, which is the shape for a LiteLLM proxy or a single upstream.

**`api_base` points it at a self-hosted endpoint** (a LiteLLM proxy, Ollama, vLLM, an OpenAI-compatible gateway). Only this provider reads it; the registry refuses it on `google` or `openrouter` at startup rather than ignoring it, because an ignored `api_base` reads as though a provider were pointed somewhere it is not.

**Structured output needs no special handling.** Where the upstream implements a JSON schema, LiteLLM's native mode is used; where it does not, the same Guard falls back to embedding the schema in the prompt (§4.1). A model missing from LiteLLM's capability map is tried natively ONCE and remembered if it refuses — a wrong "no" would silently downgrade every call for a model that does support schemas, where a wrong "yes" costs one round-trip.

**Embeddings work through it too, and the `embeddings:` pin accepts it** — `{ provider: litellm, model: "gemini/gemini-embedding-001" }`. Witnessed 2026-09-04 through the real `TaskRouter`: 768 dimensions, L2 norm 1.000000. But it is not the free edit a chat route is, and three things follow from changing it:

1. **The model must return 768-dimensional vectors**, because `profile_embeddings` is `vector(768)`. The provider requests `dimensions: 768` and REFUSES a wrong-width vector rather than letting it reach the database, where the failure would be an asyncpg error naming a column instead of a model. `openai/text-embedding-3-*` truncate on request; models that cannot are unusable here without widening the column.
2. **Every user is re-embedded, even on a switch that changes nothing numerically.** `profile_embeddings.embedding_model` stores `"<provider>/<model>"`, so moving `gemini-embedding-001` from `google` to `litellm` rewrites that string and every row reads as mismatched. `check_embedding_models` (reconciliation step 5) logs each at ERROR, blanks its `traits_hash`, and `refresh_embeddings` regenerates it before it is ever compared — the Step 15 AC6 machinery, working as designed. On the current volume that is ~59 embed calls to reproduce identical vectors.
3. **Mixing is impossible by construction**, which is why this pin is separate from routing at all: every stored vector must come from one model or cosine similarity compares incomparable things.

**Witnessed 2026-09-04** against real providers: a plain call and a native structured call through `gemini/gemini-flash-latest` (validated dict, `mode: native`), a 768-dimension embedding at L2 norm 1.000000, a real 404 classified fatal and not retried, the env-resolved key path proven through `openrouter/…`, and an OpenRouter upstream fault correctly retried three times as transient before an honest give-up (D-031).

Every stored artifact produced by a model (`traits.extracted_by`, persona digests, transcripts, judge scores) records which provider/model made it, so results stay explainable after any swap.

**Model selection status (owner, 2026-09-01):** the `free-model-of-choice` slots are deliberately unfilled — concrete models are chosen after initial tests, and the paid-balance question (which changes the available quota by an order of magnitude) is decided from those same tests. Two gates before the first real analysis:
1. The **quota fit check**: calls-per-analysis (~190 at 2 dates/candidate) must fit the chosen providers' per-minute *and* per-day limits — a spreadsheet against the published caps, verified by one full end-to-end run.
2. The **fidelity transfer check**: persona voice was hand-validated once, but on a model that is not necessarily the one routed for `date_simulation`; re-validate on the actual routed model (chat with your own persona through it, count the lines you'd never say) before trusting any date it produces.

**Language:** all prompts, routing, system content, and model outputs are English-only (owner decision, 2026-09-01 — maximizes model performance). User answers are expected in English; no localization anywhere in the pipeline.

## 4. Structured Output Guard (single choke point)

All structured generation flows through one wrapper, per principle 16 — never per-module JSON parsing:

1. Prefer the provider's native mode (Google `response_schema`; OpenRouter `response_format: json_schema` where the model supports it; otherwise schema embedded in the prompt).
2. Validate the raw output against the **versioned** JSON schema (`jsonschema`).
3. On failure: one repair prompt containing the validation error; retry up to 3 times total.
4. Give-up condition (§17): raise `StructuredOutputError` carrying the raw output; the caller's checkpoint layer decides what "incomplete" means for it. Never a silent default.

Free OpenRouter models are the reason this guard is mandatory: they hold JSON contracts less reliably than Gemini, and the guard is what makes them safely swappable anyway.

## 5. Resilience layer

- Exponential backoff on 429/5xx with per-provider rate limiters (free tiers throttle aggressively).
- Retries capped (give-up after 3), errors surfaced as typed exceptions so the Date Simulation Module's checkpointing can resume rather than restart.
- Every call logs: task, provider, model, attempt number, latency, outcome (`ok` / `malformed` / `rate_limited` / `refused` / `gave_up`). Refusal and failure paths are the ones that must log (§7).

## Locked by this document

1. Dual-provider implementation (Google AI Studio + OpenRouter) behind the `AIProvider` protocol; no module touches a concrete provider.
2. Per-task routing config, verbatim shape above; the embedding model pinned separately from all chat/generation models.
3. One Structured Output Guard for the whole app: native-mode-first, validate, 3-attempt repair, typed give-up.
4. Resilience layer semantics and the mandatory per-call log line.
5. Every model-produced artifact stores its provider + model.
