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
  registry.py     # builds provider instances from config; name -> instance
  routing.py      # task -> (provider, model) resolution
  structured.py   # Structured Output Guard — the single choke point (principle 16)
  resilience.py   # retry, backoff, per-provider rate limiting, give-up conditions
```

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

  embeddings:                       # pinned SEPARATELY from chat models:
    provider: google                # every stored vector must come from the same model
    model: text-embedding-004       # or similarity comparisons are meaningless.
                                    # Changing this = migration + re-embed all users.
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
