# Module Plan — Trait Prompting & Persona Module

Status: planning locked 2026-09-01. Depends on: `module_1_data_collection.md` (traits, answers), `ai_interaction.md` (AIProvider, Structured Output Guard). Consumed by: Date Simulation, Chat.

---

## 1. Purpose

Turns a user's structured trait rows + raw answer text into a **persona snapshot**: a frozen, versioned system prompt that makes an AI agent talk and react like that user, plus the response schema the agent must obey. Everything downstream (dates, chat) consumes snapshots, never raw traits — that is the module boundary.

## 2. Core features and async operation

- **Persona compilation.** Triggered automatically after trait extraction finishes (`POST /profile/extract` in module 1), or on demand. Runs as a background task (one AI call + template assembly, seconds not minutes); a `status` field makes progress visible. The compiled snapshot is immutable once written.
- **Two-part prompt assembly** — deterministic where possible, AI only where needed:
  1. *Code-assembled sections (no AI call):* hard facts (name, age), trait rows grouped by category (interests + how approached, qualities, flaws, behavioral markers, conversational style), and **verbatim voice samples** — 3–5 excerpts pulled from the user's own answers, chosen by length and probe-area spread. The user's real writing is the few-shot material for voice mimicry; no model paraphrase in between.
  2. *AI-composed section (one structured call):* a "behavior digest" — how this person handles tense moments, flirts, supports, opens up — synthesized from situational answers (BQ3/BQ4 + dynamic). Validated through the Structured Output Guard, model recorded.
- **Response Schema Enforcer.** Owns the versioned `agent_response` schema. Every snapshot stores which schema version it was built for.
- **Calibration chat (optional, per decision log #5).** A live chat between the user and their own persona. Each persona message can be flagged "I'd never say that" with an optional correction. Flags feed the next compilation as explicit negative examples ("never phrase things like: …").

## 3. The `agent_response` schema — v1 (frozen here)

```json
{
  "schema_version": "agent_response.v1",
  "reply":           "string  — the spoken message",
  "state_of_mind":   "string  — one sentence, what they're thinking",
  "emotional_state": "string  — 1-3 words, e.g. 'amused, a bit nervous'",
  "connection":      "integer 0-100 — felt connection to the other person right now",
  "satisfaction":    "integer 0-100 — how much they're enjoying this right now",
  "wants_to_end":    "boolean — they would naturally wrap up the conversation now"
}
```

`wants_to_end` is new here: it gives dates a natural ending mechanism (see `date_simulation.md`) instead of always hitting the hard cap. Schema versions live in one code registry (`app/schemas/agent_response.py`); transcripts and snapshots store the version string, so a future v2 never breaks stored v1 data.

## 4. Data flow and database

```
answers + traits (module 1)                     calibration flags
        │                                              │
        ▼                                              ▼
  PersonaCompiler ──one structured AI call──► behavior digest
        │
        ▼
  persona_snapshots (INSERT, immutable)
        │
        ▼
  PersonaService.get_current_snapshot(user_id) ──► Date Simulation / Chat
```

```sql
CREATE TABLE persona_snapshots (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    version        INT  NOT NULL,                    -- 1, 2, 3… per user
    status         TEXT NOT NULL CHECK (status IN ('compiling','ready','failed')),
    system_prompt  TEXT,                             -- NULL until ready
    schema_version TEXT NOT NULL,                    -- 'agent_response.v1'
    traits_hash    TEXT NOT NULL,                    -- trait set it was built from
    source_trait_ids UUID[] NOT NULL,                -- provenance (principle 9)
    digest_model   TEXT,                             -- provider/model of the AI section
    error          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, version)
);

CREATE TABLE calibration_sessions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    snapshot_id  UUID NOT NULL REFERENCES persona_snapshots(id),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE calibration_messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES calibration_sessions(id) ON DELETE CASCADE,
    seq         INT  NOT NULL,
    sender      TEXT NOT NULL CHECK (sender IN ('user','persona')),
    text        TEXT NOT NULL,
    flagged     BOOLEAN NOT NULL DEFAULT FALSE,      -- "I'd never say that"
    correction  TEXT,                                -- optional: what they'd say instead
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, seq)
);
```

Why snapshots are immutable and versioned: a date transcript must stay explainable forever — "agent said X because snapshot v3 said Y." Recompiling creates v4; old transcripts keep pointing at v3. Staleness is detectable by comparing the snapshot's `traits_hash` to the live one.

**Revised 2026-09-02 (D-017).** `traits_hash` answers "has this person changed as a person"; it was never an answer to "is this prompt still true". The prompt also STATES facts — `You are <name>`, and the `WHO YOU ARE` block with age, gender, interested-in and city — and editing one of those left every agent using the old value silently. Two additions, both of which keep immutability intact: `identity_drifted()` reads the frozen facts block back out of the prompt and compares it to the live user, and `refresh_identity()` issues a **new version** whose head and facts block are re-stated and whose every other byte is carried over unchanged — including the verbatim excerpts, which are the person's own words and are never rewritten. It costs **no model call**: a rename is not a change of character, and re-running the digest would spend a call to re-derive what did not change. Triggered by `PATCH /me` and by reconciliation step 5.

## 5. Interfaces (module boundary)

```python
class PersonaService(Protocol):
    async def compile(self, user_id: UUID) -> UUID:            # returns snapshot_id; background task
    async def get_current_snapshot(self, user_id: UUID) -> PersonaSnapshot | None:
        # newest 'ready' snapshot; None => user not simulatable yet (gate, principle 11)
    async def flag_calibration_message(self, message_id: UUID, correction: str | None) -> None:
```

- Date Simulation and Chat call **only** `get_current_snapshot`. They never read `traits` or `answers` directly.
- `None` from `get_current_snapshot` is a hard gate: a user with no ready snapshot cannot appear as a candidate (Candidate Matching checks this) — an action never runs degraded on a missing data source.

## 6. Endpoints

| Endpoint | Behavior |
|---|---|
| `POST /persona/compile` | Kicks compilation; returns `{snapshot_id, status}` immediately. |
| `GET /persona/current` | Current snapshot status + metadata (never the raw system prompt — that stays server-side). |
| `POST /calibration/sessions` | Start a "meet your AI self" chat against the current snapshot. |
| `POST /calibration/sessions/{id}/messages` | User message in → persona reply out (one AI call, same structured schema, metadata stored but not shown here). |
| `POST /calibration/messages/{id}/flag` | Mark "I'd never say that" + optional correction; logged as a trait/voice event. |

## 7. Technical decisions (trades named)

1. **Voice via verbatim excerpts, not summaries.** The model paraphrasing the user's writing before mimicry would launder out the voice. Cost: longer prompts. Accepted — persona fidelity is load-bearing assumption #1.
2. **One AI call per compilation.** Only the behavior digest needs a model; everything else is a template. Cheap, fast, and the deterministic parts can't drift.
3. **Immutable versioned snapshots.** Cost: storage of old prompts. Accepted for explainability of every transcript ever produced.
4. **`wants_to_end` added to schema v1.** Cost: one more field agents must fill. Accepted — without it every date artificially runs to the cap and endings are always awkward mid-conversation cuts.
5. **System prompt never leaves the server.** The UI sees snapshot status and metadata only. Cost: harder to debug from the client. Accepted — the prompt embeds the user's raw intimate answers.

Logging obligations (§7): compilation logs trait count, source answer IDs, digest model, and outcome; a failed compilation records the error and leaves the previous snapshot current. Calibration flags log which snapshot they criticize.

## Locked by this document

1. `agent_response.v1` schema, verbatim, including `wants_to_end`.
2. Snapshot immutability + per-user versioning; staleness via `traits_hash` for the persona, and via the frozen facts block for the stated identity (D-017) — a rename re-issues the snapshot for free rather than editing it.
3. Compilation = deterministic template + one structured AI call for the behavior digest.
4. Voice mimicry uses verbatim answer excerpts.
5. `PersonaService` as the only interface downstream modules may use; `None` snapshot = not simulatable.

## Open for later modules

- How Date Simulation composes two snapshots into a turn loop (next: `candidate_matching.md`, then `date_simulation.md`).
- Whether calibration flags should also auto-adjust trait confidence (deferred until there's calibration data to look at).
