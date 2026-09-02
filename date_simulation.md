# Module Plan — Date Simulation Module

Status: planning locked 2026-09-01. Depends on: `candidate_matching.md` (analysis + frozen snapshots), `trait_persona.md` (`agent_response.v1`), `ai_interaction.md` (AIProvider, Structured Output Guard). Consumed by: Chat (date digests), UI results screens.

---

## 1. Purpose

For a `matched` analysis: draw ONE date scenario at random and run every candidate against it, run the turn-by-turn agent-vs-agent simulations with event injection and per-turn state tracking, survive failures via checkpointing, then judge every completed date and compute per-candidate match scores. Drives `analyses.status` from `matched → simulating → complete | failed`.

## 2. Core features and async operation

- **Everything is a background pipeline.** `POST /analyses/{id}/simulate` returns immediately; the pipeline runs as an in-process asyncio task. **No Celery/Redis this phase** *(trade: a server restart kills in-flight tasks; accepted because every turn is checkpointed in Postgres and a startup reconciliation pass re-launches any analysis stuck in `simulating` — the work resumes, only the process-local task is lost)*.
- **Sequential execution, one date at a time,** one active simulation per user (enforced upstream). Free-tier rate limits make parallel dates counterproductive — parallelism would just spread the same throughput across more 429s. Global semaphore of 2 concurrent pipelines across all users.
- **Scenario generation** (one structured AI call per **ANALYSIS**): input = **one archetype drawn at random in code** from the catalogue in `app/date_archetypes.py`; output = **1 setting** `{setting_name, description, sensory_details, archetype, possible_events[4-6]}`, stored on `analyses.scenarios` and copied onto every date. **Every candidate in the analysis is run against the same setting.** The call receives no names, no ages, no interests and no traits — the fixture is deliberately nobody's.
  - **REVISED 2026-09-02 (owner decision): one random scenario per analysis, identical across all candidates.** It used to be one call per candidate, anchored in that pair's shared interests, with an empty-intersection fallback that built the evening around the candidate's world. Three candidates therefore got three different evenings, and `candidate_scores` then ranked those three numbers side by side as if they were the same measurement — they were not. A judge scoring `conversational_flow` at a car meet and at a bookshop is running two experiments and reporting one league table. The requirement is a **controlled comparison**: same fixture, three different people, so the differences in the scores are differences between the PEOPLE.
  - **Cost of the change, named:** nobody gets an evening built around their own interests any more. That is the price of a comparison that means anything, and it is the reason the anchor moved from interests to a neutral archetype — anchoring on anyone would hand one candidate a home fixture and the score would measure the anchor.
  - **The draw is in code, not in the model.** A model asked for "a first date setting" produces a coffee shop, then a wine bar, then a coffee shop again; temperature does not fix that. The catalogue is 16 written-down archetypes and the last `RECENT_ARCHETYPES_AVOIDED = 3` a user has had are excluded from their next draw, so a repeat inside four consecutive analyses is impossible rather than merely unlikely. The model still echoes the drawn key back into `archetype`, and `generate_scenarios` verifies and repairs it — provenance, not trust (§9). Schema bumped to `date_scenarios.v3`; `anchored_in_interest` is gone and `archetype` replaced it.
- **Turn loop per date** (cap: **16 agent turns, 8 each** — REVISED 2026-09-01, owner decision; was 30 messages total counting environment rows):
  1. Compose agent context: their frozen persona snapshot's system prompt + scenario description + a date-role preamble ("you are on a first date with…") + full transcript so far (a whole date is at most 19 rows and fits any context window with room to spare — no summarization needed inside a date).
  2. One structured call → `agent_response.v1` through the Structured Output Guard.
  3. Persist the message row (checkpoint) **before** advancing the turn.
  4. **Event injection:** before each turn, roll p = 0.15 (max 3 events/date, never two in a row); on hit, insert an `environment` message drawn from the scenario's `possible_events`, which both agents see as context.
  5. **Natural ending:** when both agents' latest `wants_to_end` are true, run one final closing exchange and stop. Otherwise stop at the cap.
- **Failure handling per turn (give-up ladder, §17):** resilience layer retries transient errors (3 attempts, backoff). Turn still failing → date marked `incomplete` at its last checkpointed message, pipeline moves to the next date. **Every date with a transcript is judged, however short** — an `incomplete` one is flagged partial and weighted 0.5. *(**REVISED 2026-09-02, owner decision: the ≥10-turn threshold is REMOVED.** It excluded a sub-10-turn date from scoring entirely and showed it as failed. The rule answered a question about DEPTH with a rule about ADMISSION: a four-turn date is not unjudgeable, it is thinly evidenced, and the honest response to thin evidence is a reading that claims less — not a refusal to read, delivered to someone who had just watched that date happen. Depth is now REPORTED: `judge_rubric.v2` asks the judge for its own `confidence` (0-100) and an `evidence_note`, stored beside the score and never multiplied into it. The only exclusion left is a date nobody spoke on, where there is literally no text to read. Previously revised 2026-09-01 from rows to turns; that unit distinction survives at the new floor.)* The analysis never dies because one date did — it completes with whatever finished, and says so.
- **Judge pipeline** (after all of a candidate's dates finish): per completed date, one structured call scoring fixed criteria; the **final number is computed in code**, not asked from the model:
  ```
  judge output (0-100 each): trait_alignment, conversational_flow,
                             mutual_engagement, clash_severity
       plus: clicked_subjects[], clashes[{user_trait, candidate_trait, moment}],
             per_peer_summary, verdict_summary,
             confidence (0-100), evidence_note   ← v2, reported not scored
  date_score      = 0.30*trait_alignment + 0.30*conversational_flow
                  + 0.25*mutual_engagement + 0.15*(100 - clash_severity)
  candidate_score = mean(date_scores) — incomplete-but-judged dates weighted 0.5
  ```
  Deterministic aggregation keeps scores explainable (owner's answer #5: strict criteria checks); the per-turn `connection`/`satisfaction` curves come free from stored messages — analytics reads them, no extra calls. Rubric text is versioned (`judge_rubric.v2`); judge model + rubric version stored per evaluation. **v2 also asks the judge for `confidence` (0-100) and an `evidence_note`** — stored beside the score, never folded into it, because one number meaning both "how it went" and "how much we saw" is a number nobody can read.
- **Progress reporting:** the pipeline updates a `progress` JSONB on the analysis after every stage (`"Simulating date 2 of 6 — at the car meet…"`). UI polls `GET /analyses/{id}`; **no SSE/WebSocket this phase** *(trade: up-to-3s staleness; accepted — polling one row is simpler than a push channel across Flutter web/mobile/desktop, and nothing here is realtime)*.

## 3. Data flow and database

```
analyses(matched) ─► scenario gen (1 call/ANALYSIS, random archetype) ─► dates(pending)
  └► analyses.scenarios  ── copied onto every candidate's date row
  └► per date: turn loop ──checkpoint every message──► date_messages
        └► judge (1 call/date) ─► date_evaluations ─► candidate_scores ─► analyses(complete)
```

```sql
CREATE TABLE dates (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id        UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    candidate_user_id  UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ordinal            INT  NOT NULL,               -- always 1 since 2026-09-01;
                                                   -- the column stays so the cap can
                                                   -- move again without a migration
    scenario           JSONB NOT NULL,              -- setting_name, description, sensory, events
    status             TEXT NOT NULL CHECK (status IN
                         ('pending','running','complete','incomplete','failed')),
    user_snapshot_id      UUID NOT NULL REFERENCES persona_snapshots(id),
    candidate_snapshot_id UUID NOT NULL REFERENCES persona_snapshots(id),
    schema_version     TEXT NOT NULL,               -- 'agent_response.v1'
    error              TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at        TIMESTAMPTZ,
    UNIQUE (analysis_id, candidate_user_id, ordinal)
);

CREATE TABLE date_messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date_id     UUID NOT NULL REFERENCES dates(id) ON DELETE CASCADE,
    seq         INT  NOT NULL,
    speaker     TEXT NOT NULL CHECK (speaker IN ('user_agent','candidate_agent','environment')),
    reply       TEXT NOT NULL,                      -- spoken text, or the event description
    state       JSONB,                              -- state_of_mind, emotional_state,
                                                    -- connection, satisfaction, wants_to_end
                                                    -- (NULL for environment rows)
    provider    TEXT, model_id TEXT,                -- who generated this turn
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (date_id, seq)                           -- the checkpoint invariant
);

CREATE TABLE date_evaluations (
    date_id        UUID PRIMARY KEY REFERENCES dates(id) ON DELETE CASCADE,
    criteria       JSONB NOT NULL,                  -- the four 0-100 judge scores
    clicked        JSONB NOT NULL,                  -- clicked_subjects[]
    clashes        JSONB NOT NULL,                  -- [{user_trait, candidate_trait, moment}]
    per_peer       JSONB NOT NULL,                  -- summary per participant
    verdict        TEXT  NOT NULL,
    date_score     REAL  NOT NULL,                  -- computed in code from criteria
    is_partial     BOOLEAN NOT NULL DEFAULT FALSE,  -- judged from an incomplete date
    confidence     INTEGER,                         -- 0-100, the judge's own (v2, migration 0012; NULL on v1 rows)
    evidence_note  TEXT,                            -- what the transcript could and could not show
    judge_provider TEXT NOT NULL, judge_model TEXT NOT NULL,
    rubric_version TEXT NOT NULL,                   -- 'judge_rubric.v2'
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE candidate_scores (
    analysis_id       UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    candidate_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    final_score       REAL NOT NULL,                -- weighted mean of date_scores
    dates_completed   INT  NOT NULL,
    dates_incomplete  INT  NOT NULL,
    PRIMARY KEY (analysis_id, candidate_user_id)
);

-- analyses (from candidate_matching.md) gains:  progress JSONB
--                                             scenarios JSONB  -- the ONE fixture
--   every candidate in the analysis is run against (migration 0011, 2026-09-02).
--   A JSONB array, one entry per date each candidate gets. NULL on analyses that
--   ran before the shared fixture existed -- they genuinely had none, and no
--   backfill is attempted because writing one would be inventing history.
```

Resume logic: on pipeline (re)start, for each date by status — `pending` → run from scratch; `running` → continue from `max(seq)` (the transcript is the state; nothing else to restore); `complete/incomplete` with no evaluation → judge it. Idempotent by construction: re-running a finished stage is a no-op because the rows already exist.

## 4. Interfaces (module boundary)

```python
class SimulationService(Protocol):
    async def start(self, analysis_id: UUID) -> None:          # 409 unless analysis is 'matched'
    async def get_results(self, analysis_id: UUID) -> AnalysisResults:
        # dates + transcripts + evaluations + candidate_scores; partial while running

class DateDigest(Protocol):                                     # for the Chat module
    async def digest(self, analysis_id: UUID, candidate_user_id: UUID) -> str:
        # compact factual summary of what happened on their dates (settings, clicked
        # subjects, memorable events) — compiled from evaluations, no new AI call
```

Chat consumes `DateDigest` only — it never reads raw `date_messages`.

## 5. Endpoints

| Endpoint | Behavior |
|---|---|
| `POST /analyses/{id}/simulate` | Start the pipeline (auto-chained after matching by default; explicit endpoint kept for retry). **Revised 2026-09-01 (Step 13):** also accepted on a `failed` analysis that got as far as having candidates — the pipeline resumes from its checkpointed rows (finished dates are no-ops on re-run), which is what makes the UI's "picks up where it stopped" true. A `failed` analysis with NO candidates died in matching; there is nothing to resume, and the 409 says to start a new one. Logged as `simulation_requested … resumed_after_failure: true` with the stage it died at. |
| `GET /analyses/{id}` | (matching module's endpoint) now also carries `progress` and, when done, scores. |
| `GET /analyses/{id}/dates` | All dates with status + evaluations. **Since 2026-09-01 (Step 13)** each date also carries `ended_by` (`mutual_wants_to_end` / `cap` / null) computed by the SAME `ended_by()` rule the loop used to stop it — the UI says how a date ended, it never re-derives it. |
| `GET /dates/{id}/transcript` | Messages incl. `state` metadata and environment rows — feeds the transcript viewer. **Since 2026-09-01 (Step 13)** also `analysis_id` (so the viewer watches the one analysis poller for "other dates still running") and `ended_by`. |

## 6. Technical decisions (trades named)

1. **In-process async tasks + DB checkpoints instead of a task queue.** Cost: restart kills in-flight work (resumed by reconciliation, not lost). Accepted: a queue is a second infrastructure to run in Docker for a single-host friends-scale app.
2. **Sequential dates.** Cost: wall-clock time (a full analysis ≈ 3 dates × 16 turns + overhead — **54 model calls**, measured at roughly 6 minutes on the free tier; it was 6 dates of 30 messages before the 2026-09-01 revisions). Accepted: background + notification design already assumes the user leaves; parallelism buys nothing under a shared rate limit.
3. **Score computed in code from judge criteria.** Cost: rubric weights are opinions. Accepted: weights are visible, versioned, and identical for everyone — the alternative (model picks a number) is neither.
4. **Events from pre-generated scenario lists.** Cost: less surprise than live-generated events. Accepted: one fewer call per event, and the scenario call produces better-anchored events ("a vintage Mustang pulls up") than a mid-date generic generator.
5. **Judge sees only the transcript, not the trait profiles' full text** — it receives the transcript plus both users' trait *labels* (for clash attribution). Cost: judge can't cite unexpressed traits. Accepted deliberately: the judge scores what happened on the date, not what the profiles predicted; that separation is what makes a surprising date result informative.
6. **Polling, no push channel.** Named above.

Logging obligations (§7): every turn logs date_id/seq/provider/model/attempt/outcome; event injections log the roll and the chosen event; date endings log which mechanism ended it (`mutual_wants_to_end` vs `cap`); the judge logs rubric version and raw criteria; the pipeline logs every status transition with its reason. From logs alone it must be reconstructable why any date ended, scored, or failed as it did.

## Locked by this document

1. Caps: **1 date/candidate, max 3 per analysis, 16 agent TURNS per date** (all three REVISED 2026-09-01, owner decision — were 2/candidate, 6/analysis, and a 30-message cap that counted environment rows), events p=0.15 max 3 never consecutive and NOT charged against the turn cap, global concurrency 2. A transcript is therefore at most 19 rows. **The cost of the revision, named:** a candidate's score was the mean of two independent readings and is now a single one, so one odd evening or one wobbly judge call is no longer averaged down. Accepted for roughly half the model calls per analysis (~177 → ~90).
2. Natural ending via mutual `wants_to_end`; hard cap otherwise.
3. Checkpoint-per-message; resume semantics; incomplete-date policy (**every date with a transcript is judged**; an `incomplete` one is flagged partial and weighted 0.5; the only exclusion is a date nobody spoke on — REVISED 2026-09-02, owner decision, was "≥10 agent TURNS or excluded"; revised from rows to turns 2026-09-01).
4. `judge_rubric.v2` criteria and the code-side scoring formula, verbatim. **The four criteria and their four weights are unchanged from v1** — v2 changed what the judge is TOLD (judge every date, report your own confidence) and added `confidence` + `evidence_note`. A v1 score and a v2 score are the same arithmetic over the same numbers, so the bump did not re-base a single stored value.
5. Scenario generation contract: **one random archetype per analysis, the same setting for every candidate**, drawn in code from `app/date_archetypes.py`, with no personal detail in the call (REVISED 2026-09-02, owner decision — was one interest-anchored call per candidate with an empty-intersection fallback; both are gone, along with `anchored_in_interest`).
6. In-process tasks, sequential execution, polling.
7. Full schema: `dates`, `date_messages`, `date_evaluations`, `candidate_scores`, `analyses.progress`, `analyses.scenarios`.

## Open for the next module (Chat)

- Digest content shape and how the persona is instructed to treat simulated history.
- Context management for open-ended chat length (dates are capped; chats aren't).
