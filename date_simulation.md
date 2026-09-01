# Module Plan — Date Simulation Module

Status: planning locked 2026-09-01. Depends on: `candidate_matching.md` (analysis + frozen snapshots), `trait_persona.md` (`agent_response.v1`), `ai_interaction.md` (AIProvider, Structured Output Guard). Consumed by: Chat (date digests), UI results screens.

---

## 1. Purpose

For a `matched` analysis: generate date scenarios per candidate, run the turn-by-turn agent-vs-agent simulations with event injection and per-turn state tracking, survive failures via checkpointing, then judge every completed date and compute per-candidate match scores. Drives `analyses.status` from `matched → simulating → complete | failed`.

## 2. Core features and async operation

- **Everything is a background pipeline.** `POST /analyses/{id}/simulate` returns immediately; the pipeline runs as an in-process asyncio task. **No Celery/Redis this phase** *(trade: a server restart kills in-flight tasks; accepted because every turn is checkpointed in Postgres and a startup reconciliation pass re-launches any analysis stuck in `simulating` — the work resumes, only the process-local task is lost)*.
- **Sequential execution, one date at a time,** one active simulation per user (enforced upstream). Free-tier rate limits make parallel dates counterproductive — parallelism would just spread the same throughput across more 429s. Global semaphore of 2 concurrent pipelines across all users.
- **Scenario generation** (one structured AI call per candidate): input = `shared_interests` + both users' interest traits; output = **1 setting** `{setting_name, description, sensory_details, anchored_in_interest, possible_events[4-6]}`. **Empty-intersection fallback** (flagged open in `candidate_matching.md`): when `shared_interests` is empty, the setting is built around the **candidate's** interests — matching the Source of Truth's "aligned with at least one of the individuals."
  - **REVISED 2026-09-01 (owner decision).** This read "2 distinct settings" and "one setting anchored in each person's interests (one date 'hers', one 'his')", because the cap was two dates per candidate. The owner changed that to **one date per candidate**, which made the two-settings instruction impossible to state. The fallback anchors on the CANDIDATE because a requester working through a full pool then sees three different people's worlds rather than three versions of their own. Schema bumped to `date_scenarios.v2`.
- **Turn loop per date** (cap: **16 agent turns, 8 each** — REVISED 2026-09-01, owner decision; was 30 messages total counting environment rows):
  1. Compose agent context: their frozen persona snapshot's system prompt + scenario description + a date-role preamble ("you are on a first date with…") + full transcript so far (a whole date is at most 19 rows and fits any context window with room to spare — no summarization needed inside a date).
  2. One structured call → `agent_response.v1` through the Structured Output Guard.
  3. Persist the message row (checkpoint) **before** advancing the turn.
  4. **Event injection:** before each turn, roll p = 0.15 (max 3 events/date, never two in a row); on hit, insert an `environment` message drawn from the scenario's `possible_events`, which both agents see as context.
  5. **Natural ending:** when both agents' latest `wants_to_end` are true, run one final closing exchange and stop. Otherwise stop at the cap.
- **Failure handling per turn (give-up ladder, §17):** resilience layer retries transient errors (3 attempts, backoff). Turn still failing → date marked `incomplete` at its last checkpointed message, pipeline moves to the next date. An `incomplete` date with ≥10 messages is still judged (flagged partial); under 10, it's excluded from scoring and shown as failed. The analysis never dies because one date did — it completes with whatever finished, and says so.
- **Judge pipeline** (after all of a candidate's dates finish): per completed date, one structured call scoring fixed criteria; the **final number is computed in code**, not asked from the model:
  ```
  judge output (0-100 each): trait_alignment, conversational_flow,
                             mutual_engagement, clash_severity
       plus: clicked_subjects[], clashes[{user_trait, candidate_trait, moment}],
             per_peer_summary, verdict_summary
  date_score      = 0.30*trait_alignment + 0.30*conversational_flow
                  + 0.25*mutual_engagement + 0.15*(100 - clash_severity)
  candidate_score = mean(date_scores) — incomplete-but-judged dates weighted 0.5
  ```
  Deterministic aggregation keeps scores explainable (owner's answer #5: strict criteria checks); the per-turn `connection`/`satisfaction` curves come free from stored messages — analytics reads them, no extra calls. Rubric text is versioned (`judge_rubric.v1`); judge model + rubric version stored per evaluation.
- **Progress reporting:** the pipeline updates a `progress` JSONB on the analysis after every stage (`"Simulating date 2 of 6 — at the car meet…"`). UI polls `GET /analyses/{id}`; **no SSE/WebSocket this phase** *(trade: up-to-3s staleness; accepted — polling one row is simpler than a push channel across Flutter web/mobile/desktop, and nothing here is realtime)*.

## 3. Data flow and database

```
analyses(matched) ─► scenario gen (1 call/candidate) ─► dates(pending)
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
    judge_provider TEXT NOT NULL, judge_model TEXT NOT NULL,
    rubric_version TEXT NOT NULL,                   -- 'judge_rubric.v1'
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
| `POST /analyses/{id}/simulate` | Start the pipeline (auto-chained after matching by default; explicit endpoint kept for retry). |
| `GET /analyses/{id}` | (matching module's endpoint) now also carries `progress` and, when done, scores. |
| `GET /analyses/{id}/dates` | All dates with status + evaluations. |
| `GET /dates/{id}/transcript` | Messages incl. `state` metadata and environment rows — feeds the transcript viewer. |

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
3. Checkpoint-per-message; resume semantics; incomplete-date policy (≥10 messages → judged as partial at 0.5 weight, else excluded).
4. `judge_rubric.v1` criteria and the code-side scoring formula, verbatim.
5. Scenario generation contract incl. empty-intersection fallback.
6. In-process tasks, sequential execution, polling.
7. Full schema: `dates`, `date_messages`, `date_evaluations`, `candidate_scores`, `analyses.progress`.

## Open for the next module (Chat)

- Digest content shape and how the persona is instructed to treat simulated history.
- Context management for open-ended chat length (dates are capped; chats aren't).
