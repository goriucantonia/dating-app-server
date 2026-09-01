# Module Plan — Module 1: Data Collection & Dynamic Profiling

Status: planning locked 2026-09-01 (revised same day: AI Interaction split out to `ai_interaction.md`; dynamic AI-generated questions replaced by a curated question pool — revisions named inline). Source of truth: `user_perspective.md`, `project_description.md`, `technical_details.md`.

---

## A1. Registration form (exact structure)

Collected once at account creation. These are hard facts and hard filters — never inferred, never sent to the AI as traits.

| Field | Type | Rules |
|---|---|---|
| `email` | text | unique, verified format only (no email verification this phase — friends pool) |
| `password` | text | min 8 chars, stored as `password_hash` (bcrypt) |
| `display_name` | text | 1–50 chars |
| `birth_date` | date | must yield age ≥ 18; age computed, never stored |
| `gender` | choice | `man` / `woman` / `nonbinary` / `other` |
| `interested_in` | multi-choice | one or more of the same set; used in the mutual-fit hard filter |
| `age_pref_min`, `age_pref_max` | int | 18 ≤ min ≤ max |
| `city`, `country` | text | informational this phase; **distance filtering is deferred** — the friends pool doesn't need geo-matching, and geocoding is a whole subsystem. The columns exist so the data is there when it does. *(Trade: a match may live far away; accepted — everyone in the pool knows each other.)* |
| `opt_in` | toggle | default **off**; one-line description next to it |

Not on the form: no photos this phase (nothing in the product consumes them), no phone number, no email verification.

## A2. The 5 baseline questions (final)

Five deep open-ended questions replace the earlier "~20 questions" draft. **Revision named per principle 23:** owner decision 2026-09-01 — depth over count; each answer probes several trait areas at once, and the raw text doubles as the voice sample the Persona Module uses for few-shot mimicry. Expected onboarding: ~10 minutes.

Each question shows a nudge: *"Write at least 4–5 sentences, the way you'd actually say it. The AI learns your voice from how you write here."* Minimum enforced: 200 characters per answer.

| Code | Probe area | Question text |
|---|---|---|
| `BQ1` | interests + approach | "What do you love spending your time on? Pick one or two things and walk me through how you actually engage with them — do you read and research, learn by doing, do it with people or alone? Give a recent concrete example." |
| `BQ2` | partner criteria | "Describe the person you'd want to be with. Which traits actually matter to you, which are nice-to-haves, and what is a hard dealbreaker? Be specific — 'kind' is not specific." |
| `BQ3` | situational (tense) | "Tell me about a recent disagreement or stressful moment with someone. What did you actually say and do, what were you feeling, and looking back, what would you keep or change?" |
| `BQ4` | situational (flirty/supportive) + conversational | "How do you act around someone you like? How do you show interest, what do you talk about on a good first date, and what makes you open up — or shut down?" |
| `BQ5` | self-image (qualities + flaws) | "How would a close friend honestly describe you, including the annoying parts? What do people tend to get wrong about you at first?" |

Coverage check against the Source of Truth's probe list: interests ✓ (BQ1), how interests are approached ✓ (BQ1), ideal-partner criteria ✓ (BQ2), tense situations ✓ (BQ3), flirty/supportive situations ✓ (BQ4), conversational aptitude ✓ (BQ4 + writing style across all five). Pool questions (A5) deepen every area.

Progress is saved per answer (upsert); the user can leave and resume mid-questionnaire.

## A3. PostgreSQL schema

Conventions: UUID primary keys (`gen_random_uuid()`), `timestamptz` everywhere, **text + CHECK constraints instead of native Postgres enums** *(trade: slightly weaker typing; accepted because ALTERing native enums during iteration is painful)*, `ON DELETE CASCADE` from `users` so account deletion is one statement. Migrations via Alembic. Baseline **and pool** questions are seeded by **startup reconciliation** — desired rows compared against actual on every boot (principle 12), never a "skip the check, it shipped with us" path.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    display_name    TEXT NOT NULL CHECK (char_length(display_name) BETWEEN 1 AND 50),
    birth_date      DATE NOT NULL,
    gender          TEXT NOT NULL CHECK (gender IN ('man','woman','nonbinary','other')),
    interested_in   TEXT[] NOT NULL CHECK (cardinality(interested_in) >= 1),
    age_pref_min    INT  NOT NULL DEFAULT 18 CHECK (age_pref_min >= 18),
    age_pref_max    INT  NOT NULL,
    city            TEXT,
    country         TEXT,
    opt_in          BOOLEAN NOT NULL DEFAULT FALSE,
    is_demo         BOOLEAN NOT NULL DEFAULT FALSE,   -- labeled demo/seed profile
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (age_pref_max >= age_pref_min)
);

-- REVISED 2026-09-01: 'question_batches' (AI generation runs) is gone — dynamic
-- questions now come from a curated pool (A5). Three origins remain:
--   baseline: BQ1-BQ5, global (user_id NULL), seeded
--   pool:     PQ01-PQ30, global (user_id NULL), seeded, ordered by pool_order
--   dispute:  AI-generated per-user follow-up targeting one disputed trait
CREATE TABLE questions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,   -- NULL = shared (baseline/pool)
    origin      TEXT NOT NULL CHECK (origin IN ('baseline','pool','dispute')),
    code        TEXT UNIQUE,                 -- 'BQ1'..'BQ5', 'PQ01'..'PQ30'; NULL for dispute
    pool_order  INT UNIQUE,                  -- 1..30, pool questions only; batch = next 5 unanswered
    probe_area  TEXT NOT NULL CHECK (probe_area IN
                  ('interests','partner_criteria','situational','conversational','self_image')),
    text        TEXT NOT NULL,
    trait_id    UUID REFERENCES traits(id) ON DELETE CASCADE,  -- dispute questions: the trait they correct
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((origin IN ('baseline','pool')) = (user_id IS NULL)),
    CHECK ((origin = 'pool') = (pool_order IS NOT NULL)),
    CHECK ((origin = 'dispute') = (trait_id IS NOT NULL))
);

CREATE TABLE answers (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question_id UUID NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    answer_text TEXT NOT NULL CHECK (char_length(answer_text) >= 200),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),   -- bumped on edit; drives staleness (A5.1)
    UNIQUE (user_id, question_id)                      -- upsert target; enables save/resume AND edit
);

CREATE TABLE traits (                        -- the structured traits profile (no prose blob)
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category          TEXT NOT NULL CHECK (category IN
                        ('interest','quality','flaw','behavioral','conversational_style','partner_preference')),
    label             TEXT NOT NULL,         -- short: "restores old cars"
    description       TEXT NOT NULL,         -- full: "Spends weekends restoring a '72 BMW; learns by doing…"
    confidence        REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    status            TEXT NOT NULL DEFAULT 'inferred' CHECK (status IN
                        ('inferred','confirmed','disputed','corrected','retracted')),
    source_answer_ids UUID[] NOT NULL,       -- provenance: which answers produced this (principle 9)
    extracted_by      TEXT NOT NULL,         -- provider/model that inferred it
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE trait_events (                  -- append-only audit trail (principle 7)
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trait_id   UUID NOT NULL REFERENCES traits(id) ON DELETE CASCADE,
    event      TEXT NOT NULL CHECK (event IN
                 ('created','updated','disputed','corrected','confirmed','retracted')),
    detail     TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- REVISED 2026-09-01 by candidate_matching.md: two vectors per user ('identity' and
-- 'preference'), so partner-criteria answers (BQ2) participate in matching.
CREATE TABLE profile_embeddings (
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL CHECK (kind IN ('identity','preference')),
    embedding       vector(768) NOT NULL,
    embedding_model TEXT NOT NULL,           -- pinned; a model change forces re-embed of everyone
    traits_hash     TEXT NOT NULL,           -- hash of the trait set embedded; detects staleness
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, kind)
);
```

Schema decisions worth naming:

- **Traits are rows, not a prose blob.** The persona prompt is rebuilt from current rows, so the profile can evolve forever without the prompt growing without bound (slow-fuse fix from the review).
- **Every trait carries provenance, confidence, and a dispute status.** An inferred trait is a hypothesis until confirmed (`development_principles.md` §9). The `trait_events` table makes every change reconstructable from logs alone (§7).
- **`vector(768)`** matches Google `text-embedding-004`. The dimension is baked into the column — switching embedding models means a migration plus re-embedding every user. *(Trade: accepted; at friends scale a full re-embed is minutes.)* `traits_hash` lets the matching module detect a stale embedding after any profile change.
- **`is_demo` on users** implements the labeled-seed-profile decision at the schema level, so no UI can forget it.
- **The pool lives in `questions`, not a separate table.** One table, one `origin` discriminator, one `answers` FK — "which pool questions has this user answered" is a join, not a sync problem between two tables.

## A4. Module endpoints

| Endpoint | Behavior |
|---|---|
| `POST /auth/register` | Creates user from the form in A1. |
| `POST /auth/login` | JWT session. |
| `GET /me` / `PATCH /me` | Read/update profile fields, preferences, `opt_in`. |
| `DELETE /me` | Full cascade deletion; logs what was deleted. |
| `GET /questions` | Everything answerable by this user — baseline + pool + their dispute questions — each with answered/unanswered state and the answer text when present (drives save/resume **and** the edit view). |
| `GET /questions/next-batch` | The next expansion batch (A5.2): up to 5 unanswered pool questions by `pool_order`, plus progress `{answered_pool, total_pool}`. Pool done → `{status: 'pool_exhausted', questions: []}`. |
| `PUT /answers/{question_id}` | Upsert one answer — first write and every later edit go through the same path. |
| `POST /profile/extract` | Runs trait extraction over all answered questions **against the existing trait rows**, applying the model's per-row `keep`/`update`/`retract`/`add` verdicts (A5.1); writes `trait_events`, refreshes both `profile_embeddings` only when something actually changed. At most one extraction runs per user; a request during a run queues one follow-up run (no pile-up). |
| `GET /traits` | The trait profile, including confidence and status — the UI's trait display with dispute controls. |
| `POST /traits/{id}/dispute` | Marks `disputed`, AI-generates one follow-up question targeted at the trait (`origin='dispute'`, linked via `trait_id`) so the next extraction corrects rather than re-infers. |

Logging obligations (ship in the same commit as the feature, §7): trait extraction and dispute-question generation each log provider, model, input answer IDs, produced items, and — on the refusal/failure path — the raw model output. Answer edits log old/new length and the trait IDs whose provenance includes the edited answer.

## A5. Profile updates and expansion (the four data flows)

### A5.1 Changing past answers

Any previously submitted answer — baseline, pool, or dispute — can be edited at any time.

```
UI edit ─► PUT /answers/{question_id} (same upsert; updated_at bumped)
   ─► on leaving the edit session, UI calls POST /profile/extract
        ─► extraction re-reads ALL answers, reconciles traits:
             · traits whose source_answer_ids include an edited answer are re-examined first
             · changed → status stays/updates + trait_event 'updated'
             · no longer supported → 'retracted' (never silently deleted)
        ─► traits_hash changes ─► both embeddings stale (matching re-embeds before next analysis)
        ─► persona snapshot stale (trait_persona.md: header shows "profile changed — rebuild")
```

Three rules that make edits safe:

- **Edits are forward-looking only.** Past analyses, date transcripts, and chat sessions keep the persona snapshots they pinned; an edit changes who you are *from now on*, never the record of what was simulated. (This falls out of snapshot immutability — stated here so nobody "fixes" it.)
- **Extraction is holistic reconciliation, not incremental patching.** It always re-reads the full answer set and converges the trait rows to it. *(Trade: more tokens per run than a delta approach; accepted — a reconcile can't accumulate drift, and at 5–35 answers the cost difference is noise.)*
- **Trait identity is verdict-based (owner decision, 2026-09-01).** The extraction call receives the user's *existing trait rows* (id, label, description, status) alongside the answers, and must return an explicit verdict per existing row — `keep`, `update`, or `retract` — plus `add` entries for genuinely new traits. Rows are matched by id, never by re-matching wording, so a rephrased description is an `update` to the same row: **`confirmed` and `disputed` statuses, provenance, and dispute history survive re-extraction.** A no-edit re-run must produce all-`keep` (or near it); `probe_answer_edit.py` asserts that a confirmed trait survives an unrelated edit, and `trait_events` volume on a no-edit re-run is the drift alarm. Only an `update`/`retract`/`add` changes `traits_hash` — an all-`keep` run leaves embeddings and the persona snapshot fresh instead of marking them stale for nothing.

### A5.2 Profile expansion — batches of 5 from the pool

**Revision named per principle 23:** this replaces the AI-generated Dynamic Questionnaire Generator (and its `question_batches` table). New information: owner decision 2026-09-01 — expansion uses a **curated, pre-defined pool** served in fixed batches of 5. *(Trade: questions aren't personalized to the individual; accepted — curated questions have uniform trait coverage, zero generation failures, no moderation risk, and identical questions across users make profiles more comparable for matching. The one remaining AI-generated question type is dispute follow-ups, which target a specific trait and can't be pre-written.)*

Flow: `GET /questions/next-batch` → the next 5 unanswered pool questions in `pool_order` (deterministic — no assignment table needed: the answer set *is* the cursor) → user answers them with the same one-per-page UI and autosave as onboarding → completing the batch triggers extract → compile, same as A5.1. A user who abandons a batch mid-way sees the same remaining questions next time. 30 questions ÷ 5 = exactly 6 batches.

### A5.3 The question pool — PQ01–PQ30 (final)

Thirty questions, six per probe area, seeded by startup reconciliation exactly like BQ1–BQ5. Same 200-character minimum and voice nudge.

| Code | Probe area | Question |
|---|---|---|
| PQ01 | interests | "What's a hobby or skill you abandoned but still think about? What pulled you away, and what would bring you back?" |
| PQ02 | interests | "You get a completely free Saturday — no obligations, no guilt. Walk me through it, morning to night." |
| PQ03 | interests | "What topic could you talk about for an hour with no preparation? How did you get into it?" |
| PQ04 | interests | "What are you currently learning, or itching to learn? How do you go about learning things — courses, videos, just diving in?" |
| PQ05 | interests | "Describe your ideal trip. Planned to the hour or improvised? Packed with activity or still? Who's with you, if anyone?" |
| PQ06 | interests | "What do you make, fix, build, or grow? Anything counts — food, code, furniture, playlists, a garden." |
| PQ07 | partner_criteria | "Think of a couple you actually admire. What do they have that you want for yourself?" |
| PQ08 | partner_criteria | "What's something a partner did — or could do — that instantly makes you feel cared for? Something small counts." |
| PQ09 | partner_criteria | "What would a partner simply have to accept about you, no negotiation? Be honest." |
| PQ10 | partner_criteria | "In a normal week, how much together-time versus alone-time do you actually need? What happens when you don't get the alone part?" |
| PQ11 | partner_criteria | "When you're struggling, what do you want from a partner — solutions, listening, distraction, space? What do people usually get wrong?" |
| PQ12 | partner_criteria | "What's a relationship opinion you hold that most people around you disagree with?" |
| PQ13 | situational | "Your date is 25 minutes late, phone off. They arrive full of apologies. What do you actually say — and what are you feeling underneath?" |
| PQ14 | situational | "You deeply disagree with someone you care about on something that matters. Walk me through what you actually do, step by step." |
| PQ15 | situational | "Your partner had an awful day and snaps at you over nothing. What happens in the next five minutes?" |
| PQ16 | situational | "You're at a party where you know almost nobody. What do you actually do for the first half hour?" |
| PQ17 | situational | "Something genuinely embarrassing happens to you on a date. What's your move — laugh it off, own it, die inside quietly?" |
| PQ18 | situational | "You planned something special and it falls apart at the last minute. What's your honest, real-time reaction?" |
| PQ19 | conversational | "What's your favorite kind of conversation — deep and personal, playful banter, a good-natured argument, trading stories? Give a real example of a great one you had." |
| PQ20 | conversational | "How do you tell someone something they don't want to hear? Tell me about a time you had to." |
| PQ21 | conversational | "In a group conversation, what role do you naturally fall into — the storyteller, the questioner, the referee, the quiet one who lands one good line?" |
| PQ22 | conversational | "What question do you wish people asked you more often? Answer it." |
| PQ23 | conversational | "When someone tells you a long story, what's honestly going on in your head — absorbed listening, planning your reply, connecting it to your own life?" |
| PQ24 | conversational | "Teasing: do you dish it out, take it, both, neither? Where exactly is the line for you?" |
| PQ25 | self_image | "What's a compliment that stuck with you for years? Why did that one land?" |
| PQ26 | self_image | "What's a flaw you've genuinely worked on? How is that going — honestly?" |
| PQ27 | self_image | "When you're stressed, what do you do that other people notice before you do?" |
| PQ28 | self_image | "After a conflict with someone close, what do you need — space, contact, a resolution talk, humor? How long until you're actually over it?" |
| PQ29 | self_image | "What are you proud of that you rarely get to talk about?" |
| PQ30 | self_image | "Your best friend gets to warn your future partner about one thing. What do they say — and are they right?" |

### A5.4 Pool exhaustion

The tracking mechanism is the `answers` table itself — a pool question is "used" for a user exactly when their answer row exists. When all 30 are answered:

- `GET /questions/next-batch` returns `{status: 'pool_exhausted', questions: [], progress: {answered_pool: 30, total_pool: 30}}` — a defined state, not an empty-list ambiguity.
- The UI replaces the expansion CTA with a completed state: "You've answered everything — your profile is as deep as it gets for now." No error styling; this is an achievement, not a failure.
- Dispute follow-up questions remain available regardless — they're per-user and outside the pool, and don't count toward `answered_pool`.
- Editing existing answers (A5.1) remains the way to keep refining after exhaustion; the UI's exhausted state says so.
- Growing the pool later = appending PQ31+ rows with higher `pool_order` via a new seed reconciliation — previously exhausted users automatically get fresh batches. No migration, no special case.

---

## Locked by this document

1. Registration form fields and rules (A1); distance filtering deferred, columns kept.
2. The five baseline questions, verbatim, with codes `BQ1`–`BQ5` (A2).
3. The full Postgres schema (A3) — including the revised `questions` table (three origins, `pool_order`, `trait_id`) and the removal of `question_batches`.
4. Traits stored as structured rows with provenance + confidence + status; prose profile abolished.
5. Answer editing flow: same upsert path, holistic re-extraction, forward-looking only (A5.1).
6. Expansion in fixed batches of 5 from the curated pool; no AI-generated expansion questions (A5.2).
7. The 30 pool questions, verbatim, `PQ01`–`PQ30` (A5.3).
8. Pool exhaustion as a defined, graceful state; answers table as the sole tracking mechanism; pool growable by appending (A5.4).

The AI Interaction Module (foundational service: providers, routing, structured-output guard, resilience) is documented separately in `ai_interaction.md`.

## Open for other modules

- Embedding input composition and matching formula: `candidate_matching.md` (done).
- Persona compilation from these tables: `trait_persona.md` (done).
