# Module Plan — Candidate Matching Module

Status: planning locked 2026-09-01. Depends on: `module_1_data_collection.md` (users, traits, embeddings), `trait_persona.md` (snapshot gate). Consumed by: Date Simulation.

---

## 1. Purpose

Given a requesting user, produce up to 3 candidates who (a) pass the hard filters and (b) score highest on trait compatibility — honestly reporting when the pool is smaller than 3 or empty. This module owns the `analyses` lifecycle up to the point where simulation takes over.

## 2. Core features and async operation

- **Analysis creation is a background job.** `POST /analyses` returns an `analysis_id` immediately with status `matching`. The job: refresh stale embeddings (may need AI calls — this is why it's async) → hard filter → vector scoring → write candidates → status `matched`. Fast path (nothing stale, small pool) completes in under a second; the API contract doesn't care.
- **Hard-filter pre-pass (pure SQL, no AI).** A candidate is eligible iff:
  1. `opt_in = TRUE` and not the requester;
  2. mutual gender fit: candidate's `gender = ANY(requester.interested_in)` **and** requester's `gender = ANY(candidate.interested_in)`;
  3. mutual age fit: each person's age (computed from `birth_date`) inside the other's `age_pref_min..age_pref_max`;
  4. has a `ready` persona snapshot and a fresh-enough profile embedding — no snapshot, no candidacy (the gate from `trait_persona.md` §5; a candidate who can't be simulated must not be offered).
  Distance is not filtered this phase (columns exist; decision in `module_1_data_collection.md` A1). `is_demo` profiles are eligible — they exist to fill the pool and are labeled everywhere.
- **Two-vector mutual compatibility scoring.** Each user has **two** embeddings:
  - `identity` — who they are: interests, qualities, flaws, behavioral, conversational-style trait rows serialized deterministically (category-ordered, `label: description` lines);
  - `preference` — who they want: `partner_preference` trait rows serialized the same way.
  Compatibility between requester R and candidate C:
  ```
  fit(R→C) = cosine(R.preference, C.identity)   # does C match what R wants?
  fit(C→R) = cosine(C.preference, R.identity)   # does R match what C wants?
  compatibility = (fit(R→C) + fit(C→R)) / 2     # mutual, symmetric by construction
  ```
  *Why (revision named, principle 23):* the earlier single-embedding design compared identity-to-identity, which measures "are these two people similar" and silently discards everything BQ2 (partner criteria) collects. The new information is concrete: we now have a `partner_preference` trait category with nowhere to go. Similarity-as-v1-heuristic (decision log #6) still stands — this is still cosine similarity, aimed at the right vectors.
- **"Why chosen" reasons, computed in code.** Shared interests = intersection of `interest` trait labels (case-folded token overlap); top contributing preference↔identity trait pairs come from per-category sub-scores. No LLM call — reasons must be exactly true, not plausible.
- **Honest small-pool behavior.** 3+ eligible → top 3. 1–2 → those, with `pool_status = 'partial'`. 0 → status `no_candidates`, message "there is no one to match you with yet." Never fabricated, never padded with ineligible users.

## 3. Data flow and database

```
POST /analyses ──► analyses(status='matching') ──► [background job]
   1. staleness check: profile_embeddings.traits_hash vs live hash ──► re-embed via AIProvider.embed (both vectors)
   2. hard filter (SQL over users ⋈ persona_snapshots ⋈ profile_embeddings)
   3. score: pgvector cosine on the two-vector formula, top 3
   4. INSERT analysis_candidates; analyses.status='matched' (or 'no_candidates')
```

Schema revision to `module_1_data_collection.md` A3 (recorded there): `profile_embeddings` gains a `kind` column.

```sql
-- REVISED from module_1_data_collection.md: two rows per user, one per vector kind
CREATE TABLE profile_embeddings (
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL CHECK (kind IN ('identity','preference')),
    embedding       vector(768) NOT NULL,
    embedding_model TEXT NOT NULL,
    traits_hash     TEXT NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, kind)
);

CREATE TABLE analyses (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status       TEXT NOT NULL CHECK (status IN
                   ('matching','matched','no_candidates','simulating','complete','failed')),
    pool_status  TEXT CHECK (pool_status IN ('full','partial','empty')),
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE analysis_candidates (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id       UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    candidate_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rank              INT  NOT NULL,                  -- 1..3
    fit_forward       REAL NOT NULL,                  -- cosine(R.pref, C.identity)
    fit_backward      REAL NOT NULL,                  -- cosine(C.pref, R.identity)
    compatibility     REAL NOT NULL,                  -- the mean; what the UI shows
    shared_interests  TEXT[] NOT NULL,                -- computed, exact
    reason_summary    TEXT NOT NULL,                  -- code-assembled from the above
    snapshot_id       UUID NOT NULL REFERENCES persona_snapshots(id),  -- frozen at match time
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (analysis_id, candidate_user_id),
    UNIQUE (analysis_id, rank)
);
```

`analysis_candidates.snapshot_id` freezes the candidate's persona **at match time** — if the candidate answers more questions mid-simulation, the dates still run against the persona that was matched, keeping scores and transcripts consistent within one analysis.

The `analyses.status` state machine is owned jointly: this module drives `matching → matched | no_candidates | failed`; Date Simulation drives `matched → simulating → complete | failed`. One table, one row per run — the UI polls a single object for the whole journey.

## 4. Interfaces (module boundary)

```python
class MatchingService(Protocol):
    async def start_analysis(self, user_id: UUID) -> UUID:        # analysis_id; job runs in background
    async def get_analysis(self, analysis_id: UUID) -> Analysis:  # status + candidates + reasons
```

Date Simulation consumes `Analysis.candidates` (each carrying its frozen `snapshot_id`). It never re-queries eligibility.

## 5. Endpoints

| Endpoint | Behavior |
|---|---|
| `POST /analyses` | Start a run; 409 if this user already has one in `matching`/`simulating` (one active analysis per user — free-tier rate limits make concurrent runs pointless). |
| `GET /analyses/{id}` | Status, pool_status, candidates with ranks/reasons. The UI's single polling target. |
| `GET /analyses` | This user's history, newest first (the revisitable-results decision). |

## 6. Technical decisions (trades named)

1. **Two vectors per user.** Cost: doubles embedding storage and re-embed calls (trivial at this scale). Accepted because it's the only way collected partner-criteria data participates in matching.
2. **Exact scoring, no ANN index.** With a friends-sized pool, brute-force cosine over the filtered set is microseconds; an IVFFlat/HNSW index would be tuning work for nothing. Add an index when the pool passes ~10k. One-way doors: none — it's an `ALTER TABLE` later.
3. **Reasons computed, not generated.** Cost: less florid copy. Accepted: a fabricated "you both love hiking" that isn't in the data would be the trust-killer here.
4. **Candidate snapshots frozen per analysis.** Cost: a candidate's newest self isn't reflected mid-run. Accepted for internal consistency of each analysis.
5. **One active analysis per user.** Cost: no queuing up runs. Accepted; simplifies state and matches free-tier throughput reality.

Logging obligations (§7): the job logs pool size after each filter step (opt-in count → mutual-gender count → mutual-age count → snapshot-ready count), the three scores per selected candidate, and — on `no_candidates` — which filter emptied the pool. That last line is the debugging tool for "why am I getting no matches."

## Locked by this document

1. Hard-filter definition, including the persona-snapshot gate; distance deferred.
2. Two-vector mutual compatibility formula, verbatim.
3. `analyses` / `analysis_candidates` schema and the shared status state machine.
4. `profile_embeddings` revision (`kind` column, PK `(user_id, kind)`).
5. Honest pool handling: `full` / `partial` / `empty`, never padded.
6. One active analysis per user.

## Open for the next module (Date Simulation)

- Scenario generation from `shared_interests` (and the fallback when the intersection is empty — candidate chosen on complementary preference fit can share zero interest labels).
- Turn orchestration, checkpointing, event injection, and the judge pipeline.
