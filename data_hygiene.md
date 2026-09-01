# Module Plan — Data Hygiene Module

Status: planning locked 2026-09-01. Cross-cutting: touches every table. Last server module.

---

## 1. Purpose

Owns the obligations that span modules: full account deletion, demo-profile seeding and labeling, and the startup reconciliation jobs that keep "must be present by default" data honest (principle 12). Small on purpose — most hygiene is enforced by schema design (`ON DELETE CASCADE`, `is_demo`, CHECK constraints) rather than by code here.

## 2. Core features

- **Account deletion (`DELETE /me`).** One transaction: `DELETE FROM users WHERE id = …`. The cascade graph — verified, not assumed (principle 13): answers, dispute questions, traits, trait events, both embeddings, persona snapshots, calibration data, analyses + candidates, dates + messages + evaluations + scores, chat sessions + messages. (Baseline and pool questions are global rows and survive, as they must.)
  Two cross-user effects, named plainly:
  1. Dates/chats where the deleted user was the **candidate** also disappear from other users' analyses and chat lists. *(Trade: a friend's deletion punches holes in your history; accepted — their persona, answers, and simulated behavior are their data, and privacy beats history. The UI shows "this person removed their account" for a dangling analysis.)*
  2. `analysis_candidates` referencing the deleted user cascade too, so an old analysis may show 2 of 3 candidates; the analysis row survives with a gap, honestly labeled.
  The endpoint logs, before deleting, a per-table row count of what is about to go (§7) — the deletion trace without retaining the data.
- **Demo profile seeding.** A fixture file (`seeds/demo_profiles.yaml`) defines demo users: form fields + five baseline answers each, `is_demo = TRUE`. On startup, reconciliation compares fixtures against the database by a stable seed key and inserts/updates what's missing — the same code path as real data (registration + answer upsert + extraction pipeline), **no shortcut inserts**, so demo profiles have real traits, snapshots, and embeddings like everyone else. `is_demo` surfaces in every candidate/API payload; the UI labels them.
- **Startup reconciliation pass** (one place, ordered):
  1. Baseline questions BQ1–BQ5 and pool questions PQ01–PQ30 (from `module_1_data_collection.md`).
  2. Demo profiles (above) — including re-running extraction/compilation/embedding for any demo user missing them.
  3. Re-launch analyses stuck in `matching`/`simulating` (the resume pass from `date_simulation.md`).
  4. Embedding-model consistency check: every `profile_embeddings.embedding_model` must equal the pinned config value; mismatches are logged loudly and queued for re-embedding, never silently compared.
- **Dead-data scan (principle 22, scripted).** A maintenance script (committed, run manually) reports: users with zero answers older than 30 days, `failed` snapshots/analyses, orphaned `running` dates. Report only — deletion of real users' data is always a human decision.

## 3. Interfaces and endpoints

| Endpoint | Behavior |
|---|---|
| `DELETE /me` | Full deletion as above; returns the per-table counts it logged. |
| *(internal)* `reconcile()` | The startup pass; also invocable from a management script. |
| *(script)* `scripts/scan_dead_data.py` | The report above. |

## Locked by this document

1. Deletion = single cascade transaction; cross-user effects accepted and surfaced in UI ("this person removed their account").
2. Demo profiles seeded through the real pipeline via startup reconciliation; never shortcut-inserted.
3. The four-step reconciliation order.
4. Embedding-model consistency check on every boot.

---

*Server Repository planning is complete: `module_1_data_collection.md`, `ai_interaction.md` (foundational service), `trait_persona.md`, `candidate_matching.md`, `date_simulation.md`, `chat.md`, `data_hygiene.md`. Next: UI/UX Repository modules.*
