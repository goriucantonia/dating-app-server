"""Dates and date messages — verbatim from date_simulation.md §3 (S11-B1).

`analyses.progress JSONB` is NOT added here: migration 0005 added it already,
deliberately and ahead of need, because the analysis row is the ONE object the
UI polls for the whole journey and adding a column to a table the UI is already
polling is a migration nobody needs. S11-B1 names it; 0005 delivered it.

The judge's tables (`date_evaluations`, `candidate_scores`) are deliberately
NOT here — they belong to Step 12 (S12-B1) and land with the code that writes
them. A table with no writer is indistinguishable from a table whose writer is
broken.

Three constraints in here are mechanisms, not decoration:

- `UNIQUE (date_id, seq)` is **the checkpoint invariant**. A turn is persisted
  before the turn advances (§19), and this is what makes that persistence
  meaningful: a resumed date cannot write a second message at a seq it already
  wrote, so resume is idempotent by construction rather than by care.
- `UNIQUE (analysis_id, candidate_user_id, ordinal)` caps a candidate at their
  two dates in schema, so a re-launched pipeline cannot quietly create a third.
- The two `snapshot_id` columns have **no cascade**, the same as
  `analysis_candidates.snapshot_id`: a transcript must always be able to name
  the exact persona version that spoke it, or "the agent said X because
  snapshot v3 said Y" stops being answerable.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-01
"""
from __future__ import annotations

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # One op.execute per statement — asyncpg cannot prepare multi-command SQL.
    op.execute(
        """
CREATE TABLE dates (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id        UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    candidate_user_id  UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ordinal            INT  NOT NULL,
    scenario           JSONB NOT NULL,
    status             TEXT NOT NULL CHECK (status IN
                         ('pending','running','complete','incomplete','failed')),
    user_snapshot_id      UUID NOT NULL REFERENCES persona_snapshots(id),
    candidate_snapshot_id UUID NOT NULL REFERENCES persona_snapshots(id),
    schema_version     TEXT NOT NULL,
    error              TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at        TIMESTAMPTZ,
    UNIQUE (analysis_id, candidate_user_id, ordinal)
);
"""
    )
    op.execute(
        """
CREATE TABLE date_messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date_id     UUID NOT NULL REFERENCES dates(id) ON DELETE CASCADE,
    seq         INT  NOT NULL,
    speaker     TEXT NOT NULL CHECK (speaker IN
                  ('user_agent','candidate_agent','environment')),
    reply       TEXT NOT NULL,
    state       JSONB,
    provider    TEXT,
    model_id    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (date_id, seq)
);
"""
    )
    # Every read of a date is "the transcript in seq order" and every resume is
    # "max(seq) for this date". Both are this index.
    op.execute("CREATE INDEX ix_date_messages_date_seq ON date_messages (date_id, seq)")
    # The reconciliation relaunch (S11-B9) and the results screen both ask
    # "the dates of this analysis, in order".
    op.execute("CREATE INDEX ix_dates_analysis ON dates (analysis_id, created_at)")


def downgrade() -> None:
    op.execute("DROP TABLE date_messages")
    op.execute("DROP TABLE dates")
