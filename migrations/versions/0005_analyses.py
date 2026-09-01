"""Analyses and analysis candidates — verbatim from candidate_matching.md §3 (S9-B1).

`analyses.progress` is JSONB reserved for Step 11's simulation progress, added
now rather than later: the row is the ONE object the UI polls for the whole
journey (matching → simulating → complete), and adding a column to a table the
UI is already polling is a migration nobody needs.

`analysis_candidates.snapshot_id` has NO cascade, deliberately, for the same
reason `calibration_sessions.snapshot_id` does not: it freezes the candidate's
persona AT MATCH TIME (trade #4). If the candidate answers more questions
mid-simulation the dates still run against the persona that was matched, so
scores and transcripts stay consistent within one analysis — and that is only
true if the snapshot cannot be deleted out from under the row.

The two UNIQUE constraints are the honesty guarantees in schema form: a
candidate cannot appear twice in one analysis, and two candidates cannot claim
the same rank.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-01
"""
from __future__ import annotations

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # One op.execute per statement — asyncpg cannot prepare multi-command SQL.
    op.execute(
        """
CREATE TABLE analyses (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status       TEXT NOT NULL CHECK (status IN
                   ('matching','matched','no_candidates','simulating','complete','failed')),
    pool_status  TEXT CHECK (pool_status IN ('full','partial','empty')),
    progress     JSONB,
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
    )
    op.execute(
        """
CREATE TABLE analysis_candidates (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id       UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    candidate_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rank              INT  NOT NULL,
    fit_forward       REAL NOT NULL,
    fit_backward      REAL NOT NULL,
    compatibility     REAL NOT NULL,
    shared_interests  TEXT[] NOT NULL,
    reason_summary    TEXT NOT NULL,
    snapshot_id       UUID NOT NULL REFERENCES persona_snapshots(id),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (analysis_id, candidate_user_id),
    UNIQUE (analysis_id, rank)
);
"""
    )
    # `GET /analyses` is history newest-first, and the 409 check asks "does this
    # user have an active run" on every POST. Both are this index.
    op.execute(
        "CREATE INDEX ix_analyses_user_recent ON analyses (user_id, created_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE analysis_candidates")
    op.execute("DROP TABLE analyses")
