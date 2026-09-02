"""Candidate rejection: `analysis_candidates.status` + `rejected_at` (S17-B1).

A person can turn down one of their three before the dates run, and the
next-best eligible person takes the empty seat.

**The rejected row is kept, not deleted.** Three reasons, in order of how much
they matter:

1. It is the record of a decision the user made. `trait_events`, retracted
   traits and superseded persona snapshots all follow the same rule — this
   system does not erase a user's own history to keep a table tidy.
2. It is how the replacement search knows not to offer the same person again
   two seconds later. A deleted row would make rejection a shuffle.
3. `UNIQUE (analysis_id, candidate_user_id)` still holds, so a rejected person
   cannot be re-inserted as a replacement by any code path, present or future.

**`UNIQUE (analysis_id, rank)` becomes a PARTIAL unique index over the active
rows only.** Ranks are re-assigned 1..n by compatibility after every
replacement, so a rejected row keeps the rank it held when the user saw it —
which is the honest historical value and would collide with the new occupant
under the old table-wide constraint. The invariant people actually rely on
("the live candidates are ranked 1..n, each rank once") is exactly what the
partial index states.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-02
"""
from __future__ import annotations

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE analysis_candidates "
        "ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
    )
    op.execute(
        "ALTER TABLE analysis_candidates ADD CONSTRAINT analysis_candidates_status_check "
        "CHECK (status IN ('active','rejected'))"
    )
    op.execute("ALTER TABLE analysis_candidates ADD COLUMN rejected_at TIMESTAMPTZ")
    # The rejected row must carry WHEN, or "did she reject him before or after
    # she saw the second one" is unanswerable from the database.
    op.execute(
        "ALTER TABLE analysis_candidates ADD CONSTRAINT analysis_candidates_rejected_at_check "
        "CHECK ((status = 'rejected') = (rejected_at IS NOT NULL))"
    )
    # Postgres named the table-wide constraint for us at creation time.
    op.execute(
        "ALTER TABLE analysis_candidates "
        "DROP CONSTRAINT IF EXISTS analysis_candidates_analysis_id_rank_key"
    )
    op.execute(
        "CREATE UNIQUE INDEX ux_analysis_candidates_active_rank "
        "ON analysis_candidates (analysis_id, rank) WHERE status = 'active'"
    )


def downgrade() -> None:
    # Rejected rows would violate the restored table-wide constraint, so they
    # go first — stated here rather than discovered during a rollback.
    op.execute("DELETE FROM analysis_candidates WHERE status = 'rejected'")
    op.execute("DROP INDEX IF EXISTS ux_analysis_candidates_active_rank")
    op.execute(
        "ALTER TABLE analysis_candidates "
        "ADD CONSTRAINT analysis_candidates_analysis_id_rank_key UNIQUE (analysis_id, rank)"
    )
    op.execute(
        "ALTER TABLE analysis_candidates "
        "DROP CONSTRAINT IF EXISTS analysis_candidates_rejected_at_check"
    )
    op.execute(
        "ALTER TABLE analysis_candidates "
        "DROP CONSTRAINT IF EXISTS analysis_candidates_status_check"
    )
    op.execute("ALTER TABLE analysis_candidates DROP COLUMN rejected_at")
    op.execute("ALTER TABLE analysis_candidates DROP COLUMN status")
