"""`date_evaluations.confidence` and `.evidence_note` (2026-09-02).

The judge now says how much the transcript supported its own reading.

This is what replaces the "fewer than 10 agent turns, not judged at all" rule,
which answered a question about DEPTH with a rule about ADMISSION and left a
person who had just watched a date happen with nothing to read about it. Every
date with a transcript is judged; a thin one is judged carefully and marked as
thinly evidenced.

Both columns are NULLABLE and stay that way. Rows scored under
`judge_rubric.v1` have no confidence and never will -- the v1 judge was never
asked. A default would be a judgement nobody made, sitting in the column that
exists to say how sure somebody was.

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-02
"""
from __future__ import annotations

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE date_evaluations ADD COLUMN confidence INTEGER")
    op.execute(
        "ALTER TABLE date_evaluations ADD CONSTRAINT date_evaluations_confidence_range "
        "CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 100)"
    )
    op.execute("ALTER TABLE date_evaluations ADD COLUMN evidence_note TEXT")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE date_evaluations "
        "DROP CONSTRAINT IF EXISTS date_evaluations_confidence_range"
    )
    op.execute("ALTER TABLE date_evaluations DROP COLUMN IF EXISTS evidence_note")
    op.execute("ALTER TABLE date_evaluations DROP COLUMN IF EXISTS confidence")
