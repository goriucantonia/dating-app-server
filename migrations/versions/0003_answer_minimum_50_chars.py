"""Answer minimum lowered from 200 to 50 characters (owner decision, 2026-09-01).

The 200-character floor was the original A2/A3 figure. The owner judged it too
long a thing to demand of someone at every one of 35 questions, and lowered it
to 50. The SCOPE is unchanged and still stated (development_principles.md §18):
the minimum applies to baseline, pool, AND dispute answers alike.

Loosening a CHECK is safe on existing rows — every answer already stored
satisfies >= 200, so it satisfies >= 50. Nothing is rewritten, nothing is lost.
The constraint was created inline and unnamed in 0002, so Postgres named it
`answers_answer_text_check`; it is dropped and re-added rather than altered
because Postgres has no ALTER CONSTRAINT for a CHECK expression.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-01
"""
from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # One op.execute per statement — asyncpg cannot prepare multi-command SQL.
    op.execute("ALTER TABLE answers DROP CONSTRAINT answers_answer_text_check")
    op.execute(
        "ALTER TABLE answers ADD CONSTRAINT answers_answer_text_check "
        "CHECK (char_length(answer_text) >= 50)"
    )


def downgrade() -> None:
    # Not symmetric in effect: answers between 50 and 199 characters written
    # while the lower floor was in force would fail this re-tightened CHECK.
    # Downgrading a loosened constraint is only safe on data that predates it.
    op.execute("ALTER TABLE answers DROP CONSTRAINT answers_answer_text_check")
    op.execute(
        "ALTER TABLE answers ADD CONSTRAINT answers_answer_text_check "
        "CHECK (char_length(answer_text) >= 200)"
    )
