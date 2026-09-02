"""`analyses.scenarios` — the ONE setting every candidate is run against.

Scenarios used to be generated per candidate and anchored in that pair's shared
interests, so three candidates got three different evenings and
`candidate_scores` ranked the three resulting numbers side by side as if they
were the same measurement. They were not — see D-021. The column holds a JSONB
ARRAY (one entry per date each candidate gets) rather than a single object, so
it stays correct if `SETTINGS_PER_ANALYSIS` ever moves off 1.

It lives on the analysis rather than being read back off whichever date row was
created first: "all three ran the same evening" is the property every score on
the results screen depends on, and a property nothing stores is a property
nothing can enforce. `dates.scenario` keeps its own copy — a transcript must be
able to say where it happened without its analysis row still agreeing.

NULL is a real state and no backfill is attempted. Existing analyses ran under
the per-candidate design and genuinely had no single fixture; writing one now
would be inventing history. They keep their per-date scenarios and render
exactly as they always did.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-02
"""
from __future__ import annotations

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE analyses ADD COLUMN scenarios JSONB")
    # An array, checked in the database rather than only in the ORM: this
    # column is read back and iterated, and a stray object here would fail
    # deep inside the pipeline rather than at the write that caused it.
    op.execute(
        "ALTER TABLE analyses ADD CONSTRAINT analyses_scenarios_is_array "
        "CHECK (scenarios IS NULL OR jsonb_typeof(scenarios) = 'array')"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE analyses DROP CONSTRAINT IF EXISTS analyses_scenarios_is_array"
    )
    op.execute("ALTER TABLE analyses DROP COLUMN IF EXISTS scenarios")
