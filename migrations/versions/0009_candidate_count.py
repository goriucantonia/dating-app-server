"""`analyses.candidate_count` — how many people this analysis HAD (S15-B3).

Deletion punches holes: when a candidate deletes their account, their
`analysis_candidates` row cascades away and the survivor's analysis quietly
shows two people where there were three. `data_hygiene.md` §2 says the gap
must be honestly labeled — which needs the original count, because nothing
else remembers it. `pool_status` is not enough: `partial` means one or two.

Backfilled from the current rows for every existing analysis, which is exact
for every analysis nobody has deleted out of yet (all of them, today).

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-01
"""
from __future__ import annotations

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE analyses ADD COLUMN candidate_count INT")
    op.execute(
        """
UPDATE analyses a
SET candidate_count = (
    SELECT count(*) FROM analysis_candidates c WHERE c.analysis_id = a.id
)
WHERE a.status IN ('matched', 'simulating', 'complete', 'no_candidates')
"""
    )


def downgrade() -> None:
    op.execute("ALTER TABLE analyses DROP COLUMN candidate_count")
