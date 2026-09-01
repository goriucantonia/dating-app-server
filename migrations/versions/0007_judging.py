"""Date evaluations and candidate scores — verbatim from date_simulation.md §3
(S12-B1).

They arrive now, with Step 12, rather than alongside `dates` in 0006, for the
reason D-001 recorded: a table with no writer is indistinguishable from a table
whose writer is broken. Until this migration there was no judge.

Two shapes in here are the module's argument in schema form:

- **`criteria` is JSONB and `date_score` is a REAL beside it.** The four 0-100
  numbers the model produced are stored raw, and the single number the product
  shows is computed in code from them (trade #3). Keeping both means anyone can
  recompute the score by hand from the stored inputs and check it — which is
  literally what `probe_judge.py` does. Storing only the score would make the
  weights unauditable; storing only the criteria would make the score
  irreproducible after a weights change.
- **`rubric_version`, `judge_provider` and `judge_model` are NOT NULL.** A
  score without provenance is a number with no argument behind it (§9). Two
  evaluations produced months apart under different rubrics must be
  distinguishable without archaeology.

`date_evaluations.date_id` is the PRIMARY KEY, not a plain FK: one evaluation
per date, enforced. That is what makes the judge pass idempotent on re-run —
a relaunched pipeline that tries to judge an already-judged date collides
rather than quietly scoring it twice with a different model.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-01
"""
from __future__ import annotations

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # One op.execute per statement — asyncpg cannot prepare multi-command SQL.
    op.execute(
        """
CREATE TABLE date_evaluations (
    date_id        UUID PRIMARY KEY REFERENCES dates(id) ON DELETE CASCADE,
    criteria       JSONB NOT NULL,
    clicked        JSONB NOT NULL,
    clashes        JSONB NOT NULL,
    per_peer       JSONB NOT NULL,
    verdict        TEXT  NOT NULL,
    date_score     REAL  NOT NULL,
    is_partial     BOOLEAN NOT NULL DEFAULT FALSE,
    judge_provider TEXT NOT NULL,
    judge_model    TEXT NOT NULL,
    rubric_version TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
    )
    op.execute(
        """
CREATE TABLE candidate_scores (
    analysis_id       UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    candidate_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    final_score       REAL NOT NULL,
    dates_completed   INT  NOT NULL,
    dates_incomplete  INT  NOT NULL,
    PRIMARY KEY (analysis_id, candidate_user_id)
);
"""
    )


def downgrade() -> None:
    op.execute("DROP TABLE candidate_scores")
    op.execute("DROP TABLE date_evaluations")
