"""Enable the pgvector extension — the first migration, no tables yet (S1-B4).

Revision ID: 0001
Revises:
Create Date: 2026-09-01
"""
from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
