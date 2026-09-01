"""Persona snapshots and calibration — verbatim from trait_persona.md §4 (S7-B1).

Three tables: `persona_snapshots` (immutable, per-user versioned),
`calibration_sessions`, `calibration_messages`.

`persona_snapshots.system_prompt` is nullable BY DESIGN, not by oversight: a
row is INSERTed with status 'compiling' before the AI call, so a compilation
that dies mid-flight leaves evidence instead of nothing. It becomes non-NULL
when the row reaches 'ready'.

`calibration_sessions.snapshot_id` has NO ON DELETE CASCADE, matching the
document. That is the point of immutability — a calibration session must
always be able to say which persona version it was criticising, so the
snapshot it references cannot be deleted out from under it.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-01
"""
from __future__ import annotations

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # One op.execute per statement — asyncpg cannot prepare multi-command SQL.
    op.execute(
        """
CREATE TABLE persona_snapshots (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    version        INT  NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('compiling','ready','failed')),
    system_prompt  TEXT,
    schema_version TEXT NOT NULL,
    traits_hash    TEXT NOT NULL,
    source_trait_ids UUID[] NOT NULL,
    digest_model   TEXT,
    error          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, version)
);
"""
    )
    op.execute(
        """
CREATE TABLE calibration_sessions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    snapshot_id  UUID NOT NULL REFERENCES persona_snapshots(id),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
    )
    op.execute(
        """
CREATE TABLE calibration_messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES calibration_sessions(id) ON DELETE CASCADE,
    seq         INT  NOT NULL,
    sender      TEXT NOT NULL CHECK (sender IN ('user','persona')),
    text        TEXT NOT NULL,
    flagged     BOOLEAN NOT NULL DEFAULT FALSE,
    correction  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, seq)
);
"""
    )
    # The one query PersonaService.get_current_snapshot runs on every candidate
    # check in Step 9, so it gets its index in the same commit as the table.
    op.execute(
        "CREATE INDEX ix_persona_snapshots_current "
        "ON persona_snapshots (user_id, version DESC) WHERE status = 'ready'"
    )


def downgrade() -> None:
    op.execute("DROP TABLE calibration_messages")
    op.execute("DROP TABLE calibration_sessions")
    op.execute("DROP TABLE persona_snapshots")
