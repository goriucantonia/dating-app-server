"""Chat sessions and messages — verbatim from chat.md §3 (S14-B1).

Two constraints carry the module's argument:

- **`UNIQUE (analysis_id)` on `chat_sessions`** — one selection per analysis,
  enforced by the database rather than by a check in the endpoint that a
  concurrent second POST could slip past. "One selection per analysis" does
  NOT mean one chat ever (§18): a new analysis makes a new session.
- **`snapshot_id` has NO cascade** — a session must always be able to name
  the persona version it was talking to. It pins the candidate's MATCHED
  snapshot, the same one the dates ran against, so the user chats with the
  person whose transcripts they read rather than a recompiled newer self.

`chat_messages.state` is where the persona's inner state lands — stored, and
stripped from every chat payload by contract (`communication_protocol.md` §6).
It is nullable because a user's message has no inner state to store.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-01
"""
from __future__ import annotations

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # One op.execute per statement — asyncpg cannot prepare multi-command SQL.
    op.execute(
        """
CREATE TABLE chat_sessions (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    match_user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    analysis_id        UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    snapshot_id        UUID NOT NULL REFERENCES persona_snapshots(id),
    date_digest        TEXT NOT NULL,
    summary            TEXT,
    compacted_upto_seq INT NOT NULL DEFAULT 0,
    status             TEXT NOT NULL CHECK (status IN ('active','ended')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at           TIMESTAMPTZ,
    UNIQUE (analysis_id)
);
"""
    )
    op.execute(
        """
CREATE TABLE chat_messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    seq         INT  NOT NULL,
    sender      TEXT NOT NULL CHECK (sender IN ('user','persona')),
    text        TEXT NOT NULL,
    state       JSONB,
    provider    TEXT, model_id TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, seq)
);
"""
    )


def downgrade() -> None:
    op.execute("DROP TABLE chat_messages")
    op.execute("DROP TABLE chat_sessions")
