"""Two guards from the 2026-09-02 audit.

`chat_messages.client_message_id`: the client's own id for a send, unique per
session where present, so a resend after a client timeout returns the stored
pair rather than posting the same line twice (D-022).

`ux_analyses_one_active`: at most one `matching`/`simulating` analysis per
user, enforced by the database rather than by a SELECT-then-INSERT the
double-tap could race (audit finding; the router turns the violation into the
same 409 it already speaks).

Revision ID: 0013
Revises: 0012
Create Date: 2026-09-02
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_messages", sa.Column("client_message_id", sa.Text(), nullable=True)
    )
    op.create_index(
        "ux_chat_messages_client_id", "chat_messages",
        ["session_id", "client_message_id"], unique=True,
        postgresql_where=sa.text("client_message_id IS NOT NULL"),
    )
    op.create_index(
        "ux_analyses_one_active", "analyses", ["user_id"], unique=True,
        postgresql_where=sa.text("status IN ('matching','simulating')"),
    )


def downgrade() -> None:
    op.drop_index("ux_analyses_one_active", table_name="analyses")
    op.drop_index("ux_chat_messages_client_id", table_name="chat_messages")
    op.drop_column("chat_messages", "client_message_id")
