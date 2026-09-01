"""Account deletion (S15-B1, B2, B3 — `data_hygiene.md` §2).

One transaction: `DELETE FROM users WHERE id = …`, and the cascade graph does
the rest. **The graph is verified, not assumed** (§13): `cascade_tables()`
walks the ORM metadata from `users` along `ON DELETE CASCADE` foreign keys,
and a unit test asserts that every table it reaches has a line in
[USER_ROW_COUNTS] — so a future table that hangs off `users` without a count
fails a test rather than silently vanishing uncounted.

**Counts BEFORE the cascade** (§19, §7): the per-table row counts are the
deletion trace without retaining the data. They are logged and returned to
the client as the receipt. Taking them after would count nothing.

The two cross-user effects are counted and logged by name (S15-B3), because
they are the ones a survivor will notice: dates and chats where the deleted
person was the OTHER party disappear from someone else's history. That is
the named trade — their persona, answers and simulated behaviour are their
data, and privacy beats history.

Baseline and pool questions have `user_id IS NULL` and no path from `users`;
they survive every deletion, and the probe checks that they did.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import MetaData, delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging_setup import log_event
from app.models import Base, User

logger = logging.getLogger("app.deletion")

# (label, count SQL). Every table the cascade can reach from `users`, counted
# for THIS user, before anything is deleted. The labels distinguish the
# cross-user rows ("…_as_candidate", "…_as_match") from the user's own.
USER_ROW_COUNTS: list[tuple[str, str]] = [
    ("answers", "SELECT count(*) FROM answers WHERE user_id = :uid"),
    ("dispute_questions", "SELECT count(*) FROM questions WHERE user_id = :uid"),
    ("traits", "SELECT count(*) FROM traits WHERE user_id = :uid"),
    ("trait_events", (
        "SELECT count(*) FROM trait_events e JOIN traits t ON t.id = e.trait_id "
        "WHERE t.user_id = :uid"
    )),
    ("profile_embeddings", "SELECT count(*) FROM profile_embeddings WHERE user_id = :uid"),
    ("persona_snapshots", "SELECT count(*) FROM persona_snapshots WHERE user_id = :uid"),
    ("calibration_sessions", "SELECT count(*) FROM calibration_sessions WHERE user_id = :uid"),
    ("calibration_messages", (
        "SELECT count(*) FROM calibration_messages m "
        "JOIN calibration_sessions s ON s.id = m.session_id WHERE s.user_id = :uid"
    )),
    ("analyses", "SELECT count(*) FROM analyses WHERE user_id = :uid"),
    ("analysis_candidates_in_own_analyses", (
        "SELECT count(*) FROM analysis_candidates c "
        "JOIN analyses a ON a.id = c.analysis_id WHERE a.user_id = :uid"
    )),
    ("analysis_candidates_as_candidate", (
        "SELECT count(*) FROM analysis_candidates WHERE candidate_user_id = :uid"
    )),
    ("dates_in_own_analyses", (
        "SELECT count(*) FROM dates d JOIN analyses a ON a.id = d.analysis_id "
        "WHERE a.user_id = :uid"
    )),
    ("dates_as_candidate", "SELECT count(*) FROM dates WHERE candidate_user_id = :uid"),
    ("date_messages", (
        "SELECT count(*) FROM date_messages m JOIN dates d ON d.id = m.date_id "
        "JOIN analyses a ON a.id = d.analysis_id "
        "WHERE a.user_id = :uid OR d.candidate_user_id = :uid"
    )),
    ("date_evaluations", (
        "SELECT count(*) FROM date_evaluations e JOIN dates d ON d.id = e.date_id "
        "JOIN analyses a ON a.id = d.analysis_id "
        "WHERE a.user_id = :uid OR d.candidate_user_id = :uid"
    )),
    ("candidate_scores", (
        "SELECT count(*) FROM candidate_scores s JOIN analyses a ON a.id = s.analysis_id "
        "WHERE a.user_id = :uid OR s.candidate_user_id = :uid"
    )),
    ("chat_sessions_own", "SELECT count(*) FROM chat_sessions WHERE user_id = :uid"),
    ("chat_sessions_as_match", "SELECT count(*) FROM chat_sessions WHERE match_user_id = :uid"),
    ("chat_messages", (
        "SELECT count(*) FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id "
        "WHERE s.user_id = :uid OR s.match_user_id = :uid"
    )),
]

# The survivors' side, logged so the tombstone story is reconstructible from
# the log alone (§7): which OTHER users' analyses and chats lose a person.
SURVIVOR_QUERIES: dict[str, str] = {
    "analyses_of_others_losing_a_candidate": (
        "SELECT DISTINCT a.id::text FROM analysis_candidates c "
        "JOIN analyses a ON a.id = c.analysis_id "
        "WHERE c.candidate_user_id = :uid AND a.user_id <> :uid"
    ),
    "chat_sessions_of_others_losing_their_match": (
        "SELECT id::text FROM chat_sessions WHERE match_user_id = :uid AND user_id <> :uid"
    ),
}


def cascade_tables(metadata: MetaData = Base.metadata) -> set[str]:
    """Every table reachable from `users` along ON DELETE CASCADE foreign
    keys, transitively. The set the counts must cover."""
    reached: set[str] = set()
    frontier = ["users"]
    while frontier:
        parent = frontier.pop()
        for table in metadata.tables.values():
            for fk in table.foreign_keys:
                if (
                    fk.column.table.name == parent
                    and (fk.ondelete or "").upper() == "CASCADE"
                    and table.name not in reached
                    and table.name != "users"
                ):
                    reached.add(table.name)
                    frontier.append(table.name)
    return reached


def counted_tables() -> set[str]:
    """The table names the count list actually touches (parsed from the SQL)."""
    names: set[str] = set()
    for _, sql in USER_ROW_COUNTS:
        tokens = sql.replace(",", " ").split()
        for i, tok in enumerate(tokens):
            if tok.upper() in ("FROM", "JOIN") and i + 1 < len(tokens):
                names.add(tokens[i + 1])
    return names


async def count_before_delete(session: AsyncSession, user_id: uuid.UUID) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, sql in USER_ROW_COUNTS:
        counts[label] = int((await session.execute(text(sql), {"uid": user_id})).scalar_one())
    return counts


async def delete_account(session: AsyncSession, user: User) -> dict[str, int]:
    """Counts, log, cascade, log. Returns the counts as the receipt."""
    uid = user.id
    is_demo = user.is_demo
    counts = await count_before_delete(session, uid)
    survivors = {
        key: list((await session.execute(text(sql), {"uid": uid})).scalars())
        for key, sql in SURVIVOR_QUERIES.items()
    }
    # §19: the trace is written BEFORE the cascade, or there is nothing to trace.
    log_event(
        logger, "account_deletion_counts", level=logging.WARNING,
        user_id=str(uid), is_demo=is_demo, counts=counts, **survivors,
    )
    await session.execute(delete(User).where(User.id == uid))
    await session.commit()
    log_event(
        logger, "account_deleted", level=logging.WARNING,
        user_id=str(uid), rows_removed=sum(counts.values()),
    )
    return counts
