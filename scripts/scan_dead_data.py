"""Dead-data scan (S15-B7, §22, `data_hygiene.md` §2). REPORT ONLY.

Prints, and deletes nothing:

- users with zero answers whose account is older than 30 days (abandoned
  registrations);
- persona snapshots in `failed`;
- persona snapshots stuck in `compiling` for more than 10 minutes (a process
  died mid-call; the row is a tombstone of that, not a persona);
- analyses in `failed`;
- dates still `running` whose analysis is NOT `simulating` (orphaned by a
  crash the relaunch pass did not pick up — there should be none);
- chat sessions with no messages older than 7 days.

Deleting real users' data is always a human decision. This script hands the
human the list.

    docker compose exec api python scripts/scan_dead_data.py
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.config import get_settings
from app.db import create_engine

REPORTS: list[tuple[str, str]] = [
    (
        "users with zero answers, registered more than 30 days ago",
        (
            "SELECT u.id::text, u.email, u.created_at::date::text FROM users u "
            "WHERE NOT EXISTS (SELECT 1 FROM answers a WHERE a.user_id = u.id) "
            "AND u.created_at < now() - interval '30 days' ORDER BY u.created_at"
        ),
    ),
    (
        "persona snapshots in `failed`",
        (
            "SELECT id::text, user_id::text, version::text, left(coalesce(error,''), 80) "
            "FROM persona_snapshots WHERE status = 'failed' ORDER BY created_at"
        ),
    ),
    (
        "persona snapshots stuck in `compiling` for more than 10 minutes",
        (
            "SELECT id::text, user_id::text, version::text, created_at::text "
            "FROM persona_snapshots WHERE status = 'compiling' "
            "AND created_at < now() - interval '10 minutes' ORDER BY created_at"
        ),
    ),
    (
        "analyses in `failed`",
        (
            "SELECT id::text, user_id::text, created_at::date::text, "
            "left(coalesce(error,''), 80) "
            "FROM analyses WHERE status = 'failed' ORDER BY created_at"
        ),
    ),
    (
        "dates still `running` whose analysis is not `simulating` (orphans)",
        (
            "SELECT d.id::text, d.analysis_id::text, a.status, d.created_at::date::text "
            "FROM dates d JOIN analyses a ON a.id = d.analysis_id "
            "WHERE d.status = 'running' AND a.status <> 'simulating' ORDER BY d.created_at"
        ),
    ),
    (
        "chat sessions with no messages, older than 7 days",
        (
            "SELECT s.id::text, s.user_id::text, s.status, s.created_at::date::text "
            "FROM chat_sessions s "
            "WHERE NOT EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id = s.id) "
            "AND s.created_at < now() - interval '7 days' ORDER BY s.created_at"
        ),
    ),
]


async def main() -> None:
    engine = create_engine(get_settings().database_url)
    total = 0
    try:
        async with engine.connect() as conn:
            for title, sql in REPORTS:
                rows = (await conn.execute(text(sql))).all()
                total += len(rows)
                print(f"\n== {title}: {len(rows)}")
                for r in rows:
                    print("   " + " | ".join(str(c) for c in r))
    finally:
        await engine.dispose()
    print(f"\n{total} item(s) reported. Nothing was deleted; nothing will be by this script.")


if __name__ == "__main__":
    asyncio.run(main())
