"""probe_demo_seeding.py (S15-P2, §2)

Wipe a demo user's traits, run the reconciliation pass, and verify it rebuilt
them THROUGH THE REAL PIPELINE — provenance present (`source_answer_ids`,
`extracted_by`), a `ready` snapshot, two embeddings from the pinned model —
rather than as shortcut rows.

    docker compose exec api python probes/probe_demo_seeding.py [demo_email]

Runs the pass in-process (`scripts/run_reconcile.py --wait` equivalent),
because the pass is not an HTTP action — and awaits the AI half inline
rather than racing a background copy of it. Everything it then checks it checks
through the API where one exists (`/me` for `is_demo`) and through SQL for
provenance, which the API deliberately does not expose.

Cost: one extraction, one compilation, one embedding call (the wiped user's
rebuild). The other demo users are no-ops on a healthy database, and the
probe asserts that too — that is the second-run witness (§1).
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import httpx
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import create_async_engine

from app.ai.registry import build_providers
from app.ai.routing import TaskRouter
from app.config import get_settings, load_ai_config
from app.demo import load_demo_profiles
from app.logging_setup import setup_logging
from app.reconcile import run_full_pass

API = "http://localhost:8000"

checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok))
    print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))


async def main() -> int:
    setup_logging()  # the §7 lines are the evidence; a probe must print them
    password, profiles = load_demo_profiles()
    email = sys.argv[1] if len(sys.argv) > 1 else profiles[0].email
    settings = get_settings()
    ai_config = load_ai_config(settings.ai_config_path)
    engine = create_async_engine(os.environ["DATABASE_URL"])
    router = TaskRouter(build_providers(ai_config), ai_config)
    app = SimpleNamespace(state=SimpleNamespace(engine=engine, ai_router=router, ai_config=ai_config))

    async def scalar(sql: str, **p):
        async with engine.connect() as conn:
            return (await conn.execute(sql_text(sql), p)).scalar_one()

    async def rows(sql: str, **p):
        async with engine.connect() as conn:
            return (await conn.execute(sql_text(sql), p)).all()

    # Every pass below is awaited INLINE. Launching the AI half in the
    # background and running it again would race the per-user extraction
    # lock — which is how this probe's first version produced a persona
    # built from nothing (see DEFECTS D-015).
    print("== first pass (creates anything missing; a no-op on a healthy database)")
    await run_full_pass(app, inline_demo_pipeline=True)
    uid = await scalar("SELECT id::text FROM users WHERE email = :e AND is_demo", e=email)
    check("demo user exists with is_demo", bool(uid), email)
    answers = int(await scalar("SELECT count(*) FROM answers WHERE user_id = :u", u=uid))
    check("five baseline answers seeded through the answer path", answers == 5, str(answers))

    print("== wipe the traits")
    async with engine.begin() as conn:
        await conn.execute(sql_text("DELETE FROM traits WHERE user_id = :u"), {"u": uid})
    check("traits wiped", int(await scalar("SELECT count(*) FROM traits WHERE user_id = :u", u=uid)) == 0)

    print("== second pass — must rebuild through the real pipeline")
    counts = (await run_full_pass(app, inline_demo_pipeline=True))["demo_pipeline"]
    print(f"  pipeline counts: {counts}")
    check("exactly one demo user was extracted (the wiped one); the rest were no-ops",
          counts["extracted"] == 1 and counts["failed"] == 0, str(counts))

    traits = await rows(
        "SELECT label, cardinality(source_answer_ids), coalesce(extracted_by,'') "
        "FROM traits WHERE user_id = :u AND status <> 'retracted'", u=uid)
    check("traits rebuilt", len(traits) > 0, f"{len(traits)} traits")
    check("every trait carries source_answer_ids", all(r[1] > 0 for r in traits))
    check("every trait carries extracted_by", all(r[2] for r in traits),
          (traits[0][2] if traits else ""))
    snap = await rows(
        "SELECT version, status FROM persona_snapshots WHERE user_id = :u "
        "ORDER BY version DESC LIMIT 1", u=uid)
    check("newest snapshot is ready", bool(snap) and snap[0][1] == "ready",
          f"v{snap[0][0]} {snap[0][1]}" if snap else "none")
    emb = await rows("SELECT kind, embedding_model FROM profile_embeddings WHERE user_id = :u", u=uid)
    expected = f"{ai_config.embeddings.provider}/{ai_config.embeddings.model}"
    check("two embeddings from the pinned model",
          len(emb) == 2 and all(m == expected for _, m in emb), str(emb))

    print("== third pass — a healthy database is a no-op")
    counts = (await run_full_pass(app, inline_demo_pipeline=True))["demo_pipeline"]
    check("no-op on a healthy database", counts["ok"] == len(profiles) and
          counts["extracted"] == counts["compiled"] == counts["embedded"] == 0, str(counts))

    async with httpx.AsyncClient(base_url=API, timeout=30) as client:
        r = await client.post("/auth/login", json={"email": email, "password": password})
        check("demo user can sign in with the fixture password", r.status_code == 200)
        if r.status_code == 200:
            check("is_demo travels on /me", r.json()["user"]["is_demo"] is True)

    await engine.dispose()
    failed = [l for l, ok in checks if not ok]
    print(f"\nprobe_demo_seeding: {'GREEN' if not failed else 'RED'} — "
          f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
