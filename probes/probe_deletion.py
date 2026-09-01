"""probe_deletion.py (S15-P1, §2)

A user entangled everywhere deletes their account. Verify the cascade counts
against a hand count taken beforehand, the survivor's tombstones, and that
the global questions survived.

    docker compose exec api python probes/probe_deletion.py <victim_email> <survivor_email> [password]

The VICTIM must be a candidate in at least one of the SURVIVOR's analyses
(entangled as a candidate with dates), and ideally the survivor's chosen
match in a chat session — the probe checks the entanglement before it
deletes anything, and refuses to run on an un-entangled pair, because a
deletion that touches nothing cross-user proves nothing about tombstones.

**The hand count is taken by this probe with its OWN SQL**, not by asking the
server's count list — otherwise the receipt would be checked against itself.

Cost: zero model calls. The victim's data is gone afterwards, which is the
point; pick a probe account.
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import create_async_engine

from app.logging_setup import setup_logging

API = "http://localhost:8000"

checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok))
    print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))


HAND_COUNTS = {
    "answers": "SELECT count(*) FROM answers WHERE user_id = :uid",
    "traits": "SELECT count(*) FROM traits WHERE user_id = :uid",
    "persona_snapshots": "SELECT count(*) FROM persona_snapshots WHERE user_id = :uid",
    "dates_as_candidate": "SELECT count(*) FROM dates WHERE candidate_user_id = :uid",
    "chat_sessions_as_match": "SELECT count(*) FROM chat_sessions WHERE match_user_id = :uid",
    "analysis_candidates_as_candidate":
        "SELECT count(*) FROM analysis_candidates WHERE candidate_user_id = :uid",
}


async def sql_scalar(engine, sql: str, **params):
    async with engine.connect() as conn:
        return (await conn.execute(sql_text(sql), params)).scalar_one()


async def main() -> int:
    setup_logging()  # the §7 lines are the evidence; a probe must print them
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    victim_email, survivor_email = sys.argv[1], sys.argv[2]
    password = sys.argv[3] if len(sys.argv) > 3 else "probe-password"
    engine = create_async_engine(os.environ["DATABASE_URL"])

    async with httpx.AsyncClient(base_url=API, timeout=60) as client:
        v = await client.post("/auth/login", json={"email": victim_email, "password": password})
        s = await client.post("/auth/login", json={"email": survivor_email, "password": password})
        if v.status_code != 200 or s.status_code != 200:
            print(f"login failed: victim {v.status_code}, survivor {s.status_code}")
            return 2
        vtok, stok = v.json()["token"], s.json()["token"]
        vid = v.json()["user"]["id"]
        vh = {"Authorization": f"Bearer {vtok}"}
        sh = {"Authorization": f"Bearer {stok}"}

        print("== before")
        hand = {k: int(await sql_scalar(engine, q, uid=vid)) for k, q in HAND_COUNTS.items()}
        questions_before = int(await sql_scalar(
            engine, "SELECT count(*) FROM questions WHERE user_id IS NULL"))
        print(f"  hand counts: {hand}; global questions: {questions_before}")

        entangled_analyses = [
            a for a in (await client.get("/analyses", headers=sh)).json()["analyses"]
            if any(c["candidate_user_id"] == vid for c in a["candidates"])
        ]
        check("victim is a candidate in the survivor's analyses",
              bool(entangled_analyses), f"{len(entangled_analyses)} analyses")
        if not entangled_analyses:
            print("refusing to delete an un-entangled pair")
            return 1
        target = entangled_analyses[0]
        before_dates = (await client.get(f"/analyses/{target['id']}/dates", headers=sh)).json()
        victim_dates_before = [d for d in before_dates["dates"] if d["candidate_user_id"] == vid]
        print(f"  survivor's analysis {target['id'][:8]}: {len(target['candidates'])} candidates, "
              f"{len(victim_dates_before)} dates with the victim")

        sessions_before = (await client.get("/chat/sessions", headers=sh)).json()["sessions"]
        victim_sessions = [x for x in sessions_before if x["match"]["user_id"] == vid]
        print(f"  survivor's chat sessions with the victim: {len(victim_sessions)}")

        print("== delete")
        r = await client.delete("/me", headers=vh)
        check("DELETE /me returns 200 with a receipt", r.status_code == 200, str(r.status_code))
        receipt = r.json().get("deleted", {})
        print(f"  receipt: {receipt}")
        for k, hand_n in hand.items():
            check(f"receipt[{k}] matches the hand count", receipt.get(k) == hand_n,
                  f"receipt {receipt.get(k)} vs hand {hand_n}")

        print("== after")
        gone = await client.get("/me", headers=vh)
        check("victim's token is dead", gone.status_code == 401, str(gone.status_code))
        check("victim row gone",
              int(await sql_scalar(engine, "SELECT count(*) FROM users WHERE id = :uid", uid=vid)) == 0)
        questions_after = int(await sql_scalar(
            engine, "SELECT count(*) FROM questions WHERE user_id IS NULL"))
        check("global baseline and pool questions survived",
              questions_after == questions_before, f"{questions_after}")

        after = (await client.get(f"/analyses/{target['id']}", headers=sh)).json()
        check("survivor's analysis row survives", after.get("id") == target["id"])
        check("victim no longer among its candidates",
              all(c["candidate_user_id"] != vid for c in after["candidates"]))
        check("the gap is labeled: removed_candidates >= 1",
              after.get("removed_candidates", 0) >= 1, str(after.get("removed_candidates")))
        after_dates = (await client.get(f"/analyses/{target['id']}/dates", headers=sh)).json()
        check("victim's dates gone from the survivor's analysis",
              all(d["candidate_user_id"] != vid for d in after_dates["dates"]))
        sessions_after = (await client.get("/chat/sessions", headers=sh)).json()["sessions"]
        check("survivor's chat sessions with the victim are gone",
              all(x["match"]["user_id"] != vid for x in sessions_after))
        if victim_sessions:
            g = await client.get(f"/chat/sessions/{victim_sessions[0]['session_id']}", headers=sh)
            check("opening the vanished chat is a 404 with a sentence, not a crash",
                  g.status_code == 404 and "message" in g.json().get("error", {}),
                  f"{g.status_code} {g.json().get('error', {}).get('message')}")

    await engine.dispose()
    failed = [l for l, ok in checks if not ok]
    print(f"\nprobe_deletion: {'GREEN' if not failed else 'RED'} — "
          f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
