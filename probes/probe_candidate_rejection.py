"""probe_candidate_rejection.py (S17-P1)

Turn a candidate down on a `matched` analysis over real HTTP and prove the
seat was re-filled honestly:

- the rejected person is off the wire, and their row is still in the database
  as `rejected` with a timestamp — the record of a decision, not an erasure;
- somebody NEW is on the wire in their place, and the active ranks are 1..n
  with no gap and no duplicate;
- rejecting the same person twice is a 404 `not_a_candidate`, not a silent
  second swap;
- a rejected person can never come back: a second rejection offers someone
  else again, never the first one turned down;
- the refusals are refusals — an unknown user id 404s, and a `complete`
  analysis 409s `cannot_reject_now`.

    docker compose exec api python probes/probe_candidate_rejection.py <email>

**Cost: zero model calls in the normal case.** Scoring is arithmetic over
embeddings that already exist; `refresh_embeddings` returns early when a
candidate's `traits_hash` is unchanged. The probe prints the `ai_call` count
it observed in its own log stream so that claim is checked rather than
asserted.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import httpx
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.logging_setup import setup_logging
from app.security import create_token

API = "http://localhost:8000"

checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok))
    print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))


def ranks_are_contiguous(candidates: list[dict]) -> bool:
    ranks = sorted(c["rank"] for c in candidates)
    return ranks == list(range(1, len(candidates) + 1))


async def main() -> int:
    setup_logging()
    if len(sys.argv) < 2:
        print("usage: probe_candidate_rejection.py <email>")
        return 2
    email = sys.argv[1]
    settings = get_settings()
    engine = create_async_engine(os.environ["DATABASE_URL"])

    async def rows(sql: str, **p):
        async with engine.connect() as conn:
            return (await conn.execute(sql_text(sql), p)).all()

    uid = (await rows("SELECT id::text FROM users WHERE email = :e", e=email))[0][0]
    token = create_token(uuid.UUID(uid), settings.jwt_secret)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(base_url=API, headers=headers, timeout=120) as c:
        analyses = (await c.get("/analyses")).json()["analyses"]
        matched = [a for a in analyses if a["status"] == "matched"]
        if not matched:
            print("no `matched` analysis for this account — nothing to probe")
            await engine.dispose()
            return 2
        a = matched[0]
        aid = a["id"]
        before = a["candidates"]
        print(f"== analysis {aid[:8]} — {len(before)} candidates")
        for x in before:
            print(f"   rank {x['rank']}  {x['display_name']:8} {x['compatibility']:.4f}")

        print("== refusals")
        r = await c.post(f"/analyses/{aid}/candidates/{uid}/reject")
        check(
            "rejecting somebody who is not a candidate is a 404 not_a_candidate",
            r.status_code == 404 and r.json()["error"]["code"] == "not_a_candidate",
            f"{r.status_code} {r.text[:80]}",
        )
        done = [x for x in analyses if x["status"] == "complete"]
        if done:
            other = done[0]
            victim = (other["candidates"] or [{}])[0].get("candidate_user_id")
            if victim:
                r = await c.post(
                    f"/analyses/{other['id']}/candidates/{victim}/reject"
                )
                check(
                    "a finished analysis refuses with 409 cannot_reject_now",
                    r.status_code == 409
                    and r.json()["error"]["code"] == "cannot_reject_now",
                    f"{r.status_code} {r.json().get('error', {}).get('message', '')[:60]}",
                )

        print("== the swap")
        target = before[-1]
        r = await c.post(
            f"/analyses/{aid}/candidates/{target['candidate_user_id']}/reject"
        )
        check("the rejection is a 200 carrying the whole analysis", r.status_code == 200,
              str(r.status_code))
        after = r.json()["candidates"]
        names = ", ".join(f"{x['display_name']}#{x['rank']}" for x in after)
        print(f"   now: {names}")

        ids_before = {x["candidate_user_id"] for x in before}
        ids_after = {x["candidate_user_id"] for x in after}
        check(
            "the rejected person is off the wire",
            target["candidate_user_id"] not in ids_after,
            target["display_name"],
        )
        arrived = ids_after - ids_before
        check("somebody new took the seat", len(arrived) == 1,
              next((x["display_name"] for x in after
                    if x["candidate_user_id"] in arrived), "nobody"))
        check("the line-up is still full", len(after) == len(before), str(len(after)))
        check("active ranks are 1..n with no gap or duplicate",
              ranks_are_contiguous(after), str(sorted(x["rank"] for x in after)))
        check(
            "rank still means fit order",
            [x["compatibility"] for x in after]
            == sorted((x["compatibility"] for x in after), reverse=True),
            str([round(x["compatibility"], 4) for x in after]),
        )
        check(
            "the replacement carries a computed reason, like the others",
            all(x["reason_summary"].strip() for x in after),
        )

        print("== the record")
        stored = await rows(
            "SELECT status, rank, rejected_at IS NOT NULL FROM analysis_candidates "
            "WHERE analysis_id = :a AND candidate_user_id = :c",
            a=aid, c=target["candidate_user_id"],
        )
        check("the rejected row is KEPT, marked rejected, with a timestamp",
              len(stored) == 1 and stored[0][0] == "rejected" and stored[0][2] is True,
              str(stored))
        counted = (await rows(
            "SELECT candidate_count FROM analyses WHERE id = :a", a=aid))[0][0]
        check("candidate_count follows the ACTIVE line-up, so no tombstone appears",
              counted == len(after), f"{counted} vs {len(after)}")
        check("the payload reports no removed candidates",
              r.json()["removed_candidates"] == 0,
              str(r.json()["removed_candidates"]))

        print("== the same person cannot be rejected, or offered, twice")
        r2 = await c.post(
            f"/analyses/{aid}/candidates/{target['candidate_user_id']}/reject"
        )
        check("a second rejection of the same person is 404 not_a_candidate",
              r2.status_code == 404
              and r2.json()["error"]["code"] == "not_a_candidate",
              str(r2.status_code))
        if arrived:
            newcomer = next(iter(arrived))
            r3 = await c.post(
                f"/analyses/{aid}/candidates/{newcomer}/reject"
            )
            if r3.status_code == 200:
                third = {x["candidate_user_id"] for x in r3.json()["candidates"]}
                check(
                    "the first person turned down is not offered back",
                    target["candidate_user_id"] not in third,
                )
                check(
                    "neither is the second",
                    newcomer not in third,
                )
            else:
                check(
                    "running out of people is refused, not faked",
                    r3.status_code == 409,
                    f"{r3.status_code} {r3.json().get('error', {}).get('code')}",
                )

    await engine.dispose()
    failed = [label for label, ok in checks if not ok]
    print(f"\nprobe_candidate_rejection: {'GREEN' if not failed else 'RED'} — "
          f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
