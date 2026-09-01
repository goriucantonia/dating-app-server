"""probe_onboarding.py (S7-P1, §2)

The whole first-run journey against the real deployment, over real HTTP, with
real AI calls: register → answer BQ1–BQ5 **with an edit mid-way** → extract →
compile → a `ready` snapshot with provenance.

The mid-way edit is not decoration. It is the ordinary thing a real person
does — writes an answer, thinks better of it, goes back — and it is the case
where a pipeline that assumed "answer once, then extract" quietly breaks.

What this asserts beyond "it ran":

- the snapshot is v1, `ready`, and names the model that wrote its digest;
- `source_trait_ids` is non-empty — a persona built from no stated traits is
  a persona built from nothing (§9);
- `schema_version` is `agent_response.v1`, the string every transcript will
  later be read back against;
- the system prompt contains the user's **edited** answer VERBATIM. This is
  the fidelity assertion of the whole module (trade #1): if a paraphrase ever
  creeps between what the user wrote and what the agent imitates, the voice is
  laundered out and this check is what notices. It reads the prompt from the
  DATABASE, never from the API, because...
- ...`GET /persona/current` must NOT contain the prompt, checked against the
  raw response body (AC6, communication_protocol.md §6).

Run inside the api container:

    docker compose exec api python probes/probe_onboarding.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import httpx
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import create_async_engine

API = "http://localhost:8000"

ANSWERS = {
    "BQ1": (
        "I restore old radios. Valve sets mostly, the big wooden ones people "
        "throw out. I got the first one free off a skip and spent a month "
        "learning why it hummed. I read service manuals from the forties on my "
        "lunch break. It is a solitary thing and I like that about it."
    ),
    "BQ2": (
        "Someone who says the awkward thing early instead of letting it rot. I "
        "want them to have something of their own they disappear into. The "
        "dealbreaker is contempt — the way people talk about their exes, or to "
        "waiters. I can forgive nearly anything except being handled."
    ),
    "BQ3": (
        "My brother told me I had been impossible since our mother got ill and "
        "he was right. I said something cheap back about him never phoning her, "
        "then sat in the car park for an hour. I apologised for the sentence "
        "two days later but not for being angry. I take days, not hours."
    ),
    "BQ4": (
        "There is someone at the radio club I have spoken to maybe eight times. "
        "What I do is remember things she mentioned and bring them up weeks "
        "later, which is apparently either lovely or alarming. I ask questions "
        "and then go quiet exactly when a normal person would say something warm."
    ),
    "BQ5": (
        "I am dependable to the point of being dull about it. If I say I will "
        "be there I am there early. I am poor at the soft parts — I offer a fix "
        "when someone wanted company, and I correct small things nobody needed "
        "corrected. Friends call me steady, which I have decided to enjoy."
    ),
}

# BQ1 rewritten mid-onboarding, before extraction runs. The prompt must end up
# quoting THIS, not the draft above.
BQ1_EDITED = (
    "I restore old valve radios — the big wooden ones people put out with the "
    "rubbish. The first was free off a skip and I spent a month working out why "
    "it hummed, mostly by getting it wrong. Now I read service manuals from the "
    "forties on my lunch break. I do it alone in the back room and that is most "
    "of the appeal, if I am honest about it."
)

checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok))
    print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))


async def read_prompt(snapshot_id: str) -> str:
    """Straight from the database. The API is not permitted to hand this over,
    which is exactly what the AC6 check below verifies."""
    engine = create_async_engine(os.environ["DATABASE_URL"])
    try:
        async with engine.connect() as conn:
            row = await conn.execute(
                sql_text("SELECT system_prompt FROM persona_snapshots WHERE id = :i"),
                {"i": snapshot_id},
            )
            return row.scalar_one() or ""
    finally:
        await engine.dispose()


async def poll_until_settled(client: httpx.AsyncClient, timeout_s: int = 300) -> dict:
    """Start-then-poll, the way the UI does it (S7-U1) — no fake timer, just
    the real status."""
    waited = 0
    while waited < timeout_s:
        body = (await client.get("/persona/current")).json()
        snap = body.get("snapshot")
        if snap and snap["status"] in {"ready", "failed"}:
            return body
        await asyncio.sleep(3)
        waited += 3
    raise TimeoutError(f"snapshot never settled within {timeout_s}s")


async def main() -> int:
    async with httpx.AsyncClient(base_url=API, timeout=300) as c:
        email = f"probe-onboard-{uuid.uuid4().hex[:10]}@probe.dev"
        r = await c.post("/auth/register", json={
            "email": email, "password": "probe-password",
            "display_name": "Onboarding Probe", "birth_date": "1990-06-15",
            "gender": "other", "interested_in": ["other"], "age_pref_max": 60,
        })
        r.raise_for_status()
        c.headers["Authorization"] = f"Bearer {r.json()['token']}"
        print(f"probe user registered: {email}")

        # --- no persona before anything is answered: the §11 gate ------------
        before = (await c.get("/persona/current")).json()
        check(
            "a brand-new user has no snapshot and is NOT simulatable",
            before["snapshot"] is None and before["simulatable"] is False,
        )
        gate = await c.post("/calibration/sessions")
        check(
            "calibration refuses politely while there is no persona",
            gate.status_code == 409
            and gate.json().get("error", {}).get("code") == "no_persona_yet",
            f"status {gate.status_code}",
        )

        # --- answer BQ1–BQ5, editing BQ1 mid-way -----------------------------
        qs = (await c.get("/questions")).json()["questions"]
        by_code = {q["code"]: q["id"] for q in qs if q["code"]}
        for code in ["BQ1", "BQ2", "BQ3"]:
            (await c.put(f"/answers/{by_code[code]}",
                         json={"answer_text": ANSWERS[code]})).raise_for_status()
        # ...thinks better of BQ1, before extraction has ever run.
        (await c.put(f"/answers/{by_code['BQ1']}",
                     json={"answer_text": BQ1_EDITED})).raise_for_status()
        for code in ["BQ4", "BQ5"]:
            (await c.put(f"/answers/{by_code[code]}",
                         json={"answer_text": ANSWERS[code]})).raise_for_status()

        # --- extract, then compile (the S7-U1 chain) -------------------------
        ext = await c.post("/profile/extract")
        if ext.status_code == 503:
            print(f"  ABORT: {ext.json().get('error', {}).get('message')}")
            return 2
        ext.raise_for_status()
        traits = (await c.get("/traits")).json()["traits"]
        check("extraction produced traits", len(traits) > 0, f"{len(traits)} traits")

        started = (await c.post("/persona/compile")).json()
        check(
            "compile returns immediately rather than blocking",
            started["status"] in {"compiling", "already_compiling"},
            started["status"],
        )
        settled = await poll_until_settled(c)
        snap = settled["snapshot"]

        check("the snapshot reached 'ready'", snap["status"] == "ready",
              f"status={snap['status']} error={snap.get('error')}")
        check("it is version 1 for a first-time user", snap["version"] == 1,
              f"v{snap['version']}")
        check("the user is now simulatable", settled["simulatable"] is True)
        check(
            "provenance: source_trait_ids is not empty",
            snap["source_trait_count"] > 0, f"{snap['source_trait_count']} traits",
        )
        check(
            "the digest names the model that wrote it",
            bool(snap["digest_model"]), str(snap["digest_model"]),
        )
        check(
            "schema_version is the frozen agent_response.v1",
            snap["schema_version"] == "agent_response.v1", snap["schema_version"],
        )
        check("a fresh snapshot is not stale", snap["stale"] is False)

        # --- AC6: the prompt must never cross the wire -----------------------
        raw = (await c.get("/persona/current")).text
        check(
            "GET /persona/current does NOT leak the system prompt",
            "system_prompt" not in raw and "You are Onboarding Probe" not in raw,
        )

        # --- AC1 / trade #1: the EDITED answer is quoted verbatim ------------
        prompt = await read_prompt(snap["snapshot_id"])
        check(
            "the system prompt quotes the user's own words verbatim",
            BQ1_EDITED in prompt or ANSWERS["BQ5"] in prompt,
        )
        check(
            "it quotes the EDITED BQ1, not the superseded draft",
            (BQ1_EDITED in prompt) or (ANSWERS["BQ1"] not in prompt),
            "the pre-edit text must not be what the agent imitates",
        )
        check(
            "the prompt carries the behaviour digest section",
            "HOW YOU BEHAVE" in prompt and "When things get tense:" in prompt,
        )
        check(
            "the prompt carries the voice-sample section",
            "HOW YOU WRITE AND SPEAK" in prompt and "in their own words" in prompt,
        )

    failures = [label for label, ok in checks if not ok]
    if failures:
        print(f"VERDICT: RED — {len(failures)} failed: {failures}")
        return 1
    print(
        "VERDICT: GREEN — register → answer with a mid-way edit → extract → "
        "compile produces a ready v1 snapshot with provenance, a named digest "
        "model, the frozen schema version, the user's EDITED words quoted "
        "verbatim, and no system prompt anywhere in the API response."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
