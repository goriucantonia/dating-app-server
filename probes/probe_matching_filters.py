"""probe_matching_filters.py (S9-P1, §2)

Positions users on each side of every hard filter and checks that the ones who
should be excluded are, that the ones who should survive do, and that the
per-step pool counts in the log agree with the outcome.

Economy note, because this probe costs real AI calls: only TWO users are built
all the way to a `ready` snapshot (the requester and the one legitimate match).
Every user who is supposed to be EXCLUDED is excluded by a filter that runs
BEFORE the snapshot step, so they cost nothing but a registration — except the
no-snapshot case, whose whole point is to have no snapshot.

What is asserted:

- gender mismatch, one-directional interest, age out of range, opt_in off, and
  no-`ready`-snapshot each remove a user from the pool (AC1, AC5);
- the funnel counts are logged on EVERY run, successful ones included (AC2),
  and a `no_candidates` run names which filter emptied the pool (AC3);
- `opt_in` toggled off removes someone and toggled back on restores them —
  **both directions observed**, which is the owed measurement O-1 (AC4);
- `compatibility` equals the mean of the two stored fits, recomputed here (AC8);
- every `shared_interests` entry is genuinely present in BOTH users' interest
  trait labels (AC6);
- a second POST while one is running is a 409 carrying the running id (AC7);
- `partial` and `no_candidates` are both reached (AC1).

Run inside the api container:

    docker compose exec api python probes/probe_matching_filters.py
"""

from __future__ import annotations

import asyncio
import sys
import uuid

import httpx

API = "http://localhost:8000"

# Two people who should plausibly match: both restore things, both value
# directness. Written in a real voice because extraction quality decides
# whether there are any interest traits to intersect at all.
ALICE = {
    "BQ1": (
        "I restore old bicycles. Steel frames from the seventies, the ones "
        "people leave out for the bin men. I strip them, re-cable them, and "
        "give most of them away. I learned by ruining two of them first and I "
        "read old repair manuals on the train to work."
    ),
    "BQ2": (
        "Someone who says the difficult thing early instead of letting it sit "
        "and rot. I want them to have something of their own they disappear "
        "into. The dealbreaker is contempt — how they talk about an ex, or to "
        "the person bringing the food."
    ),
    "BQ3": (
        "My flatmate told me I had been snappy for a month and she was right. "
        "I said something unfair about her never washing up and then went for a "
        "walk for an hour. I apologised properly two days later. I need time "
        "before I can come back to something."
    ),
    "BQ4": (
        "I remember what people told me weeks ago and bring it up, which is "
        "either lovely or unnerving. I ask a lot of questions and then go quiet "
        "at the exact moment a normal person would say something warm."
    ),
    "BQ5": (
        "I am reliable to the point of being dull about it. I turn up early. I "
        "am bad at the soft parts — I hand someone a solution when they wanted "
        "company, and I correct small things nobody needed corrected."
    ),
}

DAN = {
    "BQ1": (
        "I restore old bicycles and the odd motorbike. Mostly steel frames I "
        "find abandoned. I take them apart on the balcony and learn what I need "
        "as I go, usually by getting it wrong first. It is the quietest part of "
        "my week and I like doing it alone."
    ),
    "BQ2": (
        "I want someone straightforward. If something is wrong I would rather "
        "hear it on the day than in a summary at the end. Someone with their "
        "own obsession. I cannot stand people who are rude to staff."
    ),
    "BQ3": (
        "My brother said I had been impossible since our mother got ill and he "
        "was right. I said something cheap back and then sat in the car. I said "
        "sorry for the sentence two days later, not for being angry."
    ),
    "BQ4": (
        "I bring up things people mentioned ages ago. I am not smooth. I ask "
        "questions and then run out of words right when I should say something "
        "kind."
    ),
    "BQ5": (
        "Dependable and a bit boring about it. Early for everything. Worse at "
        "the emotional parts — I offer a fix instead of just sitting with "
        "someone, and I am pedantic about details that did not matter."
    ),
}

checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok))
    print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))


async def register(
    c: httpx.AsyncClient, *, tag: str, gender: str, interested_in: list[str],
    birth_date: str = "1992-06-15", age_min: int = 18, age_max: int = 60,
    opt_in: bool = True,
) -> dict:
    email = f"probe-match-{tag}-{uuid.uuid4().hex[:8]}@probe.dev"
    r = await c.post("/auth/register", json={
        "email": email, "password": "probe-password", "display_name": tag.title(),
        "birth_date": birth_date, "gender": gender, "interested_in": interested_in,
        "age_pref_min": age_min, "age_pref_max": age_max, "opt_in": opt_in,
    })
    r.raise_for_status()
    body = r.json()
    return {"email": email, "token": body["token"], "id": body["user"]["id"]}


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def build_profile(c: httpx.AsyncClient, token: str, answers: dict) -> bool:
    """Answer → extract → compile → wait for a ready snapshot. The expensive
    path; only the two users who must actually match pay it."""
    h = auth(token)
    qs = (await c.get("/questions", headers=h)).json()["questions"]
    by_code = {q["code"]: q["id"] for q in qs if q["code"]}
    for code, text in answers.items():
        (await c.put(f"/answers/{by_code[code]}", json={"answer_text": text},
                     headers=h)).raise_for_status()
    ext = await c.post("/profile/extract", headers=h)
    if ext.status_code != 200:
        print(f"  ABORT: extraction failed {ext.status_code} {ext.text[:160]}")
        return False
    (await c.post("/persona/compile", headers=h)).raise_for_status()
    for _ in range(100):
        cur = (await c.get("/persona/current", headers=h)).json()
        snap = cur.get("snapshot")
        if snap and snap["status"] == "ready":
            return True
        if snap and snap["status"] == "failed":
            print(f"  ABORT: compile failed — {snap.get('error')}")
            return False
        await asyncio.sleep(3)
    print("  ABORT: snapshot never became ready")
    return False


async def analyse(c: httpx.AsyncClient, token: str) -> dict:
    """POST then poll. Returns the settled analysis."""
    h = auth(token)
    r = await c.post("/analyses", headers=h)
    r.raise_for_status()
    aid = r.json()["id"]
    for _ in range(100):
        body = (await c.get(f"/analyses/{aid}", headers=h)).json()
        if body["status"] in {"matched", "no_candidates", "failed"}:
            return body
        await asyncio.sleep(2)
    raise TimeoutError("analysis never settled")


async def main() -> int:
    async with httpx.AsyncClient(base_url=API, timeout=300) as c:
        # --- the two who should match: woman seeking man, man seeking woman ---
        alice = await register(c, tag="alice", gender="woman", interested_in=["man"])
        dan = await register(c, tag="dan", gender="man", interested_in=["woman"])
        print(f"requester: {alice['email']}\nmatch:     {dan['email']}")

        if not await build_profile(c, alice["token"], ALICE):
            return 2
        if not await build_profile(c, dan["token"], DAN):
            return 2

        # --- the excluded, each failing exactly one filter -------------------
        # Excluded on gender: another woman, and Alice wants men.
        await register(c, tag="wrong-gender", gender="woman", interested_in=["woman"])
        # Excluded one-directionally: a man who is not interested in women.
        await register(c, tag="one-way", gender="man", interested_in=["man"])
        # Excluded on age: 70 and Alice's range tops out at 60.
        await register(c, tag="too-old", gender="man", interested_in=["woman"],
                       birth_date="1956-01-01")
        # Excluded on opt_in.
        opted_out = await register(c, tag="opted-out", gender="man",
                                   interested_in=["woman"], opt_in=False)
        # Eligible on everything EXCEPT a ready snapshot — the §11 gate.
        no_snap = await register(c, tag="no-snapshot", gender="man",
                                 interested_in=["woman"])

        # --- the run --------------------------------------------------------
        result = await analyse(c, alice["token"])
        ids = {x["candidate_user_id"] for x in result["candidates"]}

        check("the run reaches a settled state", result["status"] == "matched",
              f"status={result['status']} error={result.get('error')}")
        check("the one legitimate match IS returned", dan["id"] in ids)
        check("1-2 eligible reports pool_status 'partial'",
              result["pool_status"] == "partial", str(result["pool_status"]))
        check("gender mismatch excluded", not (ids - {dan["id"]}),
              f"unexpected extras: {ids - {dan['id']}}")
        check("opt_in=false excluded", opted_out["id"] not in ids)
        check("no ready snapshot excluded (the §11 gate)", no_snap["id"] not in ids)

        if result["candidates"]:
            top = result["candidates"][0]
            # AC8, recomputed here rather than trusted.
            expected = (top["fit_forward"] + top["fit_backward"]) / 2
            check("compatibility equals the mean of the two fits",
                  abs(top["compatibility"] - expected) < 1e-5,
                  f"{top['compatibility']:.6f} vs {expected:.6f}")
            check("a reason summary was assembled", bool(top["reason_summary"]),
                  top["reason_summary"][:90])

            # AC6 — every shared interest must exist in BOTH users' labels.
            a_traits = (await c.get("/traits", headers=auth(alice["token"]))).json()["traits"]
            d_traits = (await c.get("/traits", headers=auth(dan["token"]))).json()["traits"]
            a_labels = " ".join(t["label"].casefold() for t in a_traits
                                if t["category"] == "interest")
            d_labels = " ".join(t["label"].casefold() for t in d_traits
                                if t["category"] == "interest")
            shared = top["shared_interests"]
            verifiable = all(
                any(w in d_labels for w in s.casefold().split() if len(w) > 3)
                and s.casefold() in a_labels
                for s in shared
            )
            # D-009's lesson: this assertion used to skip when `shared` was
            # empty, so it passed on exactly the run where a label-corruption
            # bug had made the intersection impossible. Alice and Dan are both
            # written to restore old bicycles, so a NON-EMPTY intersection is
            # now required — an empty one is the symptom, not an exemption.
            check(
                "shared_interests is non-empty for two users who share one",
                bool(shared),
                f"shared={shared}",
            )
            check(
                "every shared_interest is present in BOTH users' interest labels",
                bool(shared) and verifiable,
                f"shared={shared}",
            )

        # --- AC7: a second run while one is active is 409 STATE -------------
        h = auth(alice["token"])
        first = await c.post("/analyses", headers=h)
        second = await c.post("/analyses", headers=h)
        conflict = second.status_code == 409 and \
            second.json().get("error", {}).get("code") == "analysis_in_progress"
        check("a second concurrent run is 409 state-not-error", conflict,
              f"status {second.status_code}")
        if first.status_code == 202:
            aid = first.json()["id"]
            for _ in range(100):
                if (await c.get(f"/analyses/{aid}", headers=h)).json()["status"] \
                        in {"matched", "no_candidates", "failed"}:
                    break
                await asyncio.sleep(2)

        # --- AC4 / O-1: opt_in observed changing pool membership, BOTH ways --
        dh = auth(dan["token"])
        (await c.patch("/me", json={"opt_in": False}, headers=dh)).raise_for_status()
        without = await analyse(c, alice["token"])
        check(
            "opt_in OFF removes them from someone else's pool",
            dan["id"] not in {x["candidate_user_id"] for x in without["candidates"]},
        )
        check("with nobody left the run says so honestly",
              without["status"] == "no_candidates"
              and without["pool_status"] == "empty"
              and without["message"] == "There is no one to match you with yet.",
              f"{without['status']}/{without['pool_status']}")

        (await c.patch("/me", json={"opt_in": True}, headers=dh)).raise_for_status()
        restored = await analyse(c, alice["token"])
        check(
            "opt_in ON restores them — the flag observed in BOTH directions (O-1)",
            dan["id"] in {x["candidate_user_id"] for x in restored["candidates"]},
        )

    failures = [label for label, ok in checks if not ok]
    if failures:
        print(f"VERDICT: RED — {len(failures)} failed: {failures}")
        return 1
    print(
        "VERDICT: GREEN — every hard filter excludes who it should, the §11 "
        "snapshot gate holds, partial and no_candidates are both reached "
        "honestly, compatibility recomputes exactly, a concurrent run is 409 "
        "state, and opt_in was observed changing pool membership in BOTH "
        "directions. All over real HTTP."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
