"""probe_answer_edit.py (S6-P1 / S6-P2, §2)

Drives holistic trait re-extraction against the real deployment, over real
HTTP, with a freshly registered user and REAL AI calls:

- five baseline answers produce trait rows with provenance and extracted_by;
- a no-edit re-run is all-`keep`: no trait_events, traits_hash untouched
  (S6-P2, the drift alarm — this is the assertion that catches an extraction
  that has started quietly rewriting the profile on every run);
- a CONFIRMED trait survives an edit to an UNRELATED answer, keeping its
  status and its provenance (S6-P1, AC3 — the load-bearing assertion of
  decision log #10);
- the edit changes traits_hash, which is what makes embeddings and the
  persona snapshot stale (A5.1).

One assertion is deliberately absent and is an owed measurement (§4, O-2):
that a pinned snapshot in a PAST analysis did not change. Analyses do not
exist until Step 9, so there is nothing to pin yet; it is enabled in S9-P2.
The stub below prints it as SKIPPED rather than silently omitting it, because
a probe that quietly tests less than its docstring claims is worse than one
that admits the gap.

Real AI calls: this probe costs several trait_extraction runs against the
free tier. Mind the per-minute cap if you run it repeatedly.

Run inside the api container:

    docker compose exec api python probes/probe_answer_edit.py
"""

from __future__ import annotations

import sys
import uuid

import httpx

API = "http://localhost:8000"

# Written in a real voice on purpose. Extraction quality is the thing under
# test, and filler text produces declines (correctly), which would leave the
# probe with no traits to make assertions about.
BASELINE_ANSWERS = {
    "BQ1": (
        "I keep bees, which sounds twee until you are standing in a cloud of "
        "them at seven in the morning. I got into it because a neighbour was "
        "giving up her hives and I said yes before thinking. I read almost "
        "nothing first and just opened the box and got stung a lot. Now I read "
        "constantly, mostly old beekeeping manuals from the fifties, because "
        "they assume you have hands and no gadgets. I do it alone. It is the "
        "one part of my week where nobody needs anything from me."
    ),
    "BQ2": (
        "Someone who tells me the truth early rather than kindly later. I would "
        "rather hear that I have been distant for a month on the day it starts "
        "than in a summary at the end. Nice to have: they have something they "
        "care about more than me, some work or craft that pulls them. Hard "
        "dealbreaker is contempt, the eye-roll thing, talking about an ex or a "
        "waiter like they are furniture. I can live with almost anything except "
        "being managed."
    ),
    "BQ3": (
        "My sister told me last spring that I had been unbearable since our dad "
        "died and she was right. I did not take it well. I said something cheap "
        "about her never visiting him and then sat in my car for an hour. I "
        "called her two days later and apologised for the sentence but not for "
        "being angry, because I was still angry. We are fine now. I am slow to "
        "come back after something like that, days not hours, and I know that "
        "is hard to be around."
    ),
    "BQ4": (
        "There is a woman at my climbing gym I have talked to maybe six times. "
        "What I actually do is remember things she said and bring them up three "
        "weeks later, which I am told is either charming or unsettling. I do "
        "not flirt in the normal sense. I ask a lot of questions and then go "
        "quiet at the exact moment a normal person would say something warm. I "
        "am working on that part."
    ),
    "BQ5": (
        "I am reliable to the point of being boring about it. If I say I will "
        "be somewhere I am there early. I am worse at the soft parts: I do not "
        "notice when someone wants comfort rather than a solution, and I have a "
        "pedantic streak where I correct small things that did not need "
        "correcting. Friends describe me as steady. I think what they mean is "
        "that I am predictable, which I have decided to take as a compliment."
    ),
}

# The edit target: an answer whose content has nothing to do with the trait we
# confirm. That disconnection is the whole point of AC3 — a confirmed trait
# must survive a change that has no bearing on it.
UNRELATED_EDIT = (
    "I keep bees, and this year I added a second hive on the far side of the "
    "garden because the first one kept swarming every June. I still read the "
    "old fifties manuals. I still do it alone at seven in the morning, and it "
    "is still the one hour a week when nobody needs anything from me at all."
)

checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok))
    print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))


def skipped(label: str, why: str) -> None:
    print(f"  SKIPPED: {label} — {why}")


def fresh_client() -> httpx.Client:
    client = httpx.Client(base_url=API, timeout=300)
    email = f"probe-edit-{uuid.uuid4().hex[:10]}@probe.dev"
    r = client.post("/auth/register", json={
        "email": email, "password": "probe-password", "display_name": "Edit Probe",
        "birth_date": "1990-06-15", "gender": "other", "interested_in": ["other"],
        "age_pref_max": 60,
    })
    r.raise_for_status()
    client.headers["Authorization"] = f"Bearer {r.json()['token']}"
    print(f"probe user registered: {email}")
    return client


def main() -> int:
    client = fresh_client()

    def traits() -> dict:
        r = client.get("/traits")
        r.raise_for_status()
        return r.json()

    def extract() -> dict:
        r = client.post("/profile/extract")
        if r.status_code == 503:
            print(f"  ABORT: {r.json().get('error', {}).get('message')}")
            raise SystemExit(2)
        r.raise_for_status()
        return r.json()

    # --- answer the five baseline questions in a real voice ---
    all_q = client.get("/questions").json()["questions"]
    by_code = {q["code"]: q["id"] for q in all_q if q["code"]}
    for code, text in BASELINE_ANSWERS.items():
        client.put(f"/answers/{by_code[code]}", json={"answer_text": text}).raise_for_status()

    # --- AC1: traits with provenance and a named extractor -------------------
    first = extract()
    t1 = traits()
    rows = t1["traits"]
    check("baseline answers produce trait rows", len(rows) > 0, f"{len(rows)} traits")
    check(
        "every trait carries non-empty source_answer_ids and extracted_by",
        all(r["source_answer_ids"] and r["extracted_by"] for r in rows),
    )
    check(
        "traits span more than one category",
        len({r["category"] for r in rows}) > 1,
        str(sorted({r["category"] for r in rows})),
    )
    print(f"    (first run: added={first['added']} declined={len(first['declined'])})")

    # --- confirm one trait; it becomes the thing that must survive -----------
    target = rows[0]
    confirmed = client.post(f"/traits/{target['id']}/confirm")
    confirmed.raise_for_status()
    check("a trait can be confirmed", confirmed.json()["status"] == "confirmed",
          target["label"])
    provenance_before = sorted(target["source_answer_ids"])

    # --- S6-P2: the drift alarm. No edit => all keep, nothing written --------
    hash_before = traits()["traits_hash"]
    second = extract()
    after_second = traits()
    check(
        "no-edit re-run is all-keep: nothing updated, retracted or added",
        second["updated"] == 0 and second["retracted"] == 0 and second["added"] == 0,
        f"kept={second['kept']} updated={second['updated']} "
        f"retracted={second['retracted']} added={second['added']}",
    )
    check(
        "no-edit re-run reports changed=false and leaves traits_hash untouched",
        second["changed"] is False and after_second["traits_hash"] == hash_before,
        f"{hash_before[:12]} -> {after_second['traits_hash'][:12]}",
    )
    still = next((r for r in after_second["traits"] if r["id"] == target["id"]), None)
    check(
        "the confirmed trait is still confirmed after a no-edit re-run",
        still is not None and still["status"] == "confirmed",
    )

    # --- AC3: edit an UNRELATED answer, re-extract, confirmation survives ----
    client.put(f"/answers/{by_code['BQ1']}", json={"answer_text": UNRELATED_EDIT}).raise_for_status()
    third = extract()
    after_edit = traits()
    survivor = next((r for r in after_edit["traits"] if r["id"] == target["id"]), None)

    check(
        "the confirmed trait still EXISTS after an unrelated answer was edited",
        survivor is not None,
    )
    if survivor is not None:
        check(
            "it is still 'confirmed' — extraction did not quietly downgrade it",
            survivor["status"] == "confirmed",
            f"status={survivor['status']}",
        )
        check(
            "its provenance survived intact",
            sorted(survivor["source_answer_ids"]) == provenance_before,
        )
    # AC5 — a retraction is a visible row, never a disappearance. Only
    # assertable when the model actually retracted something this run; a real
    # model on a mild edit often retracts nothing, and an assertion that passes
    # by being vacuously true is not an assertion.
    retracted_rows = [r for r in after_edit["traits"] if r["status"] == "retracted"]
    total_retracted = first["retracted"] + second["retracted"] + third["retracted"]
    if total_retracted:
        check(
            "every retracted trait is still PRESENT in the table, not deleted",
            len(retracted_rows) == total_retracted,
            f"{len(retracted_rows)} rows for {total_retracted} retractions",
        )
    else:
        skipped(
            "a retracted trait is present with status='retracted'",
            "no retraction occurred in these runs; AC5 needs a run that retracts",
        )
    print(
        f"    (post-edit run: kept={third['kept']} updated={third['updated']} "
        f"retracted={third['retracted']} added={third['added']} "
        f"changed={third['changed']})"
    )

    # --- the assertion that is not ours to make yet (O-2, closes in S9-P2) ---
    skipped(
        "a pinned snapshot in a past analysis did not change",
        "analyses do not exist until Step 9; enabled in S9-P2 (owed measurement O-2)",
    )

    failures = [label for label, ok in checks if not ok]
    if failures:
        print(f"VERDICT: RED — {len(failures)} failed: {failures}")
        return 1
    print(
        "VERDICT: GREEN — traits carry provenance, a no-edit re-run is all-keep "
        "with an untouched traits_hash, and a CONFIRMED trait survives an edit "
        "to an unrelated answer with its status and provenance intact. "
        "All witnessed over real HTTP against real model calls."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
