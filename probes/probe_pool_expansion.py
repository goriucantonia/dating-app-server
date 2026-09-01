"""probe_pool_expansion.py (S5-P1, §2)

Drives the pool-expansion mechanism end to end against the real deployment,
over real HTTP, with a freshly registered user:

- batches of 5 served strictly in pool_order;
- a batch abandoned mid-way resumes with the same remaining questions;
- baseline answers never count toward answered_pool (§13 / S5-B6);
- a 199-character answer is rejected with the 422 envelope;
- an edit travels the identical upsert path and bumps updated_at;
- exactly 6 batches exhaust the pool; batch 7 is the EXACT
  `pool_exhausted` payload (a normal 200, never a 4xx).

Run inside the api container:

    docker compose exec api python probes/probe_pool_expansion.py
"""

from __future__ import annotations

import sys
import uuid

import httpx

API = "http://localhost:8000"
FILLER = (
    "This is a probe answer written to satisfy the two-hundred-character "
    "minimum that applies to every answer in this system, baseline and pool "
    "and dispute alike, as principle eighteen insists the scope be written down. "
)

checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok))
    print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))


def fresh_client(label: str) -> httpx.Client:
    client = httpx.Client(base_url=API, timeout=30)
    email = f"probe-pool-{label}-{uuid.uuid4().hex[:10]}@probe.dev"
    r = client.post("/auth/register", json={
        "email": email, "password": "probe-password", "display_name": "Pool Probe",
        "birth_date": "1990-06-15", "gender": "other", "interested_in": ["other"],
        "age_pref_max": 60,
    })
    r.raise_for_status()
    client.headers["Authorization"] = f"Bearer {r.json()['token']}"
    print(f"probe user registered: {email}")
    return client


def main() -> int:
    # User A: rejection, resume, edit — the messy path.
    client = fresh_client("a")

    def batch() -> dict:
        r = client.get("/questions/next-batch")
        r.raise_for_status()
        return r.json()

    def answer(question_id: str, text: str = "") -> httpx.Response:
        return client.put(f"/answers/{question_id}", json={"answer_text": text or FILLER})

    # --- 199 characters rejected, with the envelope ---
    b = batch()
    short = answer(b["questions"][0]["id"], "x" * 199)
    check(
        "199-char answer rejected with 422 envelope",
        short.status_code == 422 and short.json().get("error", {}).get("code") == "validation_error",
        f"status {short.status_code}",
    )

    # --- baseline answers never count toward answered_pool ---
    all_q = client.get("/questions").json()["questions"]
    bq1 = next(q for q in all_q if q["code"] == "BQ1")
    answer(bq1["id"]).raise_for_status()
    check(
        "baseline answer does not bump answered_pool",
        batch()["progress"]["answered_pool"] == 0,
    )

    # --- batch 1 in pool_order; abandon after 3; resume shows the remainder ---
    b1 = batch()
    codes = [q["code"] for q in b1["questions"]]
    check("batch 1 is PQ01–PQ05 in pool_order", codes == ["PQ01", "PQ02", "PQ03", "PQ04", "PQ05"], str(codes))
    for q in b1["questions"][:3]:
        answer(q["id"]).raise_for_status()
    resumed = batch()
    resumed_codes = [q["code"] for q in resumed["questions"]]
    check(
        "abandoned batch resumes with the same remaining questions first",
        resumed_codes[:2] == ["PQ04", "PQ05"],
        str(resumed_codes),
    )
    check("progress counts the 3 answered", resumed["progress"]["answered_pool"] == 3)

    # --- an edit travels the same upsert path and bumps updated_at ---
    first_saved = client.get("/questions").json()["questions"]
    pq01 = next(q for q in first_saved if q["code"] == "PQ01")
    before = pq01["answer_updated_at"]
    edited = answer(pq01["id"], FILLER + " Edited for the probe's updated_at assertion.")
    edited.raise_for_status()
    check(
        "edit bumps updated_at through the identical PUT path",
        edited.json()["answer_updated_at"] > before,
        f"{before} -> {edited.json()['answer_updated_at']}",
    )

    # --- User B: the straight run — exactly 6 batches of 5, batch 7 exhausted
    # (AC6). User A's abandon added an extra fetch, so the clean arithmetic
    # needs a user who never abandons.
    client_b = fresh_client("b")

    def batch_b() -> dict:
        r = client_b.get("/questions/next-batch")
        r.raise_for_status()
        return r.json()

    served_codes: list[str] = []
    batches_served = 0
    while True:
        b = batch_b()
        if b["status"] == "pool_exhausted":
            break
        batches_served += 1
        codes = [q["code"] for q in b["questions"]]
        served_codes += codes
        in_order = [q["pool_order"] for q in b["questions"]] == sorted(
            q["pool_order"] for q in b["questions"]
        )
        if not in_order:
            check(f"batch {batches_served} served in pool_order", False, str(codes))
        for q in b["questions"]:
            client_b.put(f"/answers/{q['id']}", json={"answer_text": FILLER}).raise_for_status()
        if batches_served > 10:
            check("runaway batch loop", False, "more than 10 batches served")
            break

    check("exactly 6 batches exhaust the pool (30 ÷ 5)", batches_served == 6, f"{batches_served} batches")
    check(
        "all 30 pool questions served exactly once, in pool_order",
        served_codes == [f"PQ{n:02d}" for n in range(1, 31)],
    )

    final = batch_b()
    expected = {
        "status": "pool_exhausted",
        "questions": [],
        "progress": {"answered_pool": 30, "total_pool": 30},
    }
    check("exhausted payload matches the defined shape exactly", final == expected, str(final))

    failures = [label for label, ok in checks if not ok]
    if failures:
        print(f"VERDICT: RED — {len(failures)} failed: {failures}")
        return 1
    print("VERDICT: GREEN — batches of 5 in pool_order, mid-batch resume, "
          "baseline excluded from pool progress, edit path identical, and the "
          "exact pool_exhausted payload. All witnessed over real HTTP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
