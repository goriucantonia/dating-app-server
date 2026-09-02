"""probe_judge.py (S12-P1, §2)

Two assertions, and they are the two things that make a score worth showing
someone:

1. **The same transcript judged twice lands in about the same place.** A judge
   whose answer wanders is not a measuring instrument, it is a mood. This is
   why `judging` runs at temperature 0.1.
2. **`date_score` recomputed by hand from the stored criteria equals the stored
   value.** The number is computed in code, never asked from the model
   (trade #3) — and this is the check that the claim is true rather than
   merely intended. The recomputation is done here from the numbers the WIRE
   returns, so it also catches an API that reports different criteria from the
   ones it scored.

Plus the provenance rules the module plan locks: every evaluation carries its
judge model and rubric version (AC6), an empty `clashes` array is accepted
as a verdict rather than retried into producing something (AC5, §10), and
`DateDigest` compiles from stored evaluations with no new AI call (S12-B11).

**Revised 2026-09-02.** The rubric is `judge_rubric.v2` and the ten-turn
exclusion is gone: every date with a transcript is judged, and the judge
reports its own `confidence` instead. The checks below accept a v1 evaluation
where one already exists — those rows are real history, produced under real
instructions, and a probe that demanded v2 everywhere would be asserting that
the past was different.

**Why part of this runs in-process**, unlike most probes here: judging is
IDEMPOTENT by design — `date_evaluations.date_id` is a primary key, and the
pipeline returns the stored score rather than re-judging. That is a feature,
and it means the product has no HTTP action that judges one transcript twice.
Same situation as `probe_structured_guard.py`, which reaches for the Guard
directly because a give-up has no HTTP surface either. Everything that CAN go
over HTTP does: the transcript, the stored evaluation, and the arithmetic
check all come from the API exactly as the app sees them.

Run inside the api container:

    docker compose exec api python probes/probe_judge.py

Cost: ONE judging call (the re-run). It reuses a transcript that already
exists rather than simulating a fresh date — see PICKUP's quota table for why
that matters.
"""

from __future__ import annotations

import asyncio
import sys

import httpx
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.ai.registry import build_providers
from app.ai.routing import TaskRouter
from app.config import get_settings, load_ai_config
from app.db import create_engine
from app.models import Analysis, DateEvaluation, SimulatedDate

API = "http://localhost:8000"

# The two scores must agree within this many points of 100. Stated as a number
# rather than left implicit because it IS the claim: a judge that can swing 30
# points on the same transcript cannot be used to rank two candidates who are
# 10 points apart, which is exactly what this product does with it.
TOLERANCE = 12.0

checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))


def recompute(criteria: dict) -> float:
    """The formula written out by hand, deliberately NOT importing
    `judging.date_score`. Importing it would make this check "the function
    agrees with itself"; typing the arithmetic out is what makes it agree with
    `date_simulation.md`."""
    return (
        0.30 * criteria["trait_alignment"]
        + 0.30 * criteria["conversational_flow"]
        + 0.25 * criteria["mutual_engagement"]
        + 0.15 * (100 - criteria["clash_severity"])
    )


async def find_judged_date(client: httpx.AsyncClient, token: str) -> dict | None:
    """The newest judged date belonging to this account, over HTTP."""
    analyses = (
        await client.get("/analyses", headers={"Authorization": f"Bearer {token}"})
    ).json()["analyses"]
    for analysis in analyses:
        dates = (
            await client.get(
                f"/analyses/{analysis['id']}/dates",
                headers={"Authorization": f"Bearer {token}"},
            )
        ).json()["dates"]
        for d in dates:
            if d.get("evaluation"):
                return d
    return None


async def main() -> int:
    print("probe_judge — S12-P1\n")

    email = sys.argv[1] if len(sys.argv) > 1 else None
    password = sys.argv[2] if len(sys.argv) > 2 else "probe-password"
    if not email:
        print("  usage: python probes/probe_judge.py <email> [password]")
        print("  needs an account that already has a judged analysis.")
        return 2

    async with httpx.AsyncClient(base_url=API, timeout=120) as client:
        login = await client.post(
            "/auth/login", json={"email": email, "password": password}
        )
        if login.status_code != 200:
            print(f"  could not sign in as {email}: {login.status_code} {login.text}")
            return 2
        token = login.json()["token"]

        judged = await find_judged_date(client, token)
        check("the account has a judged date to check", judged is not None)
        if judged is None:
            return report()

        evaluation = judged["evaluation"]
        print(f"  .. date {judged['date_id'][:8]} at '{judged['setting_name']}', "
              f"{judged['turn_count']} turns in {judged['message_count']} rows")

        # --- AC1 half two: the arithmetic, from the wire -------------------
        by_hand = recompute(evaluation["criteria"])
        stored = evaluation["date_score"]
        check(
            "date_score recomputed by hand from the stored criteria matches "
            "the stored value",
            abs(by_hand - stored) < 0.01,
            f"by hand {by_hand:.2f} vs stored {stored:.2f}",
        )

        # --- AC6: provenance on every evaluation ---------------------------
        check(
            "the evaluation carries its judge model and rubric version (AC6)",
            bool(evaluation["judge_model"])
            and evaluation["rubric_version"] in ("judge_rubric.v1", "judge_rubric.v2"),
            f"{evaluation['judge_provider']}/{evaluation['judge_model']} "
            f"under {evaluation['rubric_version']}",
        )

        # --- 2026-09-02: depth is reported, not used as a gate -------------
        check(
            "a v2 evaluation carries the judge's own confidence and a note "
            "about what the transcript could show; a v1 one carries neither "
            "and must not have had either invented for it",
            (
                evaluation["confidence"] is not None
                and bool(evaluation["evidence_note"])
                if evaluation["rubric_version"] == "judge_rubric.v2"
                else evaluation["confidence"] is None
            ),
            f"{evaluation['rubric_version']}: confidence="
            f"{evaluation['confidence']}, note="
            f"{(evaluation['evidence_note'] or '')[:80]!r}",
        )

        # --- AC5: an empty clashes array is a verdict, not a retry ---------
        empties = 0
        analyses = (
            await client.get("/analyses", headers={"Authorization": f"Bearer {token}"})
        ).json()["analyses"]
        for analysis in analyses:
            dates = (
                await client.get(
                    f"/analyses/{analysis['id']}/dates",
                    headers={"Authorization": f"Bearer {token}"},
                )
            ).json()["dates"]
            for d in dates:
                e = d.get("evaluation")
                if e is not None and not e["clashes"]:
                    empties += 1
        check(
            "an empty `clashes` array is stored as a valid verdict, not "
            "retried into producing one (AC5, §10)",
            empties > 0,
            f"{empties} judged dates reported no clashes",
        )

        # --- AC4, rewritten 2026-09-02: nothing is excluded for being thin --
        #
        # This block used to assert the opposite: that a sub-10-turn date was
        # excluded and said so on the wire. The owner removed the threshold, so
        # the check now looks for the failure that removal could cause — a
        # thin date quietly still being dropped.
        finished = [
            d
            for analysis in analyses
            for d in (
                await client.get(
                    f"/analyses/{analysis['id']}/dates",
                    headers={"Authorization": f"Bearer {token}"},
                )
            ).json()["dates"]
            if d["status"] in ("complete", "incomplete")
        ]
        thin = [d for d in finished if 0 < d["turn_count"] < 10]
        for d in thin:
            check(
                f"a {d['turn_count']}-turn date is JUDGED rather than thrown "
                "away, and carries a confidence saying how thin it was",
                d["evaluation"] is not None and not d["excluded_from_score"],
                f"status={d['status']}, score="
                f"{(d['evaluation'] or {}).get('date_score')}, confidence="
                f"{(d['evaluation'] or {}).get('confidence')}",
            )
        for d in [d for d in finished if d["excluded_from_score"]]:
            check(
                "the only date excluded from a score is one nobody spoke on",
                d["turn_count"] == 0 and d["evaluation"] is None,
                f"status={d['status']}, turns={d['turn_count']}, "
                f"{d['message_count']} rows",
            )

        # --- 2026-09-02: the shared fixture ---------------------------------
        for analysis in analyses:
            payload = (
                await client.get(
                    f"/analyses/{analysis['id']}/dates",
                    headers={"Authorization": f"Bearer {token}"},
                )
            ).json()
            fixture = payload.get("fixture")
            if fixture is None:
                continue  # ran before the shared fixture existed
            settings_seen = {d["setting_name"] for d in payload["dates"]}
            check(
                "every candidate in an analysis was run against the SAME "
                "evening — which is the only thing that makes their scores "
                "comparable",
                settings_seen <= {fixture["setting_name"]},
                f"fixture {fixture['archetype']!r}: {sorted(settings_seen)}",
            )

    # --- AC1 half one: the same transcript judged twice --------------------
    #
    # In-process from here: judging is idempotent by design (see the module
    # docstring), so there is no HTTP action that re-judges one transcript.
    settings = get_settings()
    ai_config = load_ai_config(settings.ai_config_path)
    router = TaskRouter(build_providers(ai_config), ai_config)
    engine = create_engine(settings.database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    from app.judging import date_digest, judge_date

    async with factory() as session:
        date_id = judged["date_id"]
        row = (
            await session.execute(
                select(SimulatedDate).where(SimulatedDate.id == date_id)
            )
        ).scalar_one()
        analysis = (
            await session.execute(
                select(Analysis).where(Analysis.id == row.analysis_id)
            )
        ).scalar_one()

        print("\n  .. re-judging the same transcript (one AI call)…")
        await session.execute(
            delete(DateEvaluation).where(DateEvaluation.date_id == row.id)
        )
        await session.commit()
        again = await judge_date(session, router, row, analysis.user_id)
        check("the re-run produced an evaluation", again is not None)
        if again is not None:
            delta = abs(again.score - stored)
            check(
                f"the same transcript scores within {TOLERANCE:g} points on a "
                "re-run (AC1)",
                delta <= TOLERANCE,
                f"{stored:.2f} then {again.score:.2f} — delta {delta:.2f}",
            )

        # --- S12-B11: the digest is compiled, not generated -----------------
        digest = await date_digest(session, row.analysis_id, row.candidate_user_id)
        check(
            "DateDigest returns a real summary compiled from stored evaluations",
            bool(digest) and digest != "You haven't been on a simulated date with "
            "them yet.",
            digest[:110] + ("…" if len(digest) > 110 else ""),
        )
        check(
            "…with NO new AI call — it cannot make one, having no router "
            "(S12-B11)",
            "router" not in date_digest.__code__.co_varnames,
        )

    await engine.dispose()
    return report()


def report() -> int:
    failed = [label for label, ok in checks if not ok]
    print(
        f"\n{'GREEN' if not failed else 'RED'} — "
        f"{len(checks) - len(failed)}/{len(checks)} checks passed"
    )
    for label in failed:
        print(f"  failed: {label}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
