"""Startup reconciliation — step 1 of the four-step pass (S3-B5,
data_hygiene.md §2): baseline + pool questions.

On EVERY boot, desired question rows (seeds/questions.yaml, generated verbatim
from module_1_data_collection.md) are compared against actual rows. Missing
rows are inserted; drifted fields are repaired; correct rows are left alone —
and each category is logged (§7, §12: there is no "it shipped with us" path).

Steps 2-4 of the pass (demo profiles, stuck-analysis relaunch, embedding-model
consistency) land in Steps 15 and 11 of the roadmap; this module is where they
will live, in order.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.logging_setup import log_event
from app.models import Question

logger = logging.getLogger("app.reconcile")

SEEDS_PATH = Path(__file__).resolve().parent.parent / "seeds" / "questions.yaml"

# The fields reconciliation owns for a seeded question. `text` drift is the
# expected case; probe_area/pool_order drift would mean the plan was revised.
_MANAGED_FIELDS = ("origin", "probe_area", "pool_order", "text")


def load_desired_questions(path: Path = SEEDS_PATH) -> list[dict]:
    seeds = yaml.safe_load(path.read_text(encoding="utf-8"))
    desired = [
        {"code": q["code"], "origin": "baseline", "probe_area": q["probe_area"],
         "pool_order": None, "text": q["text"]}
        for q in seeds["baseline"]
    ] + [
        {"code": q["code"], "origin": "pool", "probe_area": q["probe_area"],
         "pool_order": q["pool_order"], "text": q["text"]}
        for q in seeds["pool"]
    ]
    return desired


async def reconcile(engine: AsyncEngine) -> dict[str, int]:
    """Runs the pass; returns and logs {seeded, repaired, ok} counts."""
    desired = load_desired_questions()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    seeded: list[str] = []
    repaired: list[str] = []
    ok = 0

    async with session_factory() as session:
        existing = {
            q.code: q
            for q in (
                await session.scalars(select(Question).where(Question.code.is_not(None)))
            ).all()
        }
        for want in desired:
            have = existing.get(want["code"])
            if have is None:
                session.add(
                    Question(
                        user_id=None, origin=want["origin"], code=want["code"],
                        pool_order=want["pool_order"], probe_area=want["probe_area"],
                        text=want["text"],
                    )
                )
                seeded.append(want["code"])
                continue
            drifted = {
                f: (getattr(have, f), want[f])
                for f in _MANAGED_FIELDS
                if getattr(have, f) != want[f]
            }
            if drifted:
                for f, (_, new) in drifted.items():
                    setattr(have, f, new)
                repaired.append(want["code"])
                log_event(
                    logger, "reconcile_question_repaired", level=logging.WARNING,
                    code=want["code"],
                    fields={f: {"was": old, "now": new} for f, (old, new) in drifted.items()},
                )
            else:
                ok += 1
        await session.commit()

    counts = {"seeded": len(seeded), "repaired": len(repaired), "ok": ok}
    log_event(
        logger,
        "reconcile_questions" if (seeded or repaired) else "reconcile_questions_noop",
        **counts,
        seeded_codes=seeded or None,
        repaired_codes=repaired or None,
    )
    return counts
