"""Startup reconciliation — the four-step pass (S3-B5, S11-B9, S15-B6;
`data_hygiene.md` §2), in its locked order:

1. Baseline + pool questions (seeds/questions.yaml, generated verbatim from
   module_1_data_collection.md): missing rows inserted, drifted fields
   repaired, correct rows left alone — each category logged.
2. Demo profiles (seeds/demo_profiles.yaml) — accounts and answers inline
   through the REAL registration and answer paths; extraction, compilation
   and embedding in a background task through the real pipeline (app/demo.py).
3. Analyses a restart left in `matching` or `simulating` are relaunched.
4. Embedding-model consistency: every `profile_embeddings.embedding_model`
   must equal the pinned config value; a mismatch is logged loudly and the
   row is marked stale so the next matching run re-embeds it before
   comparing — never compared as-is.

On a healthy database every step logs a `*_noop` line (§1: the second run
is the witness).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import yaml
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.logging_setup import log_event
from app.models import Analysis, ProfileEmbedding, Question

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
    """Step 1. Runs the pass; returns and logs {seeded, repaired, ok} counts."""
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


# --- Step 3 of the pass: analyses a restart left mid-flight (S11-B9) --------

# The states where a process was doing work when it stopped. `matched` is NOT
# one of them: matching finished and the user has not pressed the button yet,
# so re-launching it would start dates nobody asked for. Neither is `failed`:
# that one waits for the user's retry (S13-U5), on purpose.
STUCK_STATES = ("matching", "simulating")

# The relaunched tasks, held for the same reason every other background task in
# this codebase is held: asyncio keeps only a weak reference, and a task nobody
# keeps can be collected mid-flight with no error anywhere.
_tasks: set[asyncio.Task] = set()


async def relaunch_stuck_analyses(app) -> dict[str, int]:
    """On every boot, re-launch anything left in `matching` or `simulating`.

    This is the other half of the named trade behind "no Celery" — a restart
    kills in-flight asyncio tasks, and what makes that acceptable is not that
    it rarely happens but that the work is checkpointed in Postgres and picked
    up again here. Every simulated date resumes from its last checkpointed
    message rather than starting over, which is what `probe_simulation_resume.py`
    exists to watch (§1: the second run is the product, not a corner case).

    Imports are local: this function reaches into the matching and simulation
    modules, and importing them at module scope would make the reconciliation
    pass — which the questions half needs at boot — depend on the whole AI
    stack loading first.
    """
    from app.matching import start_and_run
    from app.simulation import start_pipeline

    factory = async_sessionmaker(app.state.engine, expire_on_commit=False)
    counts = {"matching": 0, "simulating": 0}

    async with factory() as session:
        rows = list(
            (
                await session.scalars(
                    select(Analysis)
                    .where(Analysis.status.in_(STUCK_STATES))
                    .order_by(Analysis.created_at)
                )
            ).all()
        )

    for analysis in rows:
        if analysis.status == "simulating":
            start_pipeline(app, analysis.id)
        else:
            async def _rematch(analysis_id=analysis.id) -> None:
                async with factory() as session:
                    await start_and_run(session, app.state.ai_router, analysis_id)

            task = asyncio.create_task(_rematch())
            _tasks.add(task)
            task.add_done_callback(_tasks.discard)
        counts[analysis.status] += 1
        log_event(
            logger, "analysis_relaunched",
            level=logging.WARNING,
            analysis_id=str(analysis.id), user_id=str(analysis.user_id),
            was=analysis.status,
            reason="left mid-flight by a restart",
        )

    log_event(
        logger,
        "reconcile_analyses" if rows else "reconcile_analyses_noop",
        relaunched_matching=counts["matching"],
        relaunched_simulating=counts["simulating"],
    )
    return counts


# --- Step 4 of the pass: embedding-model consistency (S15-B6) ---------------


async def check_embedding_models(engine: AsyncEngine, expected_model: str) -> dict[str, int]:
    """Every stored vector must come from the pinned model, or cosine between
    two of them is a number that means nothing (ai_interaction.md §3).

    A mismatched row is logged LOUDLY, with the user and the model that wrote
    it, and its `traits_hash` is blanked — which is how "queued for
    re-embedding" is spelled in this codebase: `refresh_embeddings` compares
    the stored hash to the live one before any matching run and re-embeds on
    a mismatch, so the row is regenerated by the pinned model before it is
    ever compared. Never compared as-is.
    """
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        rows = list(
            (
                await session.scalars(
                    select(ProfileEmbedding).where(
                        ProfileEmbedding.embedding_model != expected_model
                    )
                )
            ).all()
        )
        if rows:
            for r in rows:
                log_event(
                    logger, "embedding_model_mismatch", level=logging.ERROR,
                    user_id=str(r.user_id), kind=r.kind,
                    stored_model=r.embedding_model, expected_model=expected_model,
                    action="marked stale; re-embedded by the pinned model before its next comparison",
                )
            await session.execute(
                update(ProfileEmbedding)
                .where(ProfileEmbedding.embedding_model != expected_model)
                .values(traits_hash="")
            )
            await session.commit()

    counts = {"mismatched": len(rows), "queued_for_reembedding": len(rows)}
    log_event(
        logger,
        "reconcile_embeddings" if rows else "reconcile_embeddings_noop",
        expected_model=expected_model, **counts,
    )
    return counts


# --- The whole pass, in order -------------------------------------------------


async def run_full_pass(app, *, inline_demo_pipeline: bool = False) -> dict[str, dict[str, int]]:
    """Steps 1–4 in the locked order.

    Step 2's AI half is launched in the background at boot (it is minutes of
    model calls and must not hold the health endpoint hostage) and reports
    through its own log lines (app/demo.py). A script or a probe passes
    `inline_demo_pipeline=True` to AWAIT it instead — launching it in the
    background and then running it again inline is a race against the
    per-user extraction lock, and the probe that did exactly that got a
    persona built from nothing for its trouble.
    """
    from app.demo import ensure_accounts_and_answers, run_demo_pipeline, start_demo_pipeline

    results: dict[str, dict[str, int]] = {}
    results["questions"] = await reconcile(app.state.engine)
    results["demo_accounts"] = await ensure_accounts_and_answers(app.state.engine)
    if inline_demo_pipeline:
        results["demo_pipeline"] = await run_demo_pipeline(app.state.engine, app.state.ai_router)
    else:
        start_demo_pipeline(app.state.engine, app.state.ai_router)
    results["analyses"] = await relaunch_stuck_analyses(app)
    cfg = app.state.ai_config.embeddings
    results["embeddings"] = await check_embedding_models(
        app.state.engine, f"{cfg.provider}/{cfg.model}"
    )
    log_event(logger, "reconcile_pass_complete", **{k: v for k, v in results.items()})
    return results
