"""Demo profiles through the REAL pipeline (S15-B4, B5; `data_hygiene.md` §2).

Step 2 of the reconciliation pass. On every boot each profile in
`seeds/demo_profiles.yaml` is compared to the database by its email:

- no account → created via `create_user` (the registration path), `is_demo`;
- an answer missing or different → `save_answer` (the upsert path), logged as
  `created` / `edited` exactly as a person's would be;
- no live traits → `extract_once`; no ready snapshot, or a stale one →
  `compile_persona`; no fresh embeddings → `refresh_embeddings`.

**No shortcut inserts** (§12). A demo profile that gets its traits by any
other road is a row nobody can explain: no `source_answer_ids`, no
`extracted_by`, no log line. The probe checks the provenance is there.

**The AI part runs in the background after boot.** Creating accounts and
saving answers is milliseconds and happens inline; extraction, compilation
and embedding are three model calls per profile and would hold the health
endpoint hostage for minutes. The boot logs what it will do; the background
task logs what it did; a second boot on a healthy database logs a no-op.

**Cost, stated:** a demo user whose pipeline keeps failing (a bad model day)
is retried once per boot — never in a loop within one boot, and the log says
so each time. That is the honest floor: silently giving up would leave a
demo profile that looks real and cannot be matched.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.accounts import create_user
from app.ai.routing import TaskRouter
from app.answers import save_answer
from app.logging_setup import log_event
from app.models import Answer, ProfileEmbedding, Question, Trait, User
from app.users import GENDER_VALUES, compute_age

logger = logging.getLogger("app.demo")

DEMO_PATH = Path(__file__).resolve().parent.parent / "seeds" / "demo_profiles.yaml"
BASELINE_CODES = ("BQ1", "BQ2", "BQ3", "BQ4", "BQ5")
# The floor (§18, owner decision 2026-09-01). The DB CHECK is the last line;
# this is the first, at the same boundary the endpoint enforces it.
MIN_ANSWER_CHARS = 50


@dataclass(frozen=True)
class DemoProfile:
    email: str
    display_name: str
    birth_date: date
    gender: str
    interested_in: list[str]
    age_pref_min: int
    age_pref_max: int
    city: str | None
    country: str | None
    opt_in: bool
    answers: dict[str, str]


def load_demo_profiles(path: Path = DEMO_PATH) -> tuple[str, list[DemoProfile]]:
    """Parse and validate the fixture with the A1 rules. A bad fixture fails
    here, loudly, at boot — not as a database CHECK error mid-seed."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    password = str(raw["password"])
    if len(password) < 8:
        raise ValueError("demo password must be at least 8 characters (A1)")
    profiles: list[DemoProfile] = []
    for p in raw["profiles"]:
        birth = date.fromisoformat(str(p["birth_date"]))
        if compute_age(birth) < 18:
            raise ValueError(f"{p['email']}: demo profile must be 18+ (A1)")
        if p["gender"] not in GENDER_VALUES:
            raise ValueError(f"{p['email']}: unknown gender {p['gender']!r}")
        interested = list(p["interested_in"])
        if not interested or any(g not in GENDER_VALUES for g in interested):
            raise ValueError(f"{p['email']}: interested_in must be one or more of {GENDER_VALUES}")
        if not (18 <= int(p["age_pref_min"]) <= int(p["age_pref_max"])):
            raise ValueError(f"{p['email']}: age range must satisfy 18 <= min <= max (A1)")
        answers = {k: str(v).strip() for k, v in dict(p["answers"]).items()}
        missing = [c for c in BASELINE_CODES if c not in answers]
        if missing:
            raise ValueError(f"{p['email']}: missing baseline answers {missing}")
        thin = [c for c, a in answers.items() if len(a) < MIN_ANSWER_CHARS]
        if thin:
            raise ValueError(f"{p['email']}: answers under {MIN_ANSWER_CHARS} chars: {thin}")
        profiles.append(DemoProfile(
            email=str(p["email"]), display_name=str(p["display_name"]),
            birth_date=birth, gender=str(p["gender"]), interested_in=interested,
            age_pref_min=int(p["age_pref_min"]), age_pref_max=int(p["age_pref_max"]),
            city=p.get("city"), country=p.get("country"),
            opt_in=bool(p.get("opt_in", True)), answers=answers,
        ))
    return password, profiles


# --- The inline half: accounts and answers -----------------------------------


async def ensure_accounts_and_answers(engine: AsyncEngine) -> dict[str, int]:
    password, profiles = load_demo_profiles()
    factory = async_sessionmaker(engine, expire_on_commit=False)
    counts = {"created": 0, "answers_created": 0, "answers_edited": 0, "ok": 0}

    async with factory() as session:
        questions = {
            q.code: q
            for q in (
                await session.scalars(select(Question).where(Question.code.in_(BASELINE_CODES)))
            ).all()
        }
        if len(questions) != len(BASELINE_CODES):
            # Step 1 of the pass seeds these; if they are not here, step 2 has
            # nothing to attach answers to and must say so rather than write
            # answers to nothing.
            raise RuntimeError("baseline questions are not seeded; step 1 must run first")

        for p in profiles:
            user = await session.scalar(select(User).where(User.email == p.email))
            if user is None:
                user = await create_user(
                    session, email=p.email, password=password,
                    display_name=p.display_name, birth_date=p.birth_date,
                    gender=p.gender, interested_in=p.interested_in,
                    age_pref_min=p.age_pref_min, age_pref_max=p.age_pref_max,
                    city=p.city, country=p.country, opt_in=p.opt_in, is_demo=True,
                )
                counts["created"] += 1
                log_event(logger, "demo_user_created", user_id=str(user.id), email=p.email)
            elif not user.is_demo:
                # A real person registered with a demo email. Never relabel a
                # real account; say so and leave it alone.
                log_event(
                    logger, "demo_email_taken_by_real_user", level=logging.ERROR,
                    email=p.email, user_id=str(user.id),
                )
                continue

            touched = False
            for code in BASELINE_CODES:
                _, outcome = await save_answer(session, user.id, questions[code], p.answers[code])
                if outcome == "created":
                    counts["answers_created"] += 1
                    touched = True
                elif outcome == "edited":
                    counts["answers_edited"] += 1
                    touched = True
            if not touched:
                counts["ok"] += 1

    log_event(
        logger,
        "reconcile_demo_accounts" if any(
            counts[k] for k in ("created", "answers_created", "answers_edited")
        ) else "reconcile_demo_accounts_noop",
        **counts,
    )
    return counts


# --- The background half: the real pipeline ---------------------------------


async def _pipeline_needs(session, user_id) -> dict[str, bool]:
    from app.persona import get_current_snapshot, is_stale

    live_traits = (
        await session.scalar(
            select(Trait.id).where(Trait.user_id == user_id, Trait.status != "retracted").limit(1)
        )
    ) is not None
    answered = (
        await session.scalar(select(Answer.id).where(Answer.user_id == user_id).limit(1))
    ) is not None
    ready = await get_current_snapshot(session, user_id)
    stale = await is_stale(session, user_id, ready) if ready else False
    embeddings = list(
        (await session.scalars(
            select(ProfileEmbedding).where(ProfileEmbedding.user_id == user_id)
        )).all()
    )
    return {
        "extract": answered and not live_traits,
        "compile": ready is None or stale,
        "embed": len(embeddings) < 2,
    }


async def run_demo_pipeline(engine: AsyncEngine, router: TaskRouter) -> dict[str, int]:
    """Extraction → compilation → embedding for every demo user missing any of
    them. Real code paths, real calls, one attempt each per boot."""
    from app.extraction import extract_once
    from app.matching import refresh_embeddings
    from app.persona import compile_persona

    _, profiles = load_demo_profiles()
    factory = async_sessionmaker(engine, expire_on_commit=False)
    counts = {"extracted": 0, "compiled": 0, "embedded": 0, "failed": 0, "deferred": 0, "ok": 0}

    for p in profiles:
        async with factory() as session:
            user = await session.scalar(
                select(User).where(User.email == p.email, User.is_demo.is_(True))
            )
            if user is None:
                continue
            needs = await _pipeline_needs(session, user.id)
            if not any(needs.values()):
                counts["ok"] += 1
                continue
            log_event(logger, "demo_pipeline_start", user_id=str(user.id), email=p.email, **needs)
            try:
                if needs["extract"]:
                    outcome = await extract_once(session, router, user.id)
                    if outcome is None:
                        # Another run holds this user's extraction lock (a
                        # concurrent pass, or a person editing an answer at
                        # the same moment). That run will produce the traits;
                        # compiling now would build a persona from nothing.
                        # Deferred, said so, and left for the next boot.
                        counts["deferred"] += 1
                        log_event(
                            logger, "demo_extraction_deferred", user_id=str(user.id),
                            email=p.email, reason="an extraction for this user is already in flight",
                        )
                        continue
                    counts["extracted"] += 1
                    log_event(
                        logger, "demo_extracted", user_id=str(user.id), added=outcome.added,
                    )
                    # A fresh extraction changes the hash: the snapshot is stale
                    # by definition, and the embeddings with it.
                    needs["compile"] = True
                    needs["embed"] = True
                if needs["compile"]:
                    snapshot = await compile_persona(session, router, user.id)
                    counts["compiled"] += 1
                    log_event(
                        logger, "demo_compiled", user_id=str(user.id),
                        version=snapshot.version, status=snapshot.status,
                    )
                if needs["embed"]:
                    fresh = await refresh_embeddings(session, router, user.id)
                    counts["embedded"] += 1
                    log_event(logger, "demo_embedded", user_id=str(user.id), fresh=fresh)
            except Exception as exc:  # noqa: BLE001
                # One attempt per boot, and the failure is named. Not raised:
                # the third profile must not lose its turn to the second's
                # bad model day.
                counts["failed"] += 1
                log_event(
                    logger, "demo_pipeline_failed", level=logging.ERROR,
                    user_id=str(user.id), email=p.email,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                    note="retried on the next boot",
                )

    log_event(
        logger,
        "reconcile_demo_pipeline" if any(
            counts[k] for k in ("extracted", "compiled", "embedded", "failed", "deferred")
        ) else "reconcile_demo_pipeline_noop",
        **counts,
    )
    return counts


_tasks: set[asyncio.Task] = set()


def start_demo_pipeline(engine: AsyncEngine, router: TaskRouter) -> asyncio.Task:
    """Launch the background half. Held in a module set for the reason every
    background task in this codebase is held: asyncio keeps only a weak
    reference, and a task nobody keeps can vanish mid-flight."""

    async def _run() -> None:
        try:
            await run_demo_pipeline(engine, router)
        except Exception as exc:  # noqa: BLE001
            log_event(
                logger, "demo_pipeline_crashed", level=logging.ERROR,
                error=f"{type(exc).__name__}: {exc}",
            )

    task = asyncio.create_task(_run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task
