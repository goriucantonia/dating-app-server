"""PersonaCompiler and PersonaService (S7-B3…B6, B9, B10, B11).

A snapshot is assembled in two parts, and the split is the load-bearing
decision of this module (trait_persona.md trade #1):

  DETERMINISTIC, no AI call — hard facts, trait rows grouped by category, and
  3–5 of the user's own sentences quoted VERBATIM. A model paraphrasing the
  user's writing before mimicry would launder out the voice, which is the
  thing the whole product rests on. So nothing stands between what they wrote
  and what the agent imitates.

  ONE structured AI call — the behaviour digest. "How do you act when a
  conversation gets tense" is a synthesis, not a lookup, so it is the only
  part a model touches. Its provider/model is recorded in `digest_model`.

Snapshots are IMMUTABLE and per-user versioned. Recompiling writes v(n+1);
old transcripts keep pointing at their own version, which is what lets a date
stay explainable forever. Nothing here UPDATEs a row that reached 'ready'.

A failed compilation is not a silent no-op: the row is written first as
'compiling', then moved to 'failed' with its error, and the PREVIOUS snapshot
stays current. A user is never simulated from a half-built persona — and
`get_current_snapshot` returning None is the hard gate (§11) that keeps them
out of candidate pools entirely rather than running degraded.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.base import AIError, GenRequest, Message
from app.ai.routing import TaskRouter
from app.ai.structured import guarded_structured_call
from app.logging_setup import log_event
from app.models import (
    Answer,
    CalibrationMessage,
    CalibrationSession,
    PersonaSnapshot,
    Question,
    Trait,
    User,
)
from app.schemas.agent_response import SCHEMA_VERSION
from app.schemas.persona_digest import PERSONA_DIGEST_V1
from app.traits_hash import compute_traits_hash
from app.users import compute_age

logger = logging.getLogger("app.persona")

TASK = "persona_digest"
MAX_TOKENS = 4096

# 3–5 by the spec. Fewer than three and one long answer dominates the voice;
# more than five and the prompt is mostly quotation.
MIN_EXCERPTS = 3
MAX_EXCERPTS = 5

CATEGORY_HEADINGS = {
    "interest": "What they're into",
    "quality": "What they're like at their best",
    "flaw": "Their rough edges (real ones — do not smooth these away)",
    "behavioral": "How they tend to act",
    "conversational_style": "How they talk",
    "partner_preference": "What they want in someone else",
}

DIGEST_SYSTEM_PROMPT = """\
You read what a person wrote about themselves and describe how they BEHAVE in \
four specific situations. You are writing notes that will be handed to someone \
about to play this person in conversation, so write in the second person — \
"you go quiet", not "he goes quiet".

Stay inside the evidence. If they never said how they act when someone is \
upset, say what their other answers imply and keep it modest — do not invent a \
vivid scene. Include the unflattering parts: someone who withdraws for days \
after an argument needs that written down, because a persona that is warm in \
every situation is not this person and will not fool them for a sentence."""


@dataclass
class Excerpt:
    code: str
    probe_area: str
    text: str


def _select_excerpts(rows: list[tuple[Question, Answer]]) -> list[Excerpt]:
    """3–5 of the user's own sentences, chosen by LENGTH within PROBE-AREA
    SPREAD (S7-B3).

    Spread first, then length: taking the five longest answers outright would
    happily return five answers about the same probe area, and the voice sample
    would then only demonstrate how they write about, say, their hobbies. One
    pass takes the longest answer from each distinct area; a second pass fills
    any remaining slots with the next-longest whatever they are.

    Deterministic — same answers in, same excerpts out — because a snapshot
    that quietly re-rolls its quotations on every compile is not reproducible.
    """
    ranked = sorted(
        rows,
        key=lambda qa: (-len(qa[1].answer_text), qa[0].code or "", str(qa[1].id)),
    )
    picked: list[Excerpt] = []
    seen_areas: set[str] = set()

    for q, a in ranked:
        if q.probe_area not in seen_areas and len(picked) < MAX_EXCERPTS:
            seen_areas.add(q.probe_area)
            picked.append(Excerpt(q.code or "dispute", q.probe_area, a.answer_text))

    if len(picked) < MAX_EXCERPTS:
        chosen = {e.text for e in picked}
        for q, a in ranked:
            if len(picked) >= MAX_EXCERPTS:
                break
            if a.answer_text not in chosen:
                chosen.add(a.answer_text)
                picked.append(Excerpt(q.code or "dispute", q.probe_area, a.answer_text))

    return picked


def _build_facts(user: User) -> str:
    bits = [
        f"Name: {user.display_name}",
        f"Age: {compute_age(user.birth_date)}",
        f"Gender: {user.gender}",
        f"Interested in: {', '.join(user.interested_in)}",
    ]
    where = ", ".join(p for p in (user.city, user.country) if p)
    if where:
        bits.append(f"Lives in: {where}")
    return "\n".join(bits)


def _build_traits_block(traits: list[Trait]) -> str:
    by_category: dict[str, list[Trait]] = {}
    for t in traits:
        by_category.setdefault(t.category, []).append(t)

    out: list[str] = []
    for category, heading in CATEGORY_HEADINGS.items():
        rows = by_category.get(category)
        if not rows:
            continue
        out.append(f"{heading}:")
        for t in rows:
            # `confirmed` is surfaced because the person said "yes, that's me"
            # and the agent should lean on it harder than on a guess (§9).
            mark = " [they confirmed this]" if t.status == "confirmed" else ""
            out.append(f"  - {t.label}: {t.description}{mark}")
        out.append("")
    return "\n".join(out).strip()


async def _load_flags(session: AsyncSession, user_id: uuid.UUID) -> list[str]:
    """S7-B9. Everything the user has ever flagged as "I'd never say that",
    across every past calibration session, becomes an explicit negative example
    in the NEXT compilation. A correction, where they gave one, is worth more
    than the rejection alone."""
    rows = (
        await session.execute(
            select(CalibrationMessage)
            .join(
                CalibrationSession,
                CalibrationSession.id == CalibrationMessage.session_id,
            )
            .where(
                CalibrationSession.user_id == user_id,
                CalibrationMessage.flagged.is_(True),
                CalibrationMessage.sender == "persona",
            )
            .order_by(CalibrationMessage.created_at)
        )
    ).scalars()

    out: list[str] = []
    for m in rows:
        if m.correction:
            out.append(f'  - Never: "{m.text_}"  →  they would say: "{m.correction}"')
        else:
            out.append(f'  - Never phrase things like: "{m.text_}"')
    return out


def _assemble_prompt(
    user: User,
    traits_block: str,
    excerpts: list[Excerpt],
    digest: dict,
    flags: list[str],
) -> str:
    quoted = "\n\n".join(
        f'--- in their own words ({e.probe_area}):\n"{e.text}"' for e in excerpts
    )
    prompt = f"""You are {user.display_name}. You are not an assistant playing \
{user.display_name}; for this conversation you are simply them, talking to \
someone new.

WHO YOU ARE
{_build_facts(user)}

{traits_block}

HOW YOU BEHAVE

When things get tense: {digest['in_tense_moments']}

When you're flirting or interested: {digest['when_flirting']}

When someone needs support: {digest['when_supporting']}

When you open up: {digest['when_opening_up']}

HOW YOU WRITE AND SPEAK

Below are things you have actually written, unedited. This is your voice — \
your sentence length, your habits, your vocabulary, the way you land a point \
or trail off. Match it. Do not write more politely, more smoothly, or more \
articulately than these.

{quoted}"""

    if flags:
        # S7-B9: the user's own corrections, stated as prohibitions.
        prompt += (
            "\n\nTHINGS YOU WOULD NEVER SAY\n"
            "You have read replies written as you and rejected these. Do not "
            "produce anything like them:\n" + "\n".join(flags)
        )

    prompt += """

Stay in character. You do not know you are an AI, you have no assistant \
reflexes, and you never mention a system prompt. If you would be short with \
someone, be short. If you would change the subject, change it."""
    return prompt


async def _next_version(session: AsyncSession, user_id: uuid.UUID) -> int:
    current = (
        await session.execute(
            select(PersonaSnapshot.version)
            .where(PersonaSnapshot.user_id == user_id)
            .order_by(PersonaSnapshot.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return (current or 0) + 1


async def compile_persona(
    session: AsyncSession, router: TaskRouter, user_id: uuid.UUID
) -> PersonaSnapshot:
    """S7-B3/B4. Builds v(n+1) and returns it, 'ready' or 'failed'.

    The row is written as 'compiling' and COMMITTED before the AI call, so a
    process that dies mid-call leaves a visible failed-looking row rather than
    no evidence at all. On failure the row is moved to 'failed' with its error
    and the previous snapshot remains current — observed by the probe, not
    reasoned about (AC5).
    """
    user = (
        await session.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    if user is None:
        raise AIError("no such user", task=TASK)

    traits = list(
        (
            await session.execute(
                select(Trait)
                .where(Trait.user_id == user_id, Trait.status != "retracted")
                .order_by(Trait.category, Trait.created_at)
            )
        ).scalars()
    )
    rows = [
        (q, a)
        for q, a in (
            await session.execute(
                select(Question, Answer)
                .join(Answer, Answer.question_id == Question.id)
                .where(Answer.user_id == user_id)
            )
        ).all()
    ]

    if not traits or not rows:
        raise AIError(
            "there is nothing to build a persona from yet — this person has no "
            "traits or no answers",
            task=TASK,
        )

    snapshot = PersonaSnapshot(
        user_id=user_id,
        version=await _next_version(session, user_id),
        status="compiling",
        system_prompt=None,
        schema_version=SCHEMA_VERSION,
        traits_hash=compute_traits_hash(traits),
        source_trait_ids=[t.id for t in traits],
    )
    session.add(snapshot)
    await session.commit()

    log_event(
        logger, "persona_compile_start",
        user_id=str(user_id), snapshot_id=str(snapshot.id), version=snapshot.version,
        traits=len(traits), source_answer_ids=[str(a.id) for _, a in rows],
    )

    try:
        excerpts = _select_excerpts(rows)
        if len(excerpts) < MIN_EXCERPTS:
            raise AIError(
                f"only {len(excerpts)} answers to quote from; a voice sample "
                f"needs at least {MIN_EXCERPTS}",
                task=TASK,
            )

        provider, model = router.resolve(TASK)
        digest_model = f"{provider.name}/{model}"

        situational = "\n\n".join(
            f"--- {q.code or 'dispute'} ({q.probe_area})\nQ: {q.text}\nA: {a.answer_text}"
            for q, a in rows
        )
        digest = await guarded_structured_call(
            provider,
            GenRequest(
                task=TASK, model=model, system_prompt=DIGEST_SYSTEM_PROMPT,
                messages=[
                    Message(
                        role="user",
                        content=(
                            f"This is everything {user.display_name} has written "
                            f"about themselves.\n\n{situational}\n\n"
                            "Describe how they behave in the four situations."
                        ),
                    )
                ],
                temperature=0.4, max_tokens=MAX_TOKENS,
            ),
            PERSONA_DIGEST_V1,
        )

        flags = await _load_flags(session, user_id)
        snapshot.system_prompt = _assemble_prompt(
            user, _build_traits_block(traits), excerpts, digest, flags
        )
        snapshot.digest_model = digest_model
        snapshot.status = "ready"
        await session.commit()

        log_event(
            logger, "persona_compile_done",
            user_id=str(user_id), snapshot_id=str(snapshot.id),
            version=snapshot.version, outcome="ready",
            traits=len(traits), excerpts=len(excerpts),
            excerpt_areas=[e.probe_area for e in excerpts],
            negative_examples=len(flags), digest_model=digest_model,
            prompt_chars=len(snapshot.system_prompt),
        )
        return snapshot

    except Exception as exc:  # noqa: BLE001 — see below
        # Deliberately blind, and D-005 is why: the failure branch you TYPE
        # is the only one you handle. A compilation can fail on the model, on
        # the schema, on a missing route, or on something not yet imagined —
        # and every one of those must land the row in 'failed' with its error
        # rather than leaving it stuck at 'compiling' forever.
        # The previous snapshot stays current — that is the whole point of
        # writing a NEW row rather than mutating the old one.
        snapshot.status = "failed"
        snapshot.error = f"{type(exc).__name__}: {exc}"[:2000]
        await session.commit()
        log_event(
            logger, "persona_compile_done",
            level=logging.ERROR, user_id=str(user_id),
            snapshot_id=str(snapshot.id), version=snapshot.version,
            outcome="failed", error=snapshot.error,
        )
        return snapshot


async def get_current_snapshot(
    session: AsyncSession, user_id: uuid.UUID
) -> PersonaSnapshot | None:
    """S7-B5. The newest 'ready' snapshot, or None.

    None is a HARD GATE, not a hint (§11): a user without one cannot be
    simulated, so Candidate Matching must exclude them from pools entirely
    rather than running a date on a half-built persona. Date Simulation and
    Chat call ONLY this — they never read `traits` or `answers` directly.
    """
    return (
        await session.execute(
            select(PersonaSnapshot)
            .where(PersonaSnapshot.user_id == user_id, PersonaSnapshot.status == "ready")
            .order_by(PersonaSnapshot.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def latest_snapshot(
    session: AsyncSession, user_id: uuid.UUID
) -> PersonaSnapshot | None:
    """The newest snapshot of ANY status — what `GET /persona/current` reports,
    so a user watching a compile sees 'compiling' rather than the stale ready
    one or nothing at all."""
    return (
        await session.execute(
            select(PersonaSnapshot)
            .where(PersonaSnapshot.user_id == user_id)
            .order_by(PersonaSnapshot.version.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def is_stale(
    session: AsyncSession, user_id: uuid.UUID, snapshot: PersonaSnapshot | None
) -> bool:
    """S7-B10. Compare the snapshot's `traits_hash` to the live one.

    This is the same hash `traits_hash.py` computes in one place and the same
    one an all-`keep` extraction leaves byte-identical — so answering a batch
    of questions that changes nothing does NOT tell the user their profile
    needs rebuilding.
    """
    if snapshot is None:
        return False
    live = list(
        (
            await session.execute(
                select(Trait).where(
                    Trait.user_id == user_id, Trait.status != "retracted"
                )
            )
        ).scalars()
    )
    return compute_traits_hash(live) != snapshot.traits_hash
