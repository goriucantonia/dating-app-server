"""Trait endpoints (S6-B1, B6, B7, B8).

- POST /profile/extract      — run holistic reconciliation over every answer
- GET  /traits               — the profile the UI's dispute controls read
- POST /traits/{id}/dispute  — "that's not me": mark disputed + generate ONE
                               targeted follow-up question
- POST /traits/{id}/confirm  — "yes, that's me"

§9 runs through all of it: an `inferred` trait is never treated as confirmed.
The status is carried in every payload precisely so no caller downstream has
to guess, and confirm/dispute are the only two things that change it by hand.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.ai.base import AIError, GenRequest, Message, RouteUnresolvedError
from app.ai.structured import guarded_structured_call
from app.errors import ApiError
from app.extraction import extract_once, is_running
from app.logging_setup import log_event
from app.models import Question, Trait, TraitEvent
from app.routers.persona import start_compile
from app.schemas.dispute_followup import DISPUTE_FOLLOWUP_V1
from app.security import CurrentUser, DbSession
from app.traits_hash import compute_traits_hash

router = APIRouter(tags=["traits"])
logger = logging.getLogger("app.traits")

DISPUTE_TASK = "dispute_followups"
DISPUTE_MAX_TOKENS = 2048

# `traits.category` and `questions.probe_area` are two different vocabularies,
# each with its own CHECK constraint, and a dispute question needs a probe_area.
# Routine judgement call (§25), stated rather than buried: a dispute question
# probes the same ground the trait came from. `quality` and `flaw` both map to
# `self_image` because both are the person describing what they are like.
_PROBE_AREA_FOR_CATEGORY = {
    "interest": "interests",
    "partner_preference": "partner_criteria",
    "behavioral": "situational",
    "conversational_style": "conversational",
    "quality": "self_image",
    "flaw": "self_image",
}


class TraitOut(BaseModel):
    id: str
    category: str
    label: str
    description: str
    confidence: float
    status: str
    source_answer_ids: list[str]
    extracted_by: str

    @classmethod
    def build(cls, t: Trait) -> TraitOut:
        return cls(
            id=str(t.id), category=t.category, label=t.label,
            description=t.description, confidence=float(t.confidence),
            status=t.status,
            source_answer_ids=[str(a) for a in (t.source_answer_ids or [])],
            extracted_by=t.extracted_by,
        )


class TraitsOut(BaseModel):
    traits: list[TraitOut]
    traits_hash: str


class ExtractOut(BaseModel):
    status: str  # 'done' | 'queued'
    kept: int = 0
    updated: int = 0
    retracted: int = 0
    added: int = 0
    declined: list[str] = Field(default_factory=list)
    changed: bool = False


class DisputeIn(BaseModel):
    # Optional: the user may say what is actually true. It is not required —
    # demanding a correction to register a dispute would make disputing cost
    # more than living with a wrong trait.
    correction: str | None = None


class DisputeOut(BaseModel):
    trait: TraitOut
    question_id: str
    question_text: str


@router.post("/profile/extract", response_model=ExtractOut)
async def extract(request: Request, user: CurrentUser, session: DbSession) -> ExtractOut:
    """S6-B1. Reads ALL answered questions and ALL existing trait rows, applies
    per-row verdicts. Two rapid requests produce one run and one queued
    follow-up — never two concurrent runs (S6-B5)."""
    try:
        outcome = await extract_once(session, request.app.state.ai_router, user.id)
    except RouteUnresolvedError as exc:
        raise ApiError(
            503, "model_not_chosen",
            "The trait-reading model hasn't been chosen yet. Nothing is wrong "
            "with your answers — they're saved.",
        ) from exc
    except AIError as exc:
        # The raw model output is already logged by the Guard (§7); this is the
        # user-facing half, and it says the answers are safe because that is
        # the thing they will actually be worried about.
        log_event(
            logger, "extract_failed",
            level=logging.ERROR, user_id=str(user.id), error=str(exc),
        )
        raise ApiError(
            502, "extraction_failed",
            "We couldn't read your answers just now. They're saved — try again "
            "in a moment.",
        ) from exc

    if outcome is None:
        return ExtractOut(status="queued")

    # S7-B6: compilation follows extraction automatically. Gated on
    # `outcome.changed` on purpose — an all-`keep` run leaves traits_hash
    # byte-identical, so the existing snapshot is still an accurate persona and
    # recompiling would burn an AI call to produce the same thing with a new
    # version number. The staleness rule and this trigger read the SAME hash,
    # so they can never disagree about whether a rebuild is owed.
    if outcome.changed:
        start_compile(request.app, user.id)

    return ExtractOut(
        status="done", kept=outcome.kept, updated=outcome.updated,
        retracted=outcome.retracted, added=outcome.added,
        declined=outcome.declined, changed=outcome.changed,
    )


@router.get("/traits", response_model=TraitsOut)
async def list_traits(user: CurrentUser, session: DbSession) -> TraitsOut:
    """S6-B6. Retracted rows ARE included: a retraction is a visible event in a
    person's profile history, not a disappearance (S6-B3, AC5). The UI decides
    how to show them; the API does not decide for it by hiding them."""
    rows = list(
        (
            await session.execute(
                select(Trait)
                .where(Trait.user_id == user.id)
                .order_by(Trait.category, Trait.created_at)
            )
        ).scalars()
    )
    return TraitsOut(
        traits=[TraitOut.build(t) for t in rows],
        traits_hash=compute_traits_hash(rows),
    )


async def _owned_trait(session, user_id: uuid.UUID, trait_id: str) -> Trait:
    try:
        parsed = uuid.UUID(trait_id)
    except ValueError as exc:
        raise ApiError(404, "not_found", "That trait doesn't exist.") from exc
    trait = (
        await session.execute(
            select(Trait).where(Trait.id == parsed, Trait.user_id == user_id)
        )
    ).scalar_one_or_none()
    if trait is None:
        # Same message whether it is absent or someone else's — a distinct
        # "not yours" would confirm that a given id exists.
        raise ApiError(404, "not_found", "That trait doesn't exist.")
    return trait


@router.post("/traits/{trait_id}/confirm", response_model=TraitOut)
async def confirm_trait(
    trait_id: str, user: CurrentUser, session: DbSession
) -> TraitOut:
    """S6-B8. `confirmed` is the status extraction must never quietly overwrite
    — decision log #10 — so this is deliberately a one-line status change plus
    its audit row, with nothing else attached to it."""
    trait = await _owned_trait(session, user.id, trait_id)
    if trait.status == "retracted":
        raise ApiError(
            409, "trait_retracted",
            "That one was already removed from your profile, so there's "
            "nothing to confirm.",
        )
    trait.status = "confirmed"
    session.add(TraitEvent(trait_id=trait.id, event="confirmed", detail="user confirmed"))
    await session.commit()
    log_event(logger, "trait_confirmed", user_id=str(user.id), trait_id=str(trait.id))
    return TraitOut.build(trait)


@router.post("/traits/{trait_id}/dispute", response_model=DisputeOut)
async def dispute_trait(
    trait_id: str, body: DisputeIn, request: Request,
    user: CurrentUser, session: DbSession,
) -> DisputeOut:
    """S6-B7. Marks the trait `disputed` and generates EXACTLY ONE follow-up
    question (`origin='dispute'`, linked by `trait_id`) so the next extraction
    corrects rather than re-infers.

    Ordering matters and is not arbitrary: the question is generated BEFORE
    anything is written. A dispute that marked the trait and then failed to
    produce its question would leave a disputed trait with no way to correct
    it — the state AC4 exists to forbid. Generate first, then write both
    together, or write nothing.
    """
    trait = await _owned_trait(session, user.id, trait_id)
    if trait.status == "retracted":
        raise ApiError(
            409, "trait_retracted",
            "That one was already removed from your profile, so there's "
            "nothing to dispute.",
        )

    existing = (
        await session.execute(
            select(Question).where(
                Question.origin == "dispute", Question.trait_id == trait.id
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        # Exactly one, per AC4. Disputing twice reuses the open question rather
        # than stacking a second one on the same trait.
        trait.status = "disputed"
        await session.commit()
        return DisputeOut(
            trait=TraitOut.build(trait),
            question_id=str(existing.id), question_text=existing.text,
        )

    try:
        provider, model = request.app.state.ai_router.resolve(DISPUTE_TASK)
    except RouteUnresolvedError as exc:
        raise ApiError(
            503, "model_not_chosen",
            "We can't write your follow-up question yet — the model for it "
            "hasn't been chosen. Your profile is unchanged.",
        ) from exc

    correction = (body.correction or "").strip()
    prompt = (
        "A person was told the following about themselves, and said it is "
        "wrong.\n\n"
        f"The trait: [{trait.category}] {trait.label} — {trait.description}\n"
    )
    if correction:
        prompt += f"\nWhat they say is actually true: {correction}\n"
    prompt += (
        "\nWrite ONE open question that invites them to describe what is really "
        "true here, in their own words. Do not defend the trait. Do not ask "
        "them to justify disagreeing. Address them as 'you'."
    )

    try:
        result = await guarded_structured_call(
            provider,
            GenRequest(
                task=DISPUTE_TASK, model=model,
                system_prompt=(
                    "You write single, warm, open questions that help someone "
                    "describe themselves accurately after an automated guess "
                    "about them got it wrong."
                ),
                messages=[Message(role="user", content=prompt)],
                temperature=0.4, max_tokens=DISPUTE_MAX_TOKENS,
            ),
            DISPUTE_FOLLOWUP_V1,
        )
    except AIError as exc:
        log_event(
            logger, "dispute_question_failed",
            level=logging.ERROR, user_id=str(user.id),
            trait_id=str(trait.id), error=str(exc),
        )
        raise ApiError(
            502, "dispute_question_failed",
            "We couldn't write your follow-up question just now. Your profile "
            "is unchanged — try again in a moment.",
        ) from exc

    text = (result.get("question_text") or "").strip()
    if not text:
        raise ApiError(
            502, "dispute_question_failed",
            "We couldn't write your follow-up question just now. Your profile "
            "is unchanged — try again in a moment.",
        )

    question = Question(
        # user_id is REQUIRED here, not optional: the questions table has a
        # CHECK that shared (baseline/pool) questions have a NULL user_id and
        # dispute questions do not. A dispute question belongs to one person.
        user_id=user.id,
        origin="dispute", code=None, pool_order=None,
        probe_area=_PROBE_AREA_FOR_CATEGORY.get(trait.category, "self_image"),
        text=text, trait_id=trait.id,
    )
    session.add(question)
    trait.status = "disputed"
    session.add(TraitEvent(
        trait_id=trait.id, event="disputed",
        detail=(f"user disputed; correction: {correction}" if correction
                else "user disputed")[:2000],
    ))
    await session.commit()

    log_event(
        logger, "trait_disputed",
        user_id=str(user.id), trait_id=str(trait.id),
        provider=provider.name, model=model,
        question_id=str(question.id), had_correction=bool(correction),
    )
    return DisputeOut(
        trait=TraitOut.build(trait),
        question_id=str(question.id), question_text=question.text,
    )


@router.get("/profile/extract/status")
async def extract_status(user: CurrentUser) -> dict:
    """Small, but it is what lets the UI say "still reading your answers"
    instead of showing a fake timer (§7 of new_user_creation.md, Step 7)."""
    return {"running": is_running(user.id)}
