"""Simulation endpoints (S11-B11).

- POST /analyses/{id}/simulate  — start the pipeline; 409 unless `matched`
- GET  /analyses/{id}/dates     — every date with its status and setting
- GET  /dates/{id}/transcript   — the messages, INCLUDING per-turn state

That last one is the **one place** in the whole API where both agents' inner
state is deliberately exposed (decision log #4, `communication_protocol.md`
§6). Everywhere else — calibration chat especially — the state block is stored
and never returned, because a person meeting their own AI self should be
reading what it said, not its telemetry. On a finished date the telemetry IS
the product: "she was enjoying this and he had checked out ten messages
earlier" is the thing the results screen exists to show.

What still never leaves the server, here as everywhere: system prompts, trait
descriptions, and raw answers. A transcript carries what was SAID plus how
each agent felt while saying it — nothing about what either profile contains.

The 409 on `POST /simulate` is **state, not failure** (§5, §17): the caller is
told what the analysis is currently doing so it can go and poll that, rather
than being handed an error to display.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sqlalchemy import select

from app.errors import ApiError
from app.judging import is_judgeable
from app.logging_setup import log_event
from app.models import Analysis, DateEvaluation, DateMessage, SimulatedDate, User
from app.security import CurrentUser, DbSession
from app.simulation import start_pipeline

router = APIRouter(tags=["simulation"])
logger = logging.getLogger("app.simulation")


class EvaluationOut(BaseModel):
    """What the judge found (S12). `criteria` and `date_score` are BOTH here,
    deliberately: the score is computed in code from the criteria, and shipping
    only one of them would make the arithmetic unverifiable by the person whose
    date it was."""

    criteria: dict
    date_score: float
    is_partial: bool
    clicked_subjects: list
    clashes: list
    per_peer_summary: dict
    verdict_summary: str
    judge_provider: str
    judge_model: str
    rubric_version: str


class DateOut(BaseModel):
    date_id: str
    candidate_user_id: str
    candidate_name: str
    ordinal: int
    status: str
    setting_name: str
    description: str
    sensory_details: str
    anchored_in_interest: str
    message_count: int
    error: str | None
    evaluation: EvaluationOut | None = None
    # Stated on the wire rather than re-derived by every client (S12-B7, AC4).
    # "This date was too short to score" is a thing the results screen has to
    # say out loud, and a client recomputing the 10-message rule is a client
    # that can disagree with the server about someone's score.
    excluded_from_score: bool = False


class TranscriptMessageOut(BaseModel):
    seq: int
    speaker: str
    reply: str
    # The deliberate exposure. Null on `environment` rows — an event has no
    # inner life, and a zeroed state block would read as an agent who felt
    # nothing.
    state: dict | None
    provider: str | None
    model_id: str | None


class TranscriptOut(BaseModel):
    date_id: str
    status: str
    setting_name: str
    description: str
    sensory_details: str
    user_display_name: str
    candidate_display_name: str
    schema_version: str
    messages: list[TranscriptMessageOut]


def _evaluation_out(row: DateEvaluation | None) -> EvaluationOut | None:
    if row is None:
        return None
    return EvaluationOut(
        criteria=row.criteria, date_score=float(row.date_score),
        is_partial=row.is_partial, clicked_subjects=list(row.clicked or []),
        clashes=list(row.clashes or []), per_peer_summary=dict(row.per_peer or {}),
        verdict_summary=row.verdict, judge_provider=row.judge_provider,
        judge_model=row.judge_model, rubric_version=row.rubric_version,
    )


async def _owned_analysis(session, user_id: uuid.UUID, raw_id: str) -> Analysis:
    try:
        parsed = uuid.UUID(raw_id)
    except ValueError as exc:
        raise ApiError(404, "not_found", "That analysis doesn't exist.") from exc
    row = (
        await session.execute(
            select(Analysis).where(Analysis.id == parsed, Analysis.user_id == user_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise ApiError(404, "not_found", "That analysis doesn't exist.")
    return row


@router.post("/analyses/{analysis_id}/simulate", status_code=202)
async def simulate(
    analysis_id: str, request: Request, user: CurrentUser, session: DbSession
) -> dict:
    """S11-B11. Returns immediately: a full analysis is six dates of tens of
    turns each against throttled free models, which is minutes to tens of
    minutes. The client polls `GET /analyses/{id}`.

    Kept as an explicit endpoint rather than auto-chaining off matching,
    because the reveal is a decision point — the user looks at who was found
    and chooses (S10-U12). It doubles as the retry path.
    """
    analysis = await _owned_analysis(session, user.id, analysis_id)

    if analysis.status == "simulating":
        raise ApiError(
            409, "simulation_in_progress",
            "The dates are already running — hang on for those to finish.",
            fields=[{"field": "analysis_id", "message": str(analysis.id)}],
        )
    if analysis.status != "matched":
        # Named states, not a generic refusal: "you have not been matched yet"
        # and "this one already finished" are different things to be told.
        raise ApiError(
            409, "not_ready_to_simulate",
            {
                "matching": "We're still working out who fits — give it a moment.",
                "no_candidates": "There's nobody to go on a date with yet.",
                "complete": "These dates have already run — open the results.",
                "failed": "That analysis didn't finish. Start a new one.",
            }.get(analysis.status, "This analysis isn't ready for dates."),
            fields=[{"field": "status", "message": analysis.status}],
        )

    started = start_pipeline(request.app, analysis.id)
    log_event(
        logger, "simulation_requested", user_id=str(user.id),
        analysis_id=str(analysis.id), started=started,
    )
    return {
        "analysis_id": str(analysis.id),
        "status": "simulating" if started else "already_running",
    }


@router.get("/analyses/{analysis_id}/dates")
async def list_dates(
    analysis_id: str, user: CurrentUser, session: DbSession
) -> dict:
    """S11-B11. Every date of the analysis, in the order they run."""
    analysis = await _owned_analysis(session, user.id, analysis_id)
    rows = (
        await session.execute(
            select(SimulatedDate, User)
            .join(User, User.id == SimulatedDate.candidate_user_id)
            .where(SimulatedDate.analysis_id == analysis.id)
            .order_by(SimulatedDate.created_at, SimulatedDate.ordinal)
        )
    ).all()

    counts: dict[uuid.UUID, int] = {}
    for d, _ in rows:
        counts[d.id] = len(
            (
                await session.execute(
                    select(DateMessage.id).where(DateMessage.date_id == d.id)
                )
            ).all()
        )

    evaluations = {
        e.date_id: e
        for e in (
            await session.execute(
                select(DateEvaluation).where(
                    DateEvaluation.date_id.in_([d.id for d, _ in rows])
                )
            )
        ).scalars()
    } if rows else {}

    return {
        "analysis_id": str(analysis.id),
        "status": analysis.status,
        "progress": analysis.progress,
        "dates": [
            DateOut(
                date_id=str(d.id),
                candidate_user_id=str(d.candidate_user_id),
                candidate_name=u.display_name,
                ordinal=d.ordinal,
                status=d.status,
                setting_name=d.scenario.get("setting_name", ""),
                description=d.scenario.get("description", ""),
                sensory_details=d.scenario.get("sensory_details", ""),
                anchored_in_interest=d.scenario.get("anchored_in_interest", ""),
                message_count=counts[d.id],
                error=d.error,
                evaluation=_evaluation_out(evaluations.get(d.id)),
                excluded_from_score=(
                    d.status in ("complete", "incomplete", "failed")
                    and not is_judgeable(d.status, counts[d.id])
                ),
            ).model_dump()
            for d, u in rows
        ],
    }


@router.get("/dates/{date_id}/transcript", response_model=TranscriptOut)
async def transcript(
    date_id: str, user: CurrentUser, session: DbSession
) -> TranscriptOut:
    """S11-B11. The messages with their per-turn state — see the module note."""
    try:
        parsed = uuid.UUID(date_id)
    except ValueError as exc:
        raise ApiError(404, "not_found", "That date doesn't exist.") from exc

    row = (
        await session.execute(
            select(SimulatedDate, Analysis)
            .join(Analysis, Analysis.id == SimulatedDate.analysis_id)
            .where(SimulatedDate.id == parsed, Analysis.user_id == user.id)
        )
    ).one_or_none()
    if row is None:
        raise ApiError(404, "not_found", "That date doesn't exist.")
    date, analysis = row

    me = (
        await session.execute(select(User).where(User.id == analysis.user_id))
    ).scalar_one()
    other = (
        await session.execute(select(User).where(User.id == date.candidate_user_id))
    ).scalar_one()

    messages = (
        await session.execute(
            select(DateMessage)
            .where(DateMessage.date_id == date.id)
            .order_by(DateMessage.seq)
        )
    ).scalars()

    return TranscriptOut(
        date_id=str(date.id),
        status=date.status,
        setting_name=date.scenario.get("setting_name", ""),
        description=date.scenario.get("description", ""),
        sensory_details=date.scenario.get("sensory_details", ""),
        user_display_name=me.display_name,
        candidate_display_name=other.display_name,
        schema_version=date.schema_version,
        messages=[
            TranscriptMessageOut(
                seq=m.seq, speaker=m.speaker, reply=m.reply, state=m.state,
                provider=m.provider, model_id=m.model_id,
            )
            for m in messages
        ],
    )
