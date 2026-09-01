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
from sqlalchemy import func, select

from app.errors import ApiError
from app.judging import is_judgeable
from app.logging_setup import log_event
from app.models import (
    Analysis,
    AnalysisCandidate,
    DateEvaluation,
    DateMessage,
    SimulatedDate,
    User,
)
from app.security import CurrentUser, DbSession
from app.simulation import ended_by, start_pipeline, to_views

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
    # The whole transcript, environment rows included.
    message_count: int
    # What the two of them actually SAID — the number the judging threshold
    # reads, and the one to show next to "too short to score".
    turn_count: int
    error: str | None
    # How the date finished, decided by the SAME rule the pipeline used to stop
    # it (S13-U8): `mutual_wants_to_end`, `cap`, or null when it did not reach
    # either — an incomplete date, or one still running. The UI says it; it
    # never re-derives it, because a client with its own idea of "mutual" is a
    # client that can disagree with the log about how an evening ended.
    ended_by: str | None = None
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
    # The analysis this date belongs to, so the viewer can watch the SAME
    # poller everyone else watches and say "other dates are still running"
    # while it is true (S13-U14) — rather than carrying that fact in a URL.
    analysis_id: str
    status: str
    setting_name: str
    description: str
    sensory_details: str
    user_display_name: str
    candidate_display_name: str
    schema_version: str
    ended_by: str | None
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


def simulate_refusal(status: str, has_candidates: bool) -> tuple[str, str] | None:
    """Why `POST /simulate` says no, as (code, message) — or None to proceed.

    Pure so it can be unit-tested without a database, because the boundary it
    draws is the one that matters (§18): **a `failed` analysis CAN be retried,
    but only if it got as far as having candidates.** The pipeline resumes from
    its checkpointed rows (`ensure_dates` reuses them; a finished date is a
    no-op on re-run), so the UI's "picks up where it stopped" is true rather
    than hopeful (S13-U5). A `failed` analysis with no candidates died in
    MATCHING — there is nothing to resume, and the honest answer is to start a
    new one.
    """
    if status == "simulating":
        return (
            "simulation_in_progress",
            "The dates are already running — hang on for those to finish.",
        )
    if status == "matched":
        return None
    if status == "failed" and has_candidates:
        return None
    # Named states, not a generic refusal: "you have not been matched yet" and
    # "this one already finished" are different things to be told.
    return (
        "not_ready_to_simulate",
        {
            "matching": "We're still working out who fits — give it a moment.",
            "no_candidates": "There's nobody to go on a date with yet.",
            "complete": "These dates have already run — open the results.",
            "failed": (
                "That analysis stopped before anyone was matched. Start a new one."
            ),
        }.get(status, "This analysis isn't ready for dates."),
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

    has_candidates = (
        await session.execute(
            select(func.count())
            .select_from(AnalysisCandidate)
            .where(AnalysisCandidate.analysis_id == analysis.id)
        )
    ).scalar_one() > 0
    refusal = simulate_refusal(analysis.status, has_candidates)
    if refusal is not None:
        code, message = refusal
        in_progress = code == "simulation_in_progress"
        raise ApiError(
            409, code, message,
            fields=[{
                "field": "analysis_id" if in_progress else "status",
                "message": str(analysis.id) if in_progress else analysis.status,
            }],
        )

    resuming = analysis.status == "failed"
    started = start_pipeline(request.app, analysis.id)
    log_event(
        logger, "simulation_requested", user_id=str(user.id),
        analysis_id=str(analysis.id), started=started,
        # S13-U5 / §7: a retry after a failure is logged AS a retry, with the
        # stage it is picking up from, so the log can say "resumed" rather
        # than leaving a reader to infer it from two start lines.
        resumed_after_failure=resuming,
        failed_stage=(analysis.progress or {}).get("stage") if resuming else None,
        previous_error=analysis.error if resuming else None,
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

    # Two counts, because they mean different things and the difference is
    # load-bearing (revised 2026-09-01). `message_count` is how long the
    # transcript is; `turn_count` is how much was SAID, and the judging
    # threshold reads the second one — an environment row is scenery, not a
    # person talking. Shipping only the first would leave a client unable to
    # explain why a 10-row date was excluded.
    counts: dict[uuid.UUID, int] = {}
    turns: dict[uuid.UUID, int] = {}
    endings: dict[uuid.UUID, str | None] = {}
    for d, _ in rows:
        views = to_views(
            list(
                (
                    await session.execute(
                        select(DateMessage)
                        .where(DateMessage.date_id == d.id)
                        .order_by(DateMessage.seq)
                    )
                ).scalars()
            )
        )
        counts[d.id] = len(views)
        turns[d.id] = sum(1 for v in views if v.speaker != "environment")
        # Only a finished date has an ending; a running one has a transcript
        # that happens to satisfy neither rule yet.
        endings[d.id] = ended_by(views) if d.status == "complete" else None

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
                turn_count=turns[d.id],
                error=d.error,
                ended_by=endings[d.id],
                evaluation=_evaluation_out(evaluations.get(d.id)),
                excluded_from_score=(
                    d.status in ("complete", "incomplete", "failed")
                    and not is_judgeable(d.status, turns[d.id])
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

    messages = list(
        (
            await session.execute(
                select(DateMessage)
                .where(DateMessage.date_id == date.id)
                .order_by(DateMessage.seq)
            )
        ).scalars()
    )

    return TranscriptOut(
        date_id=str(date.id),
        analysis_id=str(analysis.id),
        status=date.status,
        setting_name=date.scenario.get("setting_name", ""),
        description=date.scenario.get("description", ""),
        sensory_details=date.scenario.get("sensory_details", ""),
        user_display_name=me.display_name,
        candidate_display_name=other.display_name,
        schema_version=date.schema_version,
        ended_by=ended_by(to_views(messages)) if date.status == "complete" else None,
        messages=[
            TranscriptMessageOut(
                seq=m.seq, speaker=m.speaker, reply=m.reply, state=m.state,
                provider=m.provider, model_id=m.model_id,
            )
            for m in messages
        ],
    )
