"""Analysis endpoints (S9-B9, B10).

- POST /analyses        — start a run; returns immediately with `matching`
- GET  /analyses/{id}   — the UI's SINGLE polling target for the whole journey
- GET  /analyses        — history, newest first

The 409 on a second run is **state, not failure** (communication_protocol.md
§5, §17): the caller is told which analysis is already running so it can go
poll that one, rather than being handed an error to display.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.errors import ApiError
from app.logging_setup import log_event
from app.matching import start_and_run
from app.models import Analysis, AnalysisCandidate, CandidateScore, Trait, User
from app.security import CurrentUser, DbSession
from app.users import compute_age

router = APIRouter(tags=["analyses"])
logger = logging.getLogger("app.analyses")

# A run is "active" in these states — the ones where starting a second would
# duplicate work or race it. `matched` is NOT active: matching is done and
# Step 11 has not started simulating.
ACTIVE_STATES = ("matching", "simulating")

_tasks: set[asyncio.Task] = set()


class CandidateOut(BaseModel):
    candidate_user_id: str
    display_name: str
    age: int
    # S10-B1. `is_demo` must be present WHEREVER a user is rendered
    # (communication_protocol.md §6) — a demo profile that loses its label
    # somewhere in the UI is a person the user thinks is real.
    is_demo: bool
    # Trait LABELS only, grouped by category. Descriptions stay private to the
    # candidate: they are written about someone by an AI reading their intimate
    # answers, and showing them to a stranger is a different product. This is a
    # WIRE rule, not a UI convention — the probe checks the raw body.
    trait_labels: dict[str, list[str]]
    rank: int
    fit_forward: float
    fit_backward: float
    compatibility: float
    shared_interests: list[str]
    reason_summary: str
    snapshot_id: str
    # S12: the score, once the dates have been judged. Null until then, and
    # null is DIFFERENT from zero — every date with this person may have failed
    # too early to judge, and 0.0 would say "they were terrible together" about
    # an evening that never happened.
    final_score: float | None = None
    dates_completed: int | None = None
    dates_incomplete: int | None = None


class AnalysisOut(BaseModel):
    id: str
    status: str
    pool_status: str | None
    error: str | None
    created_at: datetime
    # S11-B10. The simulation pipeline's real stage, in the sentence the UI
    # shows. It rides on THIS payload rather than a second endpoint because
    # the analysis row is the one object the UI polls for the whole journey —
    # a separate progress endpoint would mean two loops and two truths.
    progress: dict | None = None
    candidates: list[CandidateOut] = []
    # S15-B3: how many of this analysis's people have since deleted their
    # accounts. Their candidate rows, dates and scores cascaded away with them
    # (data_hygiene.md — privacy beats history); the analysis row survives
    # with a gap, and this is the gap, labeled. Computed from the count
    # matching recorded, never from `pool_status`, which cannot tell 1 from 2.
    removed_candidates: int = 0
    # The plain sentence the UI shows when there is nobody. Server-side so the
    # honest-empty-pool wording cannot drift between clients (§26).
    message: str | None = None


async def _build(session, analysis: Analysis) -> AnalysisOut:
    rows = (
        await session.execute(
            select(AnalysisCandidate, User)
            .join(User, User.id == AnalysisCandidate.candidate_user_id)
            .where(AnalysisCandidate.analysis_id == analysis.id)
            .order_by(AnalysisCandidate.rank)
        )
    ).all()

    scores = {
        row.candidate_user_id: row
        for row in (
            await session.execute(
                select(CandidateScore).where(
                    CandidateScore.analysis_id == analysis.id
                )
            )
        ).scalars()
    }

    # One query for every candidate's labels rather than one per card.
    labels: dict[uuid.UUID, dict[str, list[str]]] = {}
    if rows:
        for t in (
            await session.execute(
                select(Trait).where(
                    Trait.user_id.in_([c.candidate_user_id for c, _ in rows]),
                    Trait.status != "retracted",
                ).order_by(Trait.category, Trait.label)
            )
        ).scalars():
            labels.setdefault(t.user_id, {}).setdefault(t.category, []).append(t.label)

    had = analysis.candidate_count if analysis.candidate_count is not None else len(rows)
    return AnalysisOut(
        id=str(analysis.id), status=analysis.status,
        pool_status=analysis.pool_status, error=analysis.error,
        created_at=analysis.created_at, progress=analysis.progress,
        removed_candidates=max(0, had - len(rows)),
        candidates=[
            CandidateOut(
                candidate_user_id=str(c.candidate_user_id),
                display_name=u.display_name, age=compute_age(u.birth_date),
                is_demo=u.is_demo, trait_labels=labels.get(c.candidate_user_id, {}),
                rank=c.rank, fit_forward=float(c.fit_forward),
                fit_backward=float(c.fit_backward),
                compatibility=float(c.compatibility),
                shared_interests=list(c.shared_interests or []),
                reason_summary=c.reason_summary, snapshot_id=str(c.snapshot_id),
                final_score=(
                    float(scores[c.candidate_user_id].final_score)
                    if c.candidate_user_id in scores
                    else None
                ),
                dates_completed=(
                    scores[c.candidate_user_id].dates_completed
                    if c.candidate_user_id in scores
                    else None
                ),
                dates_incomplete=(
                    scores[c.candidate_user_id].dates_incomplete
                    if c.candidate_user_id in scores
                    else None
                ),
            )
            for c, u in rows
        ],
        message=(
            "There is no one to match you with yet."
            if analysis.status == "no_candidates"
            else None
        ),
    )


async def _run(app, analysis_id: uuid.UUID) -> None:
    factory = async_sessionmaker(app.state.engine, expire_on_commit=False)
    async with factory() as session:
        await start_and_run(session, app.state.ai_router, analysis_id)


@router.post("/analyses", response_model=AnalysisOut, status_code=202)
async def create_analysis(
    request: Request, user: CurrentUser, session: DbSession
) -> AnalysisOut:
    """S9-B9/B10. Returns immediately — refreshing embeddings can mean AI
    calls, which is exactly why this is a background job."""
    active = (
        await session.execute(
            select(Analysis)
            .where(Analysis.user_id == user.id, Analysis.status.in_(ACTIVE_STATES))
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if active is not None:
        raise ApiError(
            409, "analysis_in_progress",
            "You've already got one running — hang on for that one to finish.",
            fields=[{"field": "analysis_id", "message": str(active.id)}],
        )

    analysis = Analysis(user_id=user.id, status="matching")
    session.add(analysis)
    await session.commit()

    log_event(
        logger, "analysis_started",
        user_id=str(user.id), analysis_id=str(analysis.id),
    )
    task = asyncio.create_task(_run(request.app, analysis.id))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return await _build(session, analysis)


@router.get("/analyses/{analysis_id}", response_model=AnalysisOut)
async def get_analysis(
    analysis_id: str, user: CurrentUser, session: DbSession
) -> AnalysisOut:
    try:
        parsed = uuid.UUID(analysis_id)
    except ValueError as exc:
        raise ApiError(404, "not_found", "That analysis doesn't exist.") from exc
    analysis = (
        await session.execute(
            select(Analysis).where(Analysis.id == parsed, Analysis.user_id == user.id)
        )
    ).scalar_one_or_none()
    if analysis is None:
        raise ApiError(404, "not_found", "That analysis doesn't exist.")
    return await _build(session, analysis)


@router.get("/analyses")
async def list_analyses(user: CurrentUser, session: DbSession) -> dict:
    """History, newest first — the revisitable-results decision."""
    rows = (
        await session.execute(
            select(Analysis)
            .where(Analysis.user_id == user.id)
            .order_by(Analysis.created_at.desc())
        )
    ).scalars()
    return {"analyses": [(await _build(session, a)).model_dump() for a in rows]}
