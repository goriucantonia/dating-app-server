"""The questionnaire loop (S5-B1…B6, module_1_data_collection.md A4/A5).

- GET /questions — everything answerable by THIS user (baseline + pool +
  their own dispute questions), each with answered state and the answer text
  when present. This one endpoint drives save/resume AND the edit view.
- GET /questions/next-batch — up to 5 unanswered POOL questions by
  pool_order. The answer set IS the cursor; there is no assignment table.
  Pool done → a normal `pool_exhausted` payload, never a 4xx (§5, A5.4).
- PUT /answers/{question_id} — ONE upsert path for the first write and every
  later edit. The 200-character minimum applies to baseline, pool, AND
  dispute answers (§18 — the scope is written down on purpose).

Dispute questions never count toward answered_pool (§13, S5-B6): they are
per-user, AI-generated, and outside pool progress.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select

from app.errors import ApiError
from app.logging_setup import log_event
from app.models import Answer, Question, Trait
from app.security import CurrentUser, DbSession

router = APIRouter(tags=["questions"])
logger = logging.getLogger("app.questions")


class QuestionOut(BaseModel):
    id: str
    origin: str
    code: str | None
    pool_order: int | None
    probe_area: str
    text: str
    answered: bool
    answer_text: str | None
    answer_updated_at: datetime | None

    @classmethod
    def build(cls, q: Question, a: Answer | None) -> QuestionOut:
        return cls(
            id=str(q.id), origin=q.origin, code=q.code, pool_order=q.pool_order,
            probe_area=q.probe_area, text=q.text,
            answered=a is not None,
            answer_text=a.answer_text if a else None,
            answer_updated_at=a.updated_at if a else None,
        )


class QuestionsOut(BaseModel):
    questions: list[QuestionOut]


class PoolProgress(BaseModel):
    answered_pool: int
    total_pool: int


class NextBatchOut(BaseModel):
    status: str  # 'ok' | 'pool_exhausted'
    questions: list[QuestionOut]
    progress: PoolProgress


class AnswerIn(BaseModel):
    answer_text: str = Field(min_length=200)


def _answerable_filter(user_id: uuid.UUID):
    """Baseline + pool are global rows; dispute questions only their owner's."""
    return or_(Question.user_id.is_(None), Question.user_id == user_id)


async def _my_answers(session, user_id: uuid.UUID) -> dict[uuid.UUID, Answer]:
    answers = (
        await session.scalars(select(Answer).where(Answer.user_id == user_id))
    ).all()
    return {a.question_id: a for a in answers}


@router.get("/questions")
async def list_questions(user: CurrentUser, session: DbSession) -> QuestionsOut:
    questions = (
        await session.scalars(select(Question).where(_answerable_filter(user.id)))
    ).all()
    # Baseline first (by code), then pool (by pool_order), then the user's
    # dispute questions (oldest first).
    by_origin = {"baseline": 0, "pool": 1, "dispute": 2}
    questions = sorted(
        questions,
        key=lambda q: (by_origin[q.origin], q.code or "", q.pool_order or 0, q.created_at),
    )
    answers = await _my_answers(session, user.id)
    return QuestionsOut(
        questions=[QuestionOut.build(q, answers.get(q.id)) for q in questions]
    )


async def _pool_progress(session, user_id: uuid.UUID) -> PoolProgress:
    total = await session.scalar(
        select(func.count()).select_from(Question).where(Question.origin == "pool")
    )
    # Only POOL answers count — a dispute or baseline answer never bumps this
    # (S5-B6, §13).
    answered = await session.scalar(
        select(func.count())
        .select_from(Answer)
        .join(Question, Question.id == Answer.question_id)
        .where(Answer.user_id == user_id, Question.origin == "pool")
    )
    return PoolProgress(answered_pool=answered or 0, total_pool=total or 0)


@router.get("/questions/next-batch")
async def next_batch(user: CurrentUser, session: DbSession) -> NextBatchOut:
    progress = await _pool_progress(session, user.id)
    answered_ids = select(Answer.question_id).where(Answer.user_id == user.id)
    batch = (
        await session.scalars(
            select(Question)
            .where(Question.origin == "pool", Question.id.not_in(answered_ids))
            .order_by(Question.pool_order)
            .limit(5)
        )
    ).all()
    if not batch:
        # A defined product state with its own UI — not an error, not a 4xx.
        log_event(logger, "pool_exhausted", user_id=str(user.id))
        return NextBatchOut(status="pool_exhausted", questions=[], progress=progress)
    return NextBatchOut(
        status="ok",
        questions=[QuestionOut.build(q, None) for q in batch],
        progress=progress,
    )


@router.put("/answers/{question_id}")
async def upsert_answer(
    question_id: uuid.UUID, payload: AnswerIn, user: CurrentUser, session: DbSession
) -> QuestionOut:
    question = await session.get(Question, question_id)
    if question is None or (
        question.user_id is not None and question.user_id != user.id
    ):
        raise ApiError(404, "question_not_found", "That question doesn't exist.")

    answer = await session.scalar(
        select(Answer).where(
            Answer.user_id == user.id, Answer.question_id == question_id
        )
    )
    if answer is None:
        answer = Answer(
            user_id=user.id, question_id=question_id, answer_text=payload.answer_text
        )
        session.add(answer)
        await session.commit()
        log_event(
            logger, "answer_saved", kind="created",
            user_id=str(user.id), question=question.code or str(question.id),
            origin=question.origin, length=len(payload.answer_text),
        )
    else:
        old_length = len(answer.answer_text)
        # Which traits were built on the answer being edited (S5-B5, §7) —
        # empty until extraction exists (Step 6), but logged from day one.
        affected = (
            await session.scalars(
                select(Trait.id).where(
                    Trait.user_id == user.id,
                    Trait.source_answer_ids.contains([answer.id]),
                )
            )
        ).all()
        answer.answer_text = payload.answer_text
        answer.updated_at = datetime.now(UTC)
        session.add(answer)
        await session.commit()
        log_event(
            logger, "answer_saved", kind="edited",
            user_id=str(user.id), question=question.code or str(question.id),
            origin=question.origin,
            old_length=old_length, new_length=len(payload.answer_text),
            traits_sourced_from_this_answer=[str(t) for t in affected],
        )
    return QuestionOut.build(question, answer)
