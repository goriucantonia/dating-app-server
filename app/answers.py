"""Saving an answer — the ONE upsert path (S5-B4, S15-B5, §16).

`PUT /answers/{question_id}` and demo-profile seeding both come through
here: first write and every later edit, for a person or for a demo account,
with the same log line. The 50-character floor is the caller's to enforce at
the boundary (pydantic at the endpoint, the seed loader for demo profiles);
the database CHECK is the last line.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging_setup import log_event
from app.models import Answer, Question, Trait

logger = logging.getLogger("app.questions")


async def save_answer(
    session: AsyncSession, user_id: uuid.UUID, question: Question, text: str
) -> tuple[Answer, str]:
    """Create or edit. Returns the row and `created` / `edited` / `unchanged`
    — the last one so a seeder re-running on a healthy database can say it
    did nothing, and mean it (§1, second-run no-op)."""
    answer = await session.scalar(
        select(Answer).where(Answer.user_id == user_id, Answer.question_id == question.id)
    )
    if answer is None:
        answer = Answer(user_id=user_id, question_id=question.id, answer_text=text)
        session.add(answer)
        await session.commit()
        log_event(
            logger, "answer_saved", kind="created",
            user_id=str(user_id), question=question.code or str(question.id),
            origin=question.origin, length=len(text),
        )
        return answer, "created"

    if answer.answer_text == text:
        return answer, "unchanged"

    old_length = len(answer.answer_text)
    # Which traits were built on the answer being edited (S5-B5, §7).
    affected = (
        await session.scalars(
            select(Trait.id).where(
                Trait.user_id == user_id,
                Trait.source_answer_ids.contains([answer.id]),
            )
        )
    ).all()
    answer.answer_text = text
    answer.updated_at = datetime.now(UTC)
    session.add(answer)
    await session.commit()
    log_event(
        logger, "answer_saved", kind="edited",
        user_id=str(user_id), question=question.code or str(question.id),
        origin=question.origin,
        old_length=old_length, new_length=len(text),
        traits_sourced_from_this_answer=[str(t) for t in affected],
    )
    return answer, "edited"
