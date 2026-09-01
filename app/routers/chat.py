"""Chat endpoints (S14-B2, B3, B4, B8).

- POST /analyses/{id}/select                 — choose the one person; creates the session
- GET  /chat/sessions                        — the user's sessions, active first
- GET  /chat/sessions/{id}                   — header detail: match, trait labels, digest
- GET  /chat/sessions/{id}/messages?after_seq=&limit=
- POST /chat/sessions/{id}/messages          — send + receive
- POST /chat/sessions/{id}/end

**No payload here carries `state`** (S14-B5, `communication_protocol.md` §6).
`ChatMessageOut` lists its fields explicitly and the field is not among them
— a chat payload that could carry the persona's inner state is one edit away
from carrying it, and a live conversation where you watch the other side's
connection meter is a different product.

Trait LABELS only in the header, the same wire rule as candidates.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.ai.base import RouteUnresolvedError
from app.chat import (
    ReplyFailed,
    end_session,
    existing_session,
    reply,
    select_match,
    selection_refusal,
)
from app.errors import ApiError
from app.logging_setup import log_event
from app.models import Analysis, AnalysisCandidate, ChatMessage, ChatSession, Trait, User
from app.security import CurrentUser, DbSession

router = APIRouter(tags=["chat"])
logger = logging.getLogger("app.chat")

PAGE_LIMIT = 50


class SelectIn(BaseModel):
    candidate_user_id: str


class MatchOut(BaseModel):
    user_id: str
    display_name: str
    # Always, wherever a user is rendered (communication_protocol.md §6).
    is_demo: bool


class ChatMessageOut(BaseModel):
    """Deliberately no `state`."""

    message_id: str
    seq: int
    sender: str  # user | persona
    text: str
    created_at: datetime


class SessionOut(BaseModel):
    session_id: str
    analysis_id: str
    match: MatchOut
    status: str
    created_at: datetime
    ended_at: datetime | None
    last_message: ChatMessageOut | None = None


class SessionDetailOut(SessionOut):
    # For the header sheet (S14-U5): labels only, the digest, and the way back.
    trait_labels: dict[str, list[str]]
    date_digest: str
    snapshot_id: str


class ReplyOut(BaseModel):
    user_message: ChatMessageOut
    persona_message: ChatMessageOut
    compacted: bool


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


def _msg_out(m: ChatMessage) -> ChatMessageOut:
    return ChatMessageOut(
        message_id=str(m.id), seq=m.seq, sender=m.sender, text=m.text_,
        created_at=m.created_at,
    )


async def _session_out(session, convo: ChatSession, match: User) -> SessionOut:
    last = (
        await session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == convo.id)
            .order_by(ChatMessage.seq.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return SessionOut(
        session_id=str(convo.id), analysis_id=str(convo.analysis_id),
        match=MatchOut(
            user_id=str(match.id), display_name=match.display_name, is_demo=match.is_demo
        ),
        status=convo.status, created_at=convo.created_at, ended_at=convo.ended_at,
        last_message=_msg_out(last) if last else None,
    )


async def _owned_session(session, user_id: uuid.UUID, raw: str) -> tuple[ChatSession, User]:
    try:
        parsed = uuid.UUID(raw)
    except ValueError as exc:
        raise ApiError(404, "not_found", "That chat doesn't exist.") from exc
    row = (
        await session.execute(
            select(ChatSession, User)
            .join(User, User.id == ChatSession.match_user_id)
            .where(ChatSession.id == parsed, ChatSession.user_id == user_id)
        )
    ).one_or_none()
    if row is None:
        raise ApiError(404, "not_found", "That chat doesn't exist.")
    return row


@router.post("/analyses/{analysis_id}/select", response_model=SessionOut, status_code=201)
async def select_endpoint(
    analysis_id: str, body: SelectIn, user: CurrentUser, session: DbSession
) -> SessionOut:
    """S14-B2/B3. The one irreversible choice in an analysis."""
    try:
        parsed = uuid.UUID(analysis_id)
        candidate_id = uuid.UUID(body.candidate_user_id)
    except ValueError as exc:
        raise ApiError(404, "not_found", "That analysis doesn't exist.") from exc
    analysis = (
        await session.execute(
            select(Analysis).where(Analysis.id == parsed, Analysis.user_id == user.id)
        )
    ).scalar_one_or_none()
    if analysis is None:
        raise ApiError(404, "not_found", "That analysis doesn't exist.")

    candidate = (
        await session.execute(
            select(AnalysisCandidate).where(
                AnalysisCandidate.analysis_id == analysis.id,
                AnalysisCandidate.candidate_user_id == candidate_id,
            )
        )
    ).scalar_one_or_none()
    already = await existing_session(session, analysis.id)

    refusal = selection_refusal(
        analysis.status, is_candidate=candidate is not None,
        already_selected=already is not None,
    )
    if refusal is not None:
        status, code, message = refusal
        log_event(
            logger, "chat_selection_refused", user_id=str(user.id),
            analysis_id=str(analysis.id), code=code, analysis_status=analysis.status,
        )
        raise ApiError(
            status, code, message,
            fields=(
                [{"field": "session_id", "message": str(already.id)}] if already else None
            ),
        )

    convo = await select_match(session, analysis, candidate)
    match = (
        await session.execute(select(User).where(User.id == convo.match_user_id))
    ).scalar_one()
    return await _session_out(session, convo, match)


@router.get("/chat/sessions")
async def list_sessions(user: CurrentUser, session: DbSession) -> dict:
    """S14-B8. Active first, then ended; newest first within each."""
    rows = (
        await session.execute(
            select(ChatSession, User)
            .join(User, User.id == ChatSession.match_user_id)
            .where(ChatSession.user_id == user.id)
            .order_by(ChatSession.status.asc(), ChatSession.created_at.desc())
        )
    ).all()
    return {
        "sessions": [
            (await _session_out(session, convo, match)).model_dump(mode="json")
            for convo, match in rows
        ]
    }


@router.get("/chat/sessions/{session_id}", response_model=SessionDetailOut)
async def session_detail(
    session_id: str, user: CurrentUser, session: DbSession
) -> SessionDetailOut:
    convo, match = await _owned_session(session, user.id, session_id)
    base = await _session_out(session, convo, match)
    labels: dict[str, list[str]] = {}
    for t in (
        await session.execute(
            select(Trait)
            .where(Trait.user_id == match.id, Trait.status != "retracted")
            .order_by(Trait.category, Trait.label)
        )
    ).scalars():
        labels.setdefault(t.category, []).append(t.label)
    return SessionDetailOut(
        **base.model_dump(), trait_labels=labels,
        date_digest=convo.date_digest, snapshot_id=str(convo.snapshot_id),
    )


@router.get("/chat/sessions/{session_id}/messages")
async def list_messages(
    session_id: str, user: CurrentUser, session: DbSession,
    after_seq: int = Query(0, ge=0), limit: int = Query(PAGE_LIMIT, ge=1, le=200),
) -> dict:
    """S14-B8. Paged, ascending from `after_seq`. No `state` field."""
    convo, _ = await _owned_session(session, user.id, session_id)
    rows = list(
        (
            await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == convo.id, ChatMessage.seq > after_seq)
                .order_by(ChatMessage.seq)
                .limit(limit + 1)
            )
        ).scalars()
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    return {
        "session_id": str(convo.id),
        "messages": [_msg_out(m).model_dump(mode="json") for m in rows],
        "has_more": has_more,
        "next_after_seq": rows[-1].seq if rows else after_seq,
    }


@router.post("/chat/sessions/{session_id}/messages", response_model=ReplyOut)
async def send_message(
    session_id: str, body: MessageIn, request: Request,
    user: CurrentUser, session: DbSession,
) -> ReplyOut:
    """S14-B4. Send + receive, one AI call, nothing in the background."""
    convo, _ = await _owned_session(session, user.id, session_id)
    if convo.status == "ended":
        raise ApiError(
            409, "chat_ended",
            "This chat has ended. You can still read it, but not add to it.",
        )
    try:
        result = await reply(session, request.app.state.ai_router, convo, body.text)
    except RouteUnresolvedError as exc:
        raise ApiError(
            503, "model_not_chosen",
            "The chat can't run yet — the model for it hasn't been chosen.",
        ) from exc
    except ReplyFailed as exc:
        # S14-B10: explicit, never a degraded plain-text stand-in.
        raise ApiError(
            502, "reply_failed",
            "They couldn't reply just now. Your message is still in the box — "
            "try sending it again.",
        ) from exc

    user_row = (
        await session.execute(
            select(ChatMessage).where(
                ChatMessage.session_id == convo.id, ChatMessage.seq == result.message.seq - 1
            )
        )
    ).scalar_one()
    return ReplyOut(
        user_message=_msg_out(user_row),
        persona_message=_msg_out(result.message),
        compacted=result.compacted,
    )


@router.post("/chat/sessions/{session_id}/end", response_model=SessionOut)
async def end_endpoint(
    session_id: str, user: CurrentUser, session: DbSession
) -> SessionOut:
    """S14-B8/B9. The row moves to `ended`; the history stays readable."""
    convo, match = await _owned_session(session, user.id, session_id)
    await end_session(session, convo)
    return await _session_out(session, convo, match)
