"""Persona and calibration endpoints (S7-B7, B8, B11).

- POST /persona/compile               — start-then-poll; returns immediately
- GET  /persona/current               — status + metadata, NEVER the prompt
- POST /calibration/sessions          — "meet your AI self"
- POST /calibration/sessions/{id}/messages
- POST /calibration/messages/{id}/flag

**The raw system prompt never leaves the server** (trait_persona.md §7.5,
communication_protocol.md §6). It embeds the user's raw intimate answers
verbatim, so there is no response model in this file that carries it, and no
field a future edit can accidentally widen into carrying it — the payloads
below list their fields explicitly rather than dumping the row.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.ai.base import AIError, GenRequest, Message, RouteUnresolvedError
from app.ai.structured import guarded_structured_call
from app.errors import ApiError
from app.logging_setup import log_event
from app.models import CalibrationMessage, CalibrationSession, PersonaSnapshot
from app.persona import compile_persona, get_current_snapshot, is_stale, latest_snapshot
from app.schemas.agent_response import AGENT_RESPONSE_V1
from app.security import CurrentUser, DbSession

router = APIRouter(tags=["persona"])
logger = logging.getLogger("app.persona")

CHAT_TASK = "chat_reply"
CHAT_MAX_TOKENS = 2048

# Per-process, like the extraction give-up and for the same reason (§17): one
# uvicorn worker IS the deployment this phase. Two compilations for one user
# would race to claim the same version number and one would lose on the
# UNIQUE (user_id, version) constraint.
_compiling: set[uuid.UUID] = set()

# asyncio only holds a WEAK reference to a running task. A bare
# `create_task(...)` whose result nobody keeps can be garbage-collected
# mid-flight, and the compilation would vanish with no error anywhere. Holding
# the task until it finishes is not optional bookkeeping.
_tasks: set[asyncio.Task] = set()


class SnapshotOut(BaseModel):
    """Metadata only. There is deliberately no `system_prompt` field here."""

    snapshot_id: str
    version: int
    status: str
    schema_version: str
    traits_hash: str
    source_trait_count: int
    digest_model: str | None
    error: str | None
    created_at: datetime
    stale: bool = False

    @classmethod
    def build(cls, s: PersonaSnapshot, *, stale: bool = False) -> SnapshotOut:
        return cls(
            snapshot_id=str(s.id), version=s.version, status=s.status,
            schema_version=s.schema_version, traits_hash=s.traits_hash,
            source_trait_count=len(s.source_trait_ids or []),
            digest_model=s.digest_model, error=s.error,
            created_at=s.created_at, stale=stale,
        )


class CompileOut(BaseModel):
    snapshot_id: str | None
    status: str  # 'compiling' | 'already_compiling'


class CurrentOut(BaseModel):
    snapshot: SnapshotOut | None
    # Explicit rather than inferred from `snapshot`: a user whose newest
    # snapshot is 'failed' still HAS a usable previous one, and the UI needs to
    # know that without reimplementing the rule.
    simulatable: bool


async def _run_compile(app, user_id: uuid.UUID) -> None:
    """The background half of start-then-poll. Owns its own session: the
    request's session is closed the moment the response is sent."""
    factory = async_sessionmaker(app.state.engine, expire_on_commit=False)
    try:
        async with factory() as session:
            await compile_persona(session, app.state.ai_router, user_id)
    except Exception as exc:  # noqa: BLE001
        # A background task that raises has nowhere to report: no request is
        # waiting on it and asyncio swallows the traceback. Blind here is the
        # only way the §7 log line is guaranteed to be written.
        log_event(
            logger, "persona_compile_crashed",
            level=logging.ERROR, user_id=str(user_id),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _compiling.discard(user_id)


def start_compile(app, user_id: uuid.UUID) -> bool:
    """Launch a background compilation unless one is already running for this
    user. Returns False when it declined because one was already in flight.

    The ONE place a compilation is started, so the endpoint (S7-B7) and the
    automatic post-extraction trigger (S7-B6) cannot drift into two different
    ideas of what "already compiling" means (§16).
    """
    if user_id in _compiling:
        return False
    _compiling.add(user_id)
    task = asyncio.create_task(_run_compile(app, user_id))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True


@router.post("/persona/compile", response_model=CompileOut)
async def compile_endpoint(request: Request, user: CurrentUser) -> CompileOut:
    """S7-B7. Returns immediately — compilation takes tens of seconds and a
    request that blocks on it would be a timeout waiting to happen. The client
    polls GET /persona/current."""
    started = start_compile(request.app, user.id)
    return CompileOut(
        snapshot_id=None, status="compiling" if started else "already_compiling"
    )


@router.get("/persona/current", response_model=CurrentOut)
async def current(user: CurrentUser, session: DbSession) -> CurrentOut:
    """S7-B7/B10. Status and metadata. Never the system prompt."""
    newest = await latest_snapshot(session, user.id)
    ready = await get_current_snapshot(session, user.id)
    stale = await is_stale(session, user.id, ready)
    return CurrentOut(
        snapshot=SnapshotOut.build(newest, stale=stale) if newest else None,
        simulatable=ready is not None,
    )


# --- Calibration: "meet your AI self" (S7-B8) -------------------------------


class SessionOut(BaseModel):
    session_id: str
    snapshot_id: str
    version: int


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class MessageOut(BaseModel):
    message_id: str
    seq: int
    sender: str
    text: str
    flagged: bool = False


class FlagIn(BaseModel):
    correction: str | None = None


@router.post("/calibration/sessions", response_model=SessionOut)
async def start_session(user: CurrentUser, session: DbSession) -> SessionOut:
    snapshot = await get_current_snapshot(session, user.id)
    if snapshot is None:
        # The §11 gate, surfaced in layman's terms (§26).
        raise ApiError(
            409, "no_persona_yet",
            "Your AI self isn't built yet. Answer your questions and give it a "
            "moment to read them.",
        )
    row = CalibrationSession(user_id=user.id, snapshot_id=snapshot.id)
    session.add(row)
    await session.commit()
    log_event(
        logger, "calibration_session_started",
        user_id=str(user.id), session_id=str(row.id),
        snapshot_id=str(snapshot.id), version=snapshot.version,
    )
    return SessionOut(
        session_id=str(row.id), snapshot_id=str(snapshot.id), version=snapshot.version
    )


async def _owned_session(session, user_id: uuid.UUID, sid: str) -> CalibrationSession:
    try:
        parsed = uuid.UUID(sid)
    except ValueError as exc:
        raise ApiError(404, "not_found", "That conversation doesn't exist.") from exc
    row = (
        await session.execute(
            select(CalibrationSession).where(
                CalibrationSession.id == parsed, CalibrationSession.user_id == user_id
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise ApiError(404, "not_found", "That conversation doesn't exist.")
    return row


@router.post("/calibration/sessions/{session_id}/messages", response_model=MessageOut)
async def send_message(
    session_id: str, body: MessageIn, request: Request,
    user: CurrentUser, session: DbSession,
) -> MessageOut:
    """One AI call, same `agent_response.v1` every speaking agent obeys.

    The inner-state fields are STORED but not returned: the user is meeting
    their AI self, not reading its telemetry (§6). They surface in Step 13's
    date results, where they are the point.
    """
    convo = await _owned_session(session, user.id, session_id)
    snapshot = (
        await session.execute(
            select(PersonaSnapshot).where(PersonaSnapshot.id == convo.snapshot_id)
        )
    ).scalar_one()

    history = list(
        (
            await session.execute(
                select(CalibrationMessage)
                .where(CalibrationMessage.session_id == convo.id)
                .order_by(CalibrationMessage.seq)
            )
        ).scalars()
    )
    next_seq = (history[-1].seq + 1) if history else 1

    session.add(CalibrationMessage(
        session_id=convo.id, seq=next_seq, sender="user", text_=body.text,
    ))
    await session.flush()

    try:
        provider, model = request.app.state.ai_router.resolve(CHAT_TASK)
    except RouteUnresolvedError as exc:
        raise ApiError(
            503, "model_not_chosen",
            "Your AI self can't talk yet — the model for it hasn't been chosen.",
        ) from exc

    messages = [
        Message(role="user" if m.sender == "user" else "assistant", content=m.text_)
        for m in history
    ]
    messages.append(Message(role="user", content=body.text))

    try:
        result = await guarded_structured_call(
            provider,
            GenRequest(
                task=CHAT_TASK, model=model,
                # The snapshot's prompt IS the persona. It is read here and
                # never returned anywhere.
                system_prompt=snapshot.system_prompt or "",
                messages=messages, temperature=0.8, max_tokens=CHAT_MAX_TOKENS,
            ),
            AGENT_RESPONSE_V1,
        )
    except AIError as exc:
        log_event(
            logger, "calibration_reply_failed",
            level=logging.ERROR, user_id=str(user.id),
            session_id=str(convo.id), error=str(exc),
        )
        raise ApiError(
            502, "reply_failed",
            "Your AI self didn't answer just now. Your message is saved — try "
            "again in a moment.",
        ) from exc

    reply = CalibrationMessage(
        session_id=convo.id, seq=next_seq + 1, sender="persona",
        text_=result["reply"],
    )
    session.add(reply)
    await session.commit()

    log_event(
        logger, "calibration_reply",
        user_id=str(user.id), session_id=str(convo.id),
        snapshot_id=str(snapshot.id), version=snapshot.version,
        provider=provider.name, model=model, seq=reply.seq,
        # Stored state, logged but not returned to the client.
        connection=result.get("connection"), satisfaction=result.get("satisfaction"),
        wants_to_end=result.get("wants_to_end"),
        emotional_state=result.get("emotional_state"),
    )
    return MessageOut(
        message_id=str(reply.id), seq=reply.seq, sender="persona", text=reply.text_,
    )


@router.get("/calibration/sessions/{session_id}/messages")
async def list_messages(
    session_id: str, user: CurrentUser, session: DbSession
) -> dict:
    convo = await _owned_session(session, user.id, session_id)
    rows = (
        await session.execute(
            select(CalibrationMessage)
            .where(CalibrationMessage.session_id == convo.id)
            .order_by(CalibrationMessage.seq)
        )
    ).scalars()
    return {
        "messages": [
            MessageOut(
                message_id=str(m.id), seq=m.seq, sender=m.sender,
                text=m.text_, flagged=m.flagged,
            ).model_dump()
            for m in rows
        ]
    }


@router.post("/calibration/messages/{message_id}/flag", response_model=MessageOut)
async def flag_message(
    message_id: str, body: FlagIn, user: CurrentUser, session: DbSession
) -> MessageOut:
    """"I'd never say that." S7-B8/B9 — the flag feeds the NEXT compilation as
    an explicit negative example, which is the only way calibration changes
    anything."""
    try:
        parsed = uuid.UUID(message_id)
    except ValueError as exc:
        raise ApiError(404, "not_found", "That message doesn't exist.") from exc

    row = (
        await session.execute(
            select(CalibrationMessage)
            .join(
                CalibrationSession,
                CalibrationSession.id == CalibrationMessage.session_id,
            )
            .where(CalibrationMessage.id == parsed, CalibrationSession.user_id == user.id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise ApiError(404, "not_found", "That message doesn't exist.")
    if row.sender != "persona":
        raise ApiError(
            409, "not_flaggable",
            "You can only flag things your AI self said, not your own messages.",
        )

    convo = (
        await session.execute(
            select(CalibrationSession).where(CalibrationSession.id == row.session_id)
        )
    ).scalar_one()

    row.flagged = True
    row.correction = (body.correction or "").strip() or None
    await session.commit()

    # §7: a flag logs WHICH snapshot it criticises — otherwise a complaint
    # about v2 looks like a complaint about v5 six weeks later.
    log_event(
        logger, "calibration_flagged",
        user_id=str(user.id), message_id=str(row.id),
        session_id=str(convo.id), snapshot_id=str(convo.snapshot_id),
        had_correction=bool(row.correction),
    )
    return MessageOut(
        message_id=str(row.id), seq=row.seq, sender=row.sender,
        text=row.text_, flagged=True,
    )


@router.get("/calibration/flags/count")
async def flag_count(user: CurrentUser, session: DbSession) -> dict:
    """What the next compilation will treat as negative examples. Small, but it
    is the only way to see that flagging did anything before a recompile."""
    n = (
        await session.execute(
            select(func.count())
            .select_from(CalibrationMessage)
            .join(
                CalibrationSession,
                CalibrationSession.id == CalibrationMessage.session_id,
            )
            .where(
                CalibrationSession.user_id == user.id,
                CalibrationMessage.flagged.is_(True),
            )
        )
    ).scalar_one()
    return {"flagged": int(n)}
