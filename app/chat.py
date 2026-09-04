"""The chat module (S14-B2…B7, B9, B10): selection, the prompt extension, the
rolling window with compaction, and one reply.

**The person the user talks to is the one whose dates they read** (S14-B2).
Selection pins the candidate's MATCHED snapshot — `analysis_candidates.
snapshot_id`, the same row the dates ran against — never "their current
persona". A candidate who has answered ten more questions since has a newer
self, and it is not the self the user chose.

**The simulated history is framed as a simulation, verbatim, every call**
(S14-B6). The persona is told the dates were simulations the human was NOT
present for, to call them "our simulated date", and never to invent a detail
beyond the digest. This is the anti-gaslighting rule from
`user_perspective.md` and it is CONTRACT, not tone guidance: a persona that
says "remember when the Mustang pulled up?" to someone who was never there is
the one failure this module exists to prevent. The text lives in one constant
below so it can be grepped, tested, and never paraphrased at a call site.

**Compaction is arithmetic on `seq`** (S14-B7). The live window is every
message with `seq > compacted_upto_seq`; when it exceeds [WINDOW] the oldest
[FOLD] of it are folded into `summary` and `compacted_upto_seq` moves by
[FOLD]. Pure functions decide WHEN and WHAT; the AI call only writes prose.
That is what lets the boundary be unit-tested without a model.

**A give-up is an error, never a degraded reply** (S14-B10). If the Guard
cannot get a valid `agent_response.v1`, the user's message is NOT persisted
(the transaction rolls back), the raw output is logged, and the client gets a
502 that says "couldn't reply, try again". The text stays in the composer on
the client — user text is never dropped, and a retry does not double-post.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.base import AIError, GenRequest, Message, StructuredOutputError
from app.ai.routing import TaskRouter
from app.ai.structured import guarded_structured_call
from app.judging import date_digest
from app.logging_setup import log_event
from app.models import (
    Analysis,
    AnalysisCandidate,
    ChatMessage,
    ChatSession,
    PersonaSnapshot,
    User,
)
from app.schemas.agent_response import AGENT_RESPONSE_V1, SCHEMA_VERSION
from app.schemas.chat_compaction import CHAT_COMPACTION_V1, COMPACTION_VERSION

logger = logging.getLogger("app.chat")

REPLY_TASK = "chat_reply"
COMPACTION_TASK = "chat_compaction"
REPLY_MAX_TOKENS = 2048
REPLY_TEMPERATURE = 0.8
COMPACTION_MAX_TOKENS = 1024
COMPACTION_TEMPERATURE = 0.2

# chat.md §2, locked #4: last 40 verbatim + a running summary, folded 20 at a
# time. Both integers; both meant to be turned together.
WINDOW = 40
FOLD = 20

# S14-B6, part 2 — verbatim contract. `{name}` is the human's display name and
# `{digest}` is `DateDigest`'s output; nothing else is interpolated.
SIMULATED_HISTORY_RULES = """\
ABOUT THE PERSON YOU ARE TALKING TO
You are now chatting with {name}, a real person typing to you. Before this \
chat, an AI version of you and an AI version of {name} went on a simulated \
date. {name} was NOT there. It was a simulation they read afterwards, like \
reading a transcript. Here is everything that is known about it:

{digest}

RULES ABOUT THAT HISTORY (these are strict):
- Call it "our simulated date". Never speak of it as something you both \
lived through, and never say "remember when" about it.
- Do not invent any detail about it beyond what is written above. If asked \
about something that is not written there, say you don't know more than that.
- {name} may mention it, or may not. Do not bring it up unprompted more than \
once.
- Everything from here on is real conversation with a real person. Be \
yourself."""

SUMMARY_BLOCK = """\

EARLIER IN THIS CHAT (summarised, because the chat is long):
{summary}"""


# --- Pure rules ---------------------------------------------------------------


def fold_plan(total_seq: int, compacted_upto: int) -> tuple[int, int] | None:
    """Which seqs to fold next, or None when the window still fits.

    The live window is `seq > compacted_upto`. It overflows when it holds
    more than WINDOW messages; the fold takes the oldest FOLD of them. Called
    BEFORE the new user message is written, so `total_seq` is the last stored
    seq (S14-B7: "runs inline before the reply call when needed").
    """
    live = total_seq - compacted_upto
    if live <= WINDOW:
        return None
    return (compacted_upto + 1, compacted_upto + FOLD)


def selection_refusal(
    status: str, is_candidate: bool, already_selected: bool
) -> tuple[int, str, str] | None:
    """Why `POST /select` says no, as (http status, code, message) — or None.

    409s are STATE, not failure (communication_protocol.md §5): the UI
    pre-empts them where it can and renders them as state. Order matters —
    "already selected" beats "not complete" so a finished-then-retried
    analysis reports the more useful fact.
    """
    if already_selected:
        return (409, "already_selected", "You've already chosen someone from this analysis.")
    if status != "complete":
        return (
            409, "analysis_not_complete",
            "The dates haven't finished yet — choose someone once the results are in.",
        )
    if not is_candidate:
        return (404, "not_a_candidate", "That person wasn't part of this analysis.")
    return None


def extend_system_prompt(
    persona_prompt: str, *, user_name: str, digest: str, summary: str | None
) -> str:
    """The persona snapshot's prompt, plus the two Step 14 parts (S14-B6)."""
    text = persona_prompt.rstrip() + "\n\n" + SIMULATED_HISTORY_RULES.format(
        name=user_name, digest=digest.strip() or "Nothing was recorded about it."
    )
    if summary:
        text += SUMMARY_BLOCK.format(summary=summary.strip())
    return text


def window_messages(rows: list[ChatMessage], compacted_upto: int) -> list[Message]:
    """The verbatim tail the model sees: everything after the fold point."""
    return [
        Message(role="user" if m.sender == "user" else "assistant", content=m.text_)
        for m in rows
        if m.seq > compacted_upto
    ]


def render_for_summary(rows: list[ChatMessage], *, user_name: str, match_name: str) -> str:
    return "\n".join(
        f"{user_name if m.sender == 'user' else match_name}: {m.text_}" for m in rows
    )


# --- Selection (S14-B2, B3) ---------------------------------------------------


async def existing_session(
    session: AsyncSession, analysis_id: uuid.UUID
) -> ChatSession | None:
    return (
        await session.execute(
            select(ChatSession).where(ChatSession.analysis_id == analysis_id)
        )
    ).scalar_one_or_none()


async def select_match(
    session: AsyncSession, analysis: Analysis, candidate: AnalysisCandidate
) -> ChatSession:
    """Create the session, pin the matched snapshot, compile the digest once
    (no AI call — `DateDigest` has nothing to make one with)."""
    digest = await date_digest(session, analysis.id, candidate.candidate_user_id)
    row = ChatSession(
        user_id=analysis.user_id,
        match_user_id=candidate.candidate_user_id,
        analysis_id=analysis.id,
        snapshot_id=candidate.snapshot_id,
        date_digest=digest,
        status="active",
    )
    session.add(row)
    await session.commit()
    log_event(
        logger, "chat_session_created",
        user_id=str(analysis.user_id), session_id=str(row.id),
        analysis_id=str(analysis.id), match_user_id=str(candidate.candidate_user_id),
        snapshot_id=str(candidate.snapshot_id), digest_chars=len(digest),
    )
    return row


# --- The reply (S14-B4…B7, B10, B11) -----------------------------------------


@dataclass(frozen=True)
class ChatReply:
    message: ChatMessage
    compacted: bool


class ReplyFailed(Exception):
    """The Guard gave up (or the provider did). The user's message was NOT
    persisted; the client keeps it in the composer."""


async def _load_messages(session: AsyncSession, session_id: uuid.UUID) -> list[ChatMessage]:
    return list(
        (
            await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.seq)
            )
        ).scalars()
    )


async def compact_if_needed(
    session: AsyncSession, router: TaskRouter, convo: ChatSession,
    rows: list[ChatMessage], *, user_name: str, match_name: str,
) -> bool:
    """S14-B7. One structured call folds the oldest FOLD live messages into
    the summary. Committed on its own BEFORE the reply call, so a reply that
    then fails does not undo a compaction that succeeded (§19: checkpoint
    before advancing)."""
    total_seq = rows[-1].seq if rows else 0
    plan = fold_plan(total_seq, convo.compacted_upto_seq)
    if plan is None:
        return False
    lo, hi = plan
    folding = [m for m in rows if lo <= m.seq <= hi]

    provider, model = router.resolve(COMPACTION_TASK)
    previous = convo.summary or "(nothing yet — this is the first summary)"
    result = await guarded_structured_call(
        provider,
        GenRequest(
            task=COMPACTION_TASK, model=model,
            system_prompt=(
                "You maintain a running summary of a chat between two people. "
                "You will be given the previous summary and the next stretch of "
                "messages. Write the new summary that replaces the previous one."
            ),
            messages=[Message(
                role="user",
                content=(
                    f"PREVIOUS SUMMARY:\n{previous}\n\n"
                    f"NEXT MESSAGES ({lo}-{hi}):\n"
                    + render_for_summary(folding, user_name=user_name, match_name=match_name)
                ),
            )],
            temperature=COMPACTION_TEMPERATURE, max_tokens=COMPACTION_MAX_TOKENS,
        ),
        CHAT_COMPACTION_V1,
    )
    convo.summary = result["summary"]
    convo.compacted_upto_seq = hi
    await session.commit()
    log_event(
        logger, "chat_compacted",
        session_id=str(convo.id), folded_from_seq=lo, folded_to_seq=hi,
        compacted_upto_seq=hi, summary_chars=len(convo.summary),
        live_window=total_seq - hi, provider=provider.name, model=model,
        schema=COMPACTION_VERSION,
    )
    return True


async def reply(
    session: AsyncSession, router: TaskRouter, convo: ChatSession, text: str,
    client_message_id: str | None = None,
) -> ChatReply:
    """Persist the user message, one Guard call, persist and return the
    persona's line. Plain request–response — seconds, no job, no polling."""
    me = (await session.execute(select(User).where(User.id == convo.user_id))).scalar_one()
    them = (
        await session.execute(select(User).where(User.id == convo.match_user_id))
    ).scalar_one()
    snapshot = (
        await session.execute(
            select(PersonaSnapshot).where(PersonaSnapshot.id == convo.snapshot_id)
        )
    ).scalar_one()

    # One send at a time per session (audit 2026-09-02). Two concurrent sends
    # computed the same next_seq; the second then WAITED the whole first
    # reply on UNIQUE(session_id, seq) and failed with IntegrityError → 500.
    # The row lock makes the second read the first's two rows and take N+3.
    await session.execute(
        select(ChatSession).where(ChatSession.id == convo.id).with_for_update()
    )
    rows = await _load_messages(session, convo.id)
    try:
        compacted = await compact_if_needed(
            session, router, convo, rows,
            user_name=me.display_name, match_name=them.display_name,
        )
    except AIError as exc:
        # A fold that fails must not take the reply with it: it used to
        # escape as a raw 500 on EVERY send once history passed the fold
        # point, which bricked the chat (audit 2026-09-02). History is
        # intact (nothing is written before the fold commits) and nothing is
        # pending, so NO rollback: one would expire `convo`, `me`, `them` and
        # `snapshot`, and the next attribute read would be MissingGreenlet
        # (review 2026-09-03). Reply on the unfolded window; fold next time.
        rows = await _load_messages(session, convo.id)
        compacted = False
        log_event(
            logger, "chat_compaction_failed", level=logging.ERROR,
            session_id=str(convo.id), user_id=str(convo.user_id),
            live_rows=len(rows), error=str(exc)[:500],
        )

    # A fold that ran COMMITTED, which released the row lock above; take it
    # again so the seq below is still computed under it.
    await session.execute(
        select(ChatSession).where(ChatSession.id == convo.id).with_for_update()
    )
    next_seq = (rows[-1].seq + 1) if rows else 1
    user_row = ChatMessage(
        session_id=convo.id, seq=next_seq, sender="user", text_=text,
        client_message_id=client_message_id,
    )
    session.add(user_row)
    await session.flush()

    provider, model = router.resolve(REPLY_TASK)
    system_prompt = extend_system_prompt(
        snapshot.system_prompt or "",
        user_name=me.display_name, digest=convo.date_digest, summary=convo.summary,
    )
    messages = window_messages(rows, convo.compacted_upto_seq)
    messages.append(Message(role="user", content=text))

    # Captured BEFORE the call: a rollback expires every loaded attribute, and
    # reading one afterwards from async code raises MissingGreenlet — which is
    # how the give-up path 500ed on its first forced run (D-014).
    uid, sid = str(convo.user_id), str(convo.id)
    try:
        result = await guarded_structured_call(
            provider,
            GenRequest(
                task=REPLY_TASK, model=model, system_prompt=system_prompt,
                messages=messages, temperature=REPLY_TEMPERATURE,
                max_tokens=REPLY_MAX_TOKENS,
            ),
            AGENT_RESPONSE_V1,
        )
    except AIError as exc:
        # S14-B10: the user's row goes with the rollback. A retry re-sends it.
        await session.rollback()
        log_event(
            logger, "chat_reply_failed", level=logging.ERROR,
            user_id=uid, session_id=sid, seq=next_seq,
            provider=provider.name, model=model, outcome="gave_up",
            error=str(exc)[:500],
            raw_output=(
                exc.raw_output[:2000] if isinstance(exc, StructuredOutputError) else None
            ),
        )
        raise ReplyFailed(str(exc)) from exc

    if not str(result.get("reply") or "").strip():
        # The schema has no minLength on `reply`; a blank reply would be a
        # blank bubble. Same path as any other failed reply.
        await session.rollback()
        log_event(
            logger, "chat_reply_failed", level=logging.ERROR,
            user_id=uid, session_id=sid, seq=next_seq,
            provider=provider.name, model=model, outcome="empty_reply",
        )
        raise ReplyFailed("the persona returned an empty reply")
    persona_row = ChatMessage(
        session_id=convo.id, seq=next_seq + 1, sender="persona",
        text_=result["reply"],
        # Stored, never returned (S14-B5).
        state={
            k: result.get(k)
            for k in ("state_of_mind", "emotional_state", "connection",
                      "satisfaction", "wants_to_end")
        },
        provider=provider.name, model_id=model,
    )
    session.add(persona_row)
    await session.commit()
    log_event(
        logger, "chat_reply",
        user_id=str(convo.user_id), session_id=str(convo.id),
        seq=persona_row.seq, provider=provider.name, model=model,
        attempt=1, outcome="ok", schema=SCHEMA_VERSION,
        window=len(messages), compacted_before_reply=compacted,
        connection=result.get("connection"), satisfaction=result.get("satisfaction"),
        wants_to_end=result.get("wants_to_end"),
    )
    return ChatReply(message=persona_row, compacted=compacted)


async def end_session(session: AsyncSession, convo: ChatSession) -> None:
    if convo.status != "ended":
        convo.status = "ended"
        convo.ended_at = datetime.now(UTC)
        await session.commit()
    log_event(logger, "chat_session_ended", user_id=str(convo.user_id), session_id=str(convo.id))
