"""The judge pipeline and match scores (S12-B3…B11).

**The number is computed here, in code, and never asked from the model**
(trade #3, S12-B5). The judge produces four 0-100 criteria; `date_score` and
`candidate_score` are arithmetic over them. The weights are opinions — that is
admitted — but they are VISIBLE opinions, versioned, and identical for
everyone. The alternative, a model picking a headline number, is an opinion
too, and an invisible one that changes between calls.

Both halves are stored: the raw `criteria` the model returned and the score the
code derived. That is what makes `probe_judge.py` able to recompute a stored
score by hand and check it, and what would let a future weights change be
applied to old evaluations without re-running a single AI call.

**The judge sees the transcript and both people's trait LABELS — never their
personas, descriptions, or answers** (trade #5). It scores what happened on the
date, not what the profiles predicted. That separation is the entire reason a
surprising result is informative rather than a bug, and it is enforced here by
building the prompt from labels only, in one function, rather than by everyone
remembering.

**The incomplete-date policy is one function with the boundary written down**
(S12-B7, §14 names it as an accretion risk): at least 10 AGENT TURNS and an
incomplete date is judged and flagged partial; under 10 it is excluded from
scoring entirely and shown as failed. Ten turns is roughly five each — below
that there is not a date to judge, there is an opening.

**Turns, not rows** (revised 2026-09-01). Environment rows are scenery: nobody
said anything, and letting them pad the count meant a date with 7 turns and 3
events was scored while one with 9 turns and no events was thrown away. The
threshold reads the same `turn_count` the date cap does, so the two cannot
drift apart again.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.base import GenRequest, Message
from app.ai.routing import TaskRouter
from app.ai.structured import guarded_structured_call
from app.logging_setup import log_event
from app.models import (
    CandidateScore,
    DateEvaluation,
    DateMessage,
    SimulatedDate,
    Trait,
    User,
)
from app.schemas.judge_rubric import (
    JUDGE_RUBRIC_V1,
    JUDGE_SYSTEM_PROMPT,
    RUBRIC_VERSION,
)
from app.simulation import JUDGEABLE_MIN_TURNS, to_views, turn_count

logger = logging.getLogger("app.judging")

TASK = "judging"
# Generous, because the judge writes two per-peer paragraphs and a verdict on
# top of its numbers — and this model has been observed burning a token budget
# on whitespace before producing anything (D-010's neighbour, seen in Step 11).
JUDGE_MAX_TOKENS = 4096
# Low, deliberately (S12-B3). The judge is a measuring instrument: the same
# transcript should score about the same twice, and AC1 asks a probe to prove
# it. Temperature is the one knob that makes that impossible.
JUDGE_TEMPERATURE = 0.1

# S12-B5, verbatim from date_simulation.md. These four numbers are the whole
# scoring opinion of this product and they are in one place on purpose.
WEIGHTS = {
    "trait_alignment": 0.30,
    "conversational_flow": 0.30,
    "mutual_engagement": 0.25,
    "clash_severity": 0.15,
}
# An incomplete-but-judged date counts half (date_simulation.md, locked #3).
PARTIAL_WEIGHT = 0.5


# --- The arithmetic, all of it pure ----------------------------------------


def date_score(criteria: dict) -> float:
    """S12-B5, verbatim:

        0.30*trait_alignment + 0.30*conversational_flow
      + 0.25*mutual_engagement + 0.15*(100 - clash_severity)

    `clash_severity` is INVERTED here and nowhere else. It is the only
    criterion where high is bad, so it is the only one subtracted from 100 —
    and doing that inversion at the single point where the score is computed
    is what stops it being done twice, or not at all, somewhere downstream.
    """
    return (
        WEIGHTS["trait_alignment"] * criteria["trait_alignment"]
        + WEIGHTS["conversational_flow"] * criteria["conversational_flow"]
        + WEIGHTS["mutual_engagement"] * criteria["mutual_engagement"]
        + WEIGHTS["clash_severity"] * (100 - criteria["clash_severity"])
    )


def is_judgeable(status: str, turns: int) -> bool:
    """S12-B7, and the boundary is the point (§18).

    A `complete` date is always judged. An `incomplete` one is judged only if
    the pair took at least 10 TURNS between them — roughly five each. Below
    that there is no date to score, only an opening, and a number derived from
    it would look exactly like a number derived from a real evening.

    `turns` counts what the agents SAID. It is not the row count: an
    environment row is the world doing something, not a person talking, and
    counting it here would let three lucky dice rolls promote a seven-turn
    fragment over a nine-turn one.
    """
    if status == "complete":
        return True
    if status == "incomplete":
        return turns >= JUDGEABLE_MIN_TURNS
    return False


def candidate_score(scored: list[tuple[float, bool]]) -> float | None:
    """S12-B6. Weighted mean of the date scores, where an
    incomplete-but-judged date carries half the weight of a complete one.

    `scored` is [(date_score, is_partial), ...]. Returns None when there is
    nothing to average — which is a real state (every date failed early) and
    must not be reported as 0.0, because 0.0 is a SCORE and means "they were
    terrible together".
    """
    if not scored:
        return None
    total = sum(s * (PARTIAL_WEIGHT if partial else 1.0) for s, partial in scored)
    weight = sum(PARTIAL_WEIGHT if partial else 1.0 for _, partial in scored)
    return total / weight


# --- What the judge is allowed to see (trade #5) ---------------------------


async def _trait_labels(session: AsyncSession, user_id: uuid.UUID) -> list[str]:
    """LABELS only. Descriptions are written about someone by an AI reading
    their intimate answers; the judge has no use for them and every extra field
    here is one the judge could start scoring people on."""
    return list(
        (
            await session.execute(
                select(Trait.label)
                .where(Trait.user_id == user_id, Trait.status != "retracted")
                .order_by(Trait.category, Trait.label)
            )
        ).scalars()
    )


def render_transcript(
    messages: list[DateMessage], user_name: str, candidate_name: str
) -> str:
    """The transcript as the judge reads it.

    Environment rows are marked as the world rather than as a speaker — a judge
    that reads "the power cuts out" as something a person SAID will score the
    conversational flow of a light fitting.
    """
    lines: list[str] = []
    for m in messages:
        if m.speaker == "environment":
            lines.append(f"[{m.seq}] (something happens around them: {m.reply})")
        else:
            who = user_name if m.speaker == "user_agent" else candidate_name
            lines.append(f"[{m.seq}] {who}: {m.reply}")
    return "\n".join(lines)


def build_judge_request(
    *,
    transcript: str,
    setting_name: str,
    user_name: str,
    candidate_name: str,
    user_labels: list[str],
    candidate_labels: list[str],
    partial: bool,
) -> str:
    body = (
        f"A date between {user_name} and {candidate_name}, at {setting_name}.\n\n"
        f"{user_name} is described as: "
        + (", ".join(user_labels) or "(no traits recorded)")
        + f"\n{candidate_name} is described as: "
        + (", ".join(candidate_labels) or "(no traits recorded)")
    )
    if partial:
        # Said out loud, because a judge that does not know the transcript was
        # cut off will score the missing ending as a bad ending.
        body += (
            "\n\nNOTE: this date was CUT SHORT by a technical failure, not by "
            "either person. It stops mid-conversation. Judge what is here and "
            "do not treat the abrupt stop as something either of them did."
        )
    return (
        body
        + "\n\nTHE TRANSCRIPT\n"
        + transcript
        + "\n\nScore this date against the four criteria and report what you found."
    )


# --- Judging one date ------------------------------------------------------


@dataclass
class JudgeOutcome:
    date_id: uuid.UUID
    candidate_user_id: uuid.UUID
    score: float
    is_partial: bool


async def judge_date(
    session: AsyncSession,
    router: TaskRouter,
    date: SimulatedDate,
    owner_id: uuid.UUID,
) -> JudgeOutcome | None:
    """One structured call, then the score in code (S12-B3…B5, B9).

    Returns None when the date is not judgeable, or is already judged. Being
    already judged is the ordinary case on a relaunched pipeline, not an error:
    the whole judge pass is idempotent because `date_evaluations.date_id` is a
    primary key and this check reads it.
    """
    existing = (
        await session.execute(
            select(DateEvaluation).where(DateEvaluation.date_id == date.id)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return JudgeOutcome(
            date_id=date.id, candidate_user_id=date.candidate_user_id,
            score=float(existing.date_score), is_partial=existing.is_partial,
        )

    messages = list(
        (
            await session.execute(
                select(DateMessage)
                .where(DateMessage.date_id == date.id)
                .order_by(DateMessage.seq)
            )
        ).scalars()
    )
    turns = turn_count(to_views(messages))
    if not is_judgeable(date.status, turns):
        # Logged rather than skipped silently: "this date was excluded from the
        # score" is a fact the results screen has to be able to state (AC4).
        log_event(
            logger, "date_not_judged", level=logging.WARNING,
            date_id=str(date.id), analysis_id=str(date.analysis_id),
            status=date.status, messages=len(messages), turns=turns,
            threshold=JUDGEABLE_MIN_TURNS, counted="agent turns, not rows",
            reason="too few turns to be a date"
            if date.status == "incomplete"
            else f"status is {date.status}",
        )
        return None

    owner = (
        await session.execute(select(User).where(User.id == owner_id))
    ).scalar_one()
    candidate = (
        await session.execute(select(User).where(User.id == date.candidate_user_id))
    ).scalar_one()

    partial = date.status == "incomplete"
    provider, model = router.resolve(TASK)
    result = await guarded_structured_call(
        provider,
        GenRequest(
            task=TASK, model=model, system_prompt=JUDGE_SYSTEM_PROMPT,
            messages=[
                Message(
                    role="user",
                    content=build_judge_request(
                        transcript=render_transcript(
                            messages, owner.display_name,
                            candidate.display_name,
                        ),
                        setting_name=date.scenario.get("setting_name", "a first date"),
                        user_name=owner.display_name,
                        candidate_name=candidate.display_name,
                        user_labels=await _trait_labels(session, owner.id),
                        candidate_labels=await _trait_labels(session, candidate.id),
                        partial=partial,
                    ),
                )
            ],
            temperature=JUDGE_TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS,
        ),
        JUDGE_RUBRIC_V1,
    )

    criteria = {k: result[k] for k in WEIGHTS}
    score = date_score(criteria)
    session.add(DateEvaluation(
        date_id=date.id, criteria=criteria,
        clicked=result["clicked_subjects"], clashes=result["clashes"],
        per_peer=result["per_peer_summary"], verdict=result["verdict_summary"],
        date_score=score, is_partial=partial,
        judge_provider=provider.name, judge_model=model,
        rubric_version=RUBRIC_VERSION,
    ))
    await session.commit()

    log_event(
        logger, "date_judged", date_id=str(date.id),
        analysis_id=str(date.analysis_id), provider=provider.name, model=model,
        rubric_version=RUBRIC_VERSION, is_partial=partial,
        messages=len(messages), turns=turns,
        # The raw criteria AND the derived score, so the arithmetic is
        # checkable straight from the log without opening the database (§7).
        criteria=criteria, date_score=round(score, 2),
        clashes=len(result["clashes"]), clicked=len(result["clicked_subjects"]),
    )
    return JudgeOutcome(
        date_id=date.id, candidate_user_id=date.candidate_user_id,
        score=score, is_partial=partial,
    )


# --- Judging a whole analysis ----------------------------------------------


async def judge_analysis(
    session: AsyncSession, router: TaskRouter, analysis_id: uuid.UUID,
    owner_id: uuid.UUID,
) -> dict[uuid.UUID, float]:
    """Judge every judgeable date, then write one `candidate_scores` row per
    candidate (S12-B6, B10).

    Idempotent: already-judged dates return their stored score without a call,
    and the score rows are recomputed from whatever is now judged.
    """
    dates = list(
        (
            await session.execute(
                select(SimulatedDate)
                .where(SimulatedDate.analysis_id == analysis_id)
                .order_by(SimulatedDate.created_at, SimulatedDate.ordinal)
            )
        ).scalars()
    )

    by_candidate: dict[uuid.UUID, list[tuple[float, bool]]] = {}
    excluded: dict[uuid.UUID, int] = {}
    for date in dates:
        outcome = await judge_date(session, router, date, owner_id)
        if outcome is None:
            excluded[date.candidate_user_id] = (
                excluded.get(date.candidate_user_id, 0) + 1
            )
            continue
        by_candidate.setdefault(date.candidate_user_id, []).append(
            (outcome.score, outcome.is_partial)
        )

    finals: dict[uuid.UUID, float] = {}
    for candidate_id in {d.candidate_user_id for d in dates}:
        scored = by_candidate.get(candidate_id, [])
        final = candidate_score(scored)
        if final is None:
            # Every date with this person failed too early to judge. There is
            # no score, and writing 0.0 would say "they were terrible together"
            # about an evening that never happened (§10, §11).
            log_event(
                logger, "candidate_not_scored", level=logging.WARNING,
                analysis_id=str(analysis_id), candidate_user_id=str(candidate_id),
                dates_excluded=excluded.get(candidate_id, 0),
                reason="no date with this candidate produced a judgeable transcript",
            )
            continue

        row = (
            await session.execute(
                select(CandidateScore).where(
                    CandidateScore.analysis_id == analysis_id,
                    CandidateScore.candidate_user_id == candidate_id,
                )
            )
        ).scalar_one_or_none()
        completed = sum(1 for _, partial in scored if not partial)
        incomplete = sum(1 for _, partial in scored if partial)
        if row is None:
            session.add(CandidateScore(
                analysis_id=analysis_id, candidate_user_id=candidate_id,
                final_score=final, dates_completed=completed,
                dates_incomplete=incomplete,
            ))
        else:
            row.final_score = final
            row.dates_completed = completed
            row.dates_incomplete = incomplete
        finals[candidate_id] = final

        log_event(
            logger, "candidate_scored",
            analysis_id=str(analysis_id), candidate_user_id=str(candidate_id),
            final_score=round(final, 2),
            # Every input to the mean, so the weighting is checkable by hand.
            date_scores=[round(s, 2) for s, _ in scored],
            partial_flags=[p for _, p in scored],
            partial_weight=PARTIAL_WEIGHT,
            dates_completed=completed, dates_incomplete=incomplete,
            dates_excluded=excluded.get(candidate_id, 0),
        )

    await session.commit()
    return finals


# --- DateDigest, for the Chat module (S12-B11) -----------------------------


async def date_digest(
    session: AsyncSession, analysis_id: uuid.UUID, candidate_user_id: uuid.UUID
) -> str:
    """A compact factual summary of what happened on their dates — compiled
    from the stored evaluations, with **no new AI call** (S12-B11).

    Chat consumes only this and never reads raw `date_messages`. That boundary
    is the point: a chat agent handed thirty messages of a simulated date will
    quote them back as if they were memories the human shares, and the human
    was never there.
    """
    rows = (
        await session.execute(
            select(SimulatedDate, DateEvaluation)
            .join(DateEvaluation, DateEvaluation.date_id == SimulatedDate.id)
            .where(
                SimulatedDate.analysis_id == analysis_id,
                SimulatedDate.candidate_user_id == candidate_user_id,
            )
            .order_by(SimulatedDate.ordinal)
        )
    ).all()
    if not rows:
        return "You haven't been on a simulated date with them yet."

    parts: list[str] = []
    for date, evaluation in rows:
        setting = date.scenario.get("setting_name", "somewhere")
        clicked = ", ".join(evaluation.clicked or [])
        bit = f"At {setting}: {evaluation.verdict}"
        if clicked:
            bit += f" You got on best about {clicked}."
        if evaluation.is_partial:
            bit += " (That one was cut short.)"
        parts.append(bit)
    return " ".join(parts)
