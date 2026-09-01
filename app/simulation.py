"""Date simulation: scenarios, the turn loop, events, checkpoints, resume
(S11-B2…B8, B10, B12).

Four things in here are load-bearing and easy to quietly break.

**The transcript IS the state.** There is no in-memory progress object that a
restart could lose. Whose turn it is, how many events have fired, whether the
pair have started saying goodbye, and where a killed date should pick up are
all COMPUTED from the stored rows, every time, by the pure functions at the top
of this file. That is why `probe_simulation_resume.py` can kill the server
mid-date and watch the same date continue rather than restart — and why every
one of those rules is unit-testable without a database.

**The message row is written BEFORE the turn advances** (§19). The loop's
"advance" is literally its next iteration, and the row is committed before it.
Reordering those two lines would turn a crash into a lost turn, which is the
one failure this module was designed to make impossible.

**One structured call per turn, and the retries are the resilience layer's**
(§16, §17). The three attempts with backoff live in `app/ai/resilience.py` and
the three validation repairs live in the Guard; this file adds NO fourth loop.
A turn that still fails after those marks the DATE `incomplete` at its last
checkpointed message and the pipeline moves on — the analysis never dies
because one date did.

**Events come from the scenario, not from a generator** (trade #4). The roll,
the rule that blocked it, and the chosen event are all logged on EVERY turn,
including the ones that did not fire — a probability you cannot see is a
probability you cannot check (§8).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.base import AIError, GenRequest, Message
from app.ai.routing import TaskRouter
from app.ai.structured import guarded_structured_call
from app.logging_setup import log_event
from app.matching import MAX_CANDIDATES
from app.models import (
    Analysis,
    AnalysisCandidate,
    DateMessage,
    PersonaSnapshot,
    SimulatedDate,
    Trait,
    User,
)
from app.persona import get_current_snapshot
from app.schemas.agent_response import AGENT_RESPONSE_V1, SCHEMA_VERSION
from app.schemas.date_scenarios import DATE_SCENARIOS_V2, SETTINGS_PER_CANDIDATE
from app.users import compute_age

logger = logging.getLogger("app.simulation")

SCENARIO_TASK = "scenario_generation"
DATE_TASK = "date_simulation"

# --- The caps, exactly as date_simulation.md locks them (S11-B5) ------------
#
# REVISED 2026-09-01 (owner decision): the cap is now on TURNS, not messages,
# and it is 16.
#
# This reverses a rule that development_principles.md §18 had held up as a
# standing example — "the 30-message date cap counts environment events as
# messages". It counted them because the cap was about transcript LENGTH. The
# owner now specifies the budget per candidate as calls: 1 scenario + 16 turns
# + 1 judge = 18. An environment row costs no call, so a cap that let events
# eat into the budget would make the actual spend 13-16 turns depending on dice
# — which is not a budget, it is a distribution.
#
# So: 16 agent turns, 8 each. Events still fire, still cap at 3, and are simply
# not charged against the turn count. The transcript is therefore at most
# TURN_CAP + MAX_EVENTS_PER_DATE rows.
TURN_CAP = 16
# REVISED 2026-09-01 (owner decision): ONE date per candidate, not two. A, B
# and C get one evening apiece.
#
# Derived from the schema's `SETTINGS_PER_CANDIDATE` rather than written out
# again: one generated setting IS one date, and two constants that must agree
# are two constants that will eventually disagree.
#
# The cost of the change, named because it is real: with two dates each, a
# candidate's score was the mean of two independent readings, and one strange
# evening or one wobbly judge call got averaged down. With one date it does
# not — a single date now fully determines a candidate's score. Cheaper (a
# full pool drops from ~177 model calls to ~90) and faster, at the price of a
# noisier number.
DATES_PER_CANDIDATE = SETTINGS_PER_CANDIDATE
# DERIVED, 2026-09-01, and it used to be the literal 3. The old comment said a
# cap that only holds because another module is small is not a cap — true, but
# a cap written as a literal beside a knob is worse: raising
# SETTINGS_PER_CANDIDATE to 2 would have produced 3 candidates wanting 6 dates
# against a ceiling of 3, and `ensure_dates` would have stopped after the
# second candidate and given the third NONE. Silently, with a log line nobody
# was looking for.
#
# Now the ceiling follows the knob. `MAX_CANDIDATES` is matching's own cap on
# pool size, imported rather than repeated for exactly the reason above.
MAX_DATES_PER_ANALYSIS = MAX_CANDIDATES * DATES_PER_CANDIDATE
EVENT_PROBABILITY = 0.15
MAX_EVENTS_PER_DATE = 3
# Derived, never set by hand: the longest a transcript can get. Every turn plus
# every event that could fire alongside them.
MAX_MESSAGES_PER_DATE = TURN_CAP + MAX_EVENTS_PER_DATE
# One closing exchange = one line each, once both of them want to wrap up.
CLOSING_TURNS = 2
# An `incomplete` date with this many AGENT TURNS is still worth judging
# (date_simulation.md, locked #3). Step 12 owns the policy; the constant lives
# here with the loop that produces the incomplete dates.
#
# REVISED 2026-09-01 (owner decision): the threshold counts TURNS, not rows.
# It used to count rows, and that was a leftover from the days when the date
# cap counted rows too. Once the cap moved to turns, the two rules disagreed
# about what a date is made of, and the disagreement was reachable:
#
#     7 turns + 3 events = 10 rows  ->  judged
#     9 turns + 0 events =  9 rows  ->  excluded
#
# The date with LESS conversation in it got scored, purely because the dice
# put events in it. Environment rows are scenery; nobody said anything. This
# now reads the same `turn_count` the cap does, so the two cannot drift again.
#
# The value stays 10, and the change actually RESTORES its original stated
# meaning — "roughly five each" was always a claim about turns. Worth knowing
# that against a 16-turn cap this is a high bar (a date must reach ~62% of a
# full evening to be scored at all), and it errs towards throwing a thin date
# away rather than scoring an evening too slight to have been one. Revisit the
# NUMBER if genuinely half-finished dates start being discarded; do not revisit
# it to make the scores look better.
JUDGEABLE_MIN_TURNS = 10

SCENARIO_MAX_TOKENS = 4096
TURN_MAX_TOKENS = 2048
TURN_TEMPERATURE = 0.85

# S11-B7: in-process asyncio tasks, no Celery/Redis (named trade #1), and a
# GLOBAL semaphore of 2 concurrent pipelines across all users. Free-tier rate
# limits make a third concurrent pipeline actively counterproductive: it would
# not run faster, it would just spread the same throughput across more 429s.
_SEMAPHORE = asyncio.Semaphore(2)
# asyncio holds only a WEAK reference to a running task; a task nobody keeps
# can be garbage-collected mid-flight and the pipeline would vanish with no
# error anywhere. Same reason the persona compiler keeps its set.
_tasks: set[asyncio.Task] = set()
# One pipeline per analysis in THIS process. The DB status is the authority
# across restarts; this stops a double POST racing itself in the meantime.
_running: set[uuid.UUID] = set()

# Module-level so a test can substitute a seeded Random and get a deterministic
# sequence of rolls without monkeypatching the stdlib.
_rng = random.Random()


# --- Pure rules: everything the loop decides, derived from the transcript ---


@dataclass(frozen=True)
class TurnView:
    """The only two things any rule in this module needs to know about a
    stored message. Keeping the rules on this rather than on the ORM row is
    what makes them testable without a database."""

    speaker: str
    wants_to_end: bool = False


def to_views(rows: list[DateMessage]) -> list[TurnView]:
    return [
        TurnView(
            speaker=r.speaker,
            wants_to_end=bool((r.state or {}).get("wants_to_end")),
        )
        for r in rows
    ]


def turn_count(views: list[TurnView]) -> int:
    """How many times an agent has spoken. NOT the row count.

    This is the number the budget is written in: one turn is one model call,
    and an environment row is not a turn. Both the alternation rule and the cap
    read it, so they cannot disagree about what a turn is.
    """
    return sum(1 for v in views if v.speaker != "environment")


def next_speaker(views: list[TurnView]) -> str:
    """Strict alternation, with the requester's agent opening.

    Derived from the COUNT of agent turns rather than from "who spoke last",
    so an environment row between two turns cannot flip the order — and so a
    resumed date computes the same answer the killed one would have.
    """
    return "user_agent" if turn_count(views) % 2 == 0 else "candidate_agent"


def event_count(views: list[TurnView]) -> int:
    return sum(1 for v in views if v.speaker == "environment")


def closing_turns_taken(views: list[TurnView]) -> int | None:
    """How many agent turns have happened since BOTH agents first said they
    were ready to wrap up — or None if that has never been true.

    Recomputed from scratch on every call, deliberately. A stored "we are
    closing now" flag would be one more thing a restart could lose, and the
    transcript already contains the answer.
    """
    latest: dict[str, bool] = {}
    trigger: int | None = None
    for i, v in enumerate(views):
        if v.speaker == "environment":
            continue
        latest[v.speaker] = v.wants_to_end
        if (
            trigger is None
            and latest.get("user_agent")
            and latest.get("candidate_agent")
        ):
            trigger = i
    if trigger is None:
        return None
    return sum(1 for v in views[trigger + 1 :] if v.speaker != "environment")


def is_closing(views: list[TurnView]) -> bool:
    """True while the pair are inside their one final closing exchange, so the
    turn prompt can ask for a goodbye instead of a new subject."""
    taken = closing_turns_taken(views)
    return taken is not None and taken < CLOSING_TURNS


def ended_by(views: list[TurnView]) -> str | None:
    """`mutual_wants_to_end`, `cap`, or None — the two ways a date can finish,
    named so the log can say which mechanism fired (AC4).

    The cap counts TURNS, not rows (revised 2026-09-01). An event that fires at
    turn 15 does not cost the pair their last exchange.
    """
    taken = closing_turns_taken(views)
    if taken is not None and taken >= CLOSING_TURNS:
        return "mutual_wants_to_end"
    if turn_count(views) >= TURN_CAP:
        return "cap"
    return None


def should_inject_event(
    roll: float, views: list[TurnView], *, probability: float = EVENT_PROBABILITY
) -> tuple[bool, str]:
    """The event rule and the REASON it decided what it did (S11-B4.4).

    Returns `(inject, reason)`. The reason is logged on every roll, because
    "an event did not fire" and "an event could not fire" are different facts
    and only one of them is a bug.

    The rules, all three locked by date_simulation.md:
      - never two events in a row;
      - at most 3 per date;
      - p = 0.15 otherwise.

    Plus one boundary this file has to settle (§18 — write the scope down):
    nothing is injected before the first line is spoken. An event is something
    that INTERRUPTS a conversation, and there is nothing to interrupt yet — an
    opening environment row would read as stage directions, not as a date.
    """
    if not views:
        return False, "no_messages_yet"
    if views[-1].speaker == "environment":
        return False, "no_consecutive_events"
    if event_count(views) >= MAX_EVENTS_PER_DATE:
        return False, "event_cap_reached"
    if roll < probability:
        return True, "roll_hit"
    return False, "roll_missed"


def pick_event(possible: list[str], used: list[str]) -> str | None:
    """The first unused event from the scenario's list, in order.

    In order rather than at random: the same event twice in one date is the
    kind of thing that makes a transcript obviously machine-made, and the
    scenario call was already asked for 4-6 distinct ones.
    """
    for e in possible:
        if e not in used:
            return e
    return None


# --- Prompts ---------------------------------------------------------------

SCENARIO_SYSTEM_PROMPT = """\
You invent places for two people to meet on a first date. You are given what \
each of them is into. You produce settings that give them something concrete \
to do and to react to — never a blank room with two chairs.

A good setting does three things: it belongs to at least one of their real \
interests, it gives them something to look at and touch, and it can be \
interrupted. Nothing you invent may state a fact about either person that you \
were not given.

They get one evening together, so make it the RIGHT one — the place these two \
particular people would actually end up, not a place any two people could."""

DATE_PREAMBLE = """\

RIGHT NOW: YOU ARE ON A DATE

You are on a first date with {other_name}, who is {other_age}. You have not \
read a profile about them and nobody has told you what they are like — you \
only know what you can see and what they say to you.

WHERE YOU ARE
{setting_name}. {description}
{sensory_details}

HOW THIS WORKS
Say one thing at a time, the way a person actually talks — you are not writing \
a scene, and you never write {other_name}'s lines. React to what is in front \
of you. If something happens around you, you notice it or you do not, the way \
you would.

Set `wants_to_end` to true when you would genuinely be ready to wrap the \
evening up — because it has run its course, or because it is not working. \
That is a normal way for a date to end, not a complaint."""

CLOSING_INSTRUCTION = """\

You are both winding down now. Say your goodbye the way you would actually say \
it — do not start a new subject."""


# --- Scenario generation (S11-B2, B3) --------------------------------------


async def _interest_traits(session: AsyncSession, user_id: uuid.UUID) -> list[Trait]:
    return list(
        (
            await session.execute(
                select(Trait)
                .where(
                    Trait.user_id == user_id,
                    Trait.category == "interest",
                    Trait.status != "retracted",
                )
                .order_by(Trait.label)
            )
        ).scalars()
    )


def _interest_block(heading: str, traits: list[Trait]) -> str:
    if not traits:
        return f"{heading}\n  (nothing recorded yet)"
    lines = [heading]
    lines.extend(f"  - {t.label}" for t in traits)
    lines.append("  For colour, what each of those means:")
    lines.extend(f"    {t.label}: {t.description}" for t in traits)
    return "\n".join(lines)


def build_scenario_request(
    *,
    user_name: str,
    candidate_name: str,
    user_interests: list[Trait],
    candidate_interests: list[Trait],
    shared: list[str],
) -> str:
    """The user message for the scenario call — including the
    **empty-intersection fallback** (S11-B3).

    The fallback was flagged open in `candidate_matching.md` and closed in
    `date_simulation.md` §2, and it is written here as a DIFFERENT INSTRUCTION
    rather than left to the model's judgement. Handed two interest lists and no
    guidance, a model averages them into a setting that is nobody's — a coffee
    shop.

    **REVISED 2026-09-01 (owner decision: one date per candidate).** The old
    fallback read "one setting anchored in HER interests and one in HIS" — an
    instruction that needs two settings, and that became impossible the moment
    a candidate got only one date. The Source of Truth asks for a setting
    "aligned with at least one of the individuals", so with one evening it has
    to be built around one person's world.

    **It is built around the CANDIDATE's**, and the reason is what a whole
    analysis looks like from the requester's side: they get one date per
    candidate, so across a full pool they are shown three different worlds —
    three people's actual lives — rather than three variations on their own.
    Anchoring on the requester would spend all three evenings in the same kind
    of place with different company, which tells them almost nothing about the
    candidates.

    `anchored_in_interest` records which interest it was, so this stays
    checkable after the fact rather than merely asserted.
    """
    # What the model may anchor on: what they share, or -- when they share
    # nothing -- the candidate's own interests, and ONLY those. Offering
    # both lists here would quietly re-open the choice the fallback exists
    # to make.
    anchor_vocabulary = shared or [t.label for t in candidate_interests]
    head = (
        f"{user_name} and {candidate_name} are going on a first date.\n\n"
        + _interest_block(f"{user_name} is into:", user_interests)
        + "\n\n"
        + _interest_block(f"{candidate_name} is into:", candidate_interests)
    )

    if shared:
        instruction = (
            "\n\nThey share these interests: "
            + ", ".join(shared)
            + ".\n\nBuild the setting around something they share. Set "
            "`anchored_in_interest` to the shared interest it is built "
            "around."
        )
    else:
        instruction = (
            "\n\nThey share NO interests at all. Do not split the difference "
            "and do not invent a common ground they do not have.\n\n"
            f"Build the setting around one of {candidate_name}'s "
            f"interests: {candidate_name} gets the evening in their own "
            f"world, showing it to {user_name}, who has never seen it. "
            "Set `anchored_in_interest` to the interest it is built around."
        )

    return (
        head
        + instruction
        + "\n\n`anchored_in_interest` must be copied word for word from this "
        "list, and nothing else:\n"
        + "\n".join(f"  - {a}" for a in anchor_vocabulary)
        + f"\n\nProduce exactly {SETTINGS_PER_CANDIDATE} "
        + ("setting." if SETTINGS_PER_CANDIDATE == 1 else "settings.")
    )


async def generate_scenarios(
    session: AsyncSession,
    router: TaskRouter,
    *,
    user: User,
    candidate: User,
    shared: list[str],
) -> list[dict]:
    """One structured call per candidate, returning its setting(s) (S11-B2).

    Returns a list because `SETTINGS_PER_CANDIDATE` is a knob, not a constant
    of nature — it was 2 until 2026-09-01 and the caller slices to it.
    """
    user_interests = await _interest_traits(session, user.id)
    candidate_interests = await _interest_traits(session, candidate.id)

    provider, model = router.resolve(SCENARIO_TASK)
    body = build_scenario_request(
        user_name=user.display_name,
        candidate_name=candidate.display_name,
        user_interests=user_interests,
        candidate_interests=candidate_interests,
        shared=shared,
    )
    result = await guarded_structured_call(
        provider,
        GenRequest(
            task=SCENARIO_TASK,
            model=model,
            system_prompt=SCENARIO_SYSTEM_PROMPT,
            messages=[Message(role="user", content=body)],
            temperature=0.9,
            max_tokens=SCENARIO_MAX_TOKENS,
        ),
        DATE_SCENARIOS_V2,
    )
    settings = result["settings"]

    log_event(
        logger,
        "scenarios_generated",
        user_id=str(user.id),
        candidate_user_id=str(candidate.id),
        provider=provider.name,
        model=model,
        # The fallback branch is logged, not inferred: "which prompt did this
        # scenario come from" is the first question when one reads badly.
        mode="shared_interests" if shared else "empty_intersection_fallback",
        shared_interests=shared,
        settings=[
            {
                "setting_name": s["setting_name"],
                "anchored_in_interest": s["anchored_in_interest"],
                "events": len(s["possible_events"]),
            }
            for s in settings
        ],
    )
    return settings


# --- The turn loop (S11-B4) ------------------------------------------------


@dataclass
class DateContext:
    """Everything one date needs, loaded once. Both system prompts are read
    from FROZEN snapshot rows — the candidate's was frozen at match time and
    the requester's at date-creation time — so answering more questions
    mid-simulation cannot change who shows up to a date already running."""

    date: SimulatedDate
    user: User
    candidate: User
    user_prompt: str
    candidate_prompt: str

    def other(self, speaker: str) -> User:
        return self.candidate if speaker == "user_agent" else self.user

    def prompt_for(self, speaker: str) -> str:
        return self.user_prompt if speaker == "user_agent" else self.candidate_prompt


def compose_system_prompt(ctx: DateContext, speaker: str, *, closing: bool) -> str:
    """The frozen persona prompt + the date preamble + the scenario (S11-B4.1).

    The persona prompt comes FIRST and unedited. It contains the user's own
    sentences, and anything prepended to it starts competing with their voice
    for the model's attention — which is the one thing this whole pipeline
    exists to preserve.
    """
    other = ctx.other(speaker)
    scenario = ctx.date.scenario
    prompt = ctx.prompt_for(speaker) + DATE_PREAMBLE.format(
        other_name=other.display_name,
        other_age=compute_age(other.birth_date),
        setting_name=scenario["setting_name"],
        description=scenario["description"],
        sensory_details=scenario["sensory_details"],
    )
    if closing:
        prompt += CLOSING_INSTRUCTION
    return prompt


def compose_messages(rows: list[DateMessage], speaker: str) -> list[Message]:
    """The transcript so far, from this agent's point of view (S11-B4.1).

    The FULL transcript, never a summary: a whole date is at most
    MAX_MESSAGES_PER_DATE rows and fits any context window with room to spare,
    and summarising inside a date would mean the agent forgets the thing they
    were teasing each other about ten lines ago.

    Consecutive same-role messages are merged. Some providers reject two
    `user` messages in a row, and an environment row followed by the other
    person's line is exactly that — transport hygiene, like stripping a code
    fence, not a change to what the agent is told.
    """
    out: list[Message] = []
    for r in rows:
        if r.speaker == "environment":
            role, content = "user", f"(Around you: {r.reply})"
        elif r.speaker == speaker:
            role, content = "assistant", r.reply
        else:
            role, content = "user", r.reply
        if out and out[-1].role == role:
            out[-1] = Message(role=role, content=out[-1].content + "\n\n" + content)
        else:
            out.append(Message(role=role, content=content))

    if not out:
        # The opening turn: there is nothing to reply to, so the cue has to say
        # so out loud, or the model answers an empty conversation with "Hello?".
        return [
            Message(
                role="user",
                content=(
                    "You have just arrived and you are the first to speak. Say "
                    "the first thing you would actually say."
                ),
            )
        ]
    if out[-1].role == "assistant":
        # Only reachable if a transcript ends on this agent's own line, which
        # strict alternation forbids — but a model handed a conversation ending
        # in its own voice replies to itself, so this is worth a cue rather
        # than an assumption.
        out.append(Message(role="user", content="(Go on.)"))
    return out


async def _load_messages(
    session: AsyncSession, date_id: uuid.UUID
) -> list[DateMessage]:
    return list(
        (
            await session.execute(
                select(DateMessage)
                .where(DateMessage.date_id == date_id)
                .order_by(DateMessage.seq)
            )
        ).scalars()
    )


async def _checkpoint(
    session: AsyncSession,
    date_id: uuid.UUID,
    *,
    seq: int,
    speaker: str,
    reply: str,
    state: dict | None = None,
    provider: str | None = None,
    model_id: str | None = None,
) -> DateMessage:
    """Write the message row and COMMIT it. §19: this happens before the turn
    advances, and the loop's next iteration IS the advance."""
    row = DateMessage(
        date_id=date_id, seq=seq, speaker=speaker, reply=reply, state=state,
        provider=provider, model_id=model_id,
    )
    session.add(row)
    await session.commit()
    return row


async def run_date(session: AsyncSession, router: TaskRouter, ctx: DateContext) -> str:
    """Run one date to an ending, resuming from wherever it already is.

    Returns the date's final status. Never raises for a turn failure — that is
    the whole point of the give-up ladder (S11-B6).
    """
    date = ctx.date
    rows = await _load_messages(session, date.id)

    # The log line is written on EVERY entry, including the one where the row
    # already said `running` — which is exactly the case a restart produces.
    # Gating it on "did the status need changing" made the most interesting
    # entry in the file the only silent one: a resumed date logged nothing at
    # all, and §7's test is that the logs alone explain what happened.
    if date.status != "running":
        date.status = "running"
        await session.commit()
    log_event(
        logger, "date_status", date_id=str(date.id),
        analysis_id=str(date.analysis_id), status="running",
        reason="resumed_from_checkpoint" if rows else "started",
        resumed_at_seq=rows[-1].seq if rows else 0,
        next_speaker=next_speaker(to_views(rows)),
    )

    provider, model = router.resolve(DATE_TASK)
    possible_events = list(date.scenario.get("possible_events") or [])

    while True:
        views = to_views(rows)

        finish = ended_by(views)
        if finish is not None:
            date.status = "complete"
            date.finished_at = datetime.now(UTC)
            await session.commit()
            log_event(
                logger, "date_finished", date_id=str(date.id),
                analysis_id=str(date.analysis_id), status="complete",
                ended_by=finish, messages=len(rows), events=event_count(views),
            )
            return "complete"

        # S11-B4.4 — the roll happens before EVERY turn and is logged whether
        # or not it fired (§8: a flag is decorative until you watch it differ).
        roll = _rng.random()
        inject, reason = should_inject_event(roll, views)
        used = [r.reply for r in rows if r.speaker == "environment"]
        chosen = pick_event(possible_events, used) if inject else None
        if inject and chosen is None:
            inject, reason = False, "scenario_events_exhausted"
        log_event(
            logger, "event_roll", date_id=str(date.id), seq=len(rows) + 1,
            roll=round(roll, 4), probability=EVENT_PROBABILITY,
            injected=inject, reason=reason, events_so_far=event_count(views),
            # NOT `event=` — that is `log_event`'s own positional parameter,
            # and passing it as a field raises TypeError from inside the
            # logging call itself (D-010).
            chosen_event=chosen,
        )
        if inject and chosen is not None:
            rows.append(
                await _checkpoint(
                    session, date.id, seq=len(rows) + 1,
                    speaker="environment", reply=chosen,
                )
            )
            # Back around: the event consumed a message slot, so the cap and
            # the ending both get re-asked before anyone speaks. The
            # no-consecutive rule blocks a second roll from firing.
            continue

        speaker = next_speaker(views)
        seq = len(rows) + 1
        try:
            result = await guarded_structured_call(
                provider,
                GenRequest(
                    task=DATE_TASK, model=model,
                    system_prompt=compose_system_prompt(
                        ctx, speaker, closing=is_closing(views)
                    ),
                    messages=compose_messages(rows, speaker),
                    temperature=TURN_TEMPERATURE, max_tokens=TURN_MAX_TOKENS,
                ),
                AGENT_RESPONSE_V1,
            )
        except Exception as exc:  # noqa: BLE001
            # S11-B6. Blind on purpose: the resilience layer and the Guard have
            # already spent their attempts, and whatever is left — a route that
            # vanished, a provider returning something nobody imagined — must
            # end THIS date at its last good message and let the pipeline carry
            # on to the next one.
            date.status = "incomplete"
            date.error = f"{type(exc).__name__}: {exc}"[:2000]
            date.finished_at = datetime.now(UTC)
            await session.commit()
            log_event(
                logger, "date_finished", level=logging.WARNING,
                date_id=str(date.id), analysis_id=str(date.analysis_id),
                status="incomplete", ended_by="turn_gave_up", failed_at_seq=seq,
                speaker=speaker, provider=provider.name, model=model,
                messages=len(rows),
                judgeable=turn_count(views) >= JUDGEABLE_MIN_TURNS,
                turns=turn_count(views),
                error=date.error,
            )
            return "incomplete"

        state = {
            "state_of_mind": result["state_of_mind"],
            "emotional_state": result["emotional_state"],
            "connection": result["connection"],
            "satisfaction": result["satisfaction"],
            "wants_to_end": result["wants_to_end"],
        }
        rows.append(
            await _checkpoint(
                session, date.id, seq=seq, speaker=speaker,
                reply=result["reply"], state=state,
                provider=provider.name, model_id=model,
            )
        )
        log_event(
            logger, "date_turn", date_id=str(date.id),
            analysis_id=str(date.analysis_id), seq=seq, speaker=speaker,
            provider=provider.name, model=model, outcome="ok",
            connection=state["connection"], satisfaction=state["satisfaction"],
            wants_to_end=state["wants_to_end"],
            emotional_state=state["emotional_state"],
            reply_chars=len(result["reply"]),
        )


# --- The pipeline (S11-B7, B8, B10) ----------------------------------------


async def _progress(
    session: AsyncSession, analysis: Analysis, stage: str, message: str, **extra
) -> None:
    """S11-B10. Real stage names, never a fake timer.

    `message` is the sentence the UI shows, written here rather than in the
    client so it cannot drift between clients (§26 — the same reason the
    empty-pool sentence is server-side)."""
    analysis.progress = {
        "stage": stage,
        "message": message,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **extra,
    }
    await session.commit()
    log_event(
        logger, "analysis_progress", analysis_id=str(analysis.id),
        stage=stage, message=message, **extra,
    )


async def _snapshot(session: AsyncSession, snapshot_id: uuid.UUID) -> PersonaSnapshot:
    return (
        await session.execute(
            select(PersonaSnapshot).where(PersonaSnapshot.id == snapshot_id)
        )
    ).scalar_one()


async def _dates_of(session: AsyncSession, analysis_id: uuid.UUID) -> list[SimulatedDate]:
    return list(
        (
            await session.execute(
                select(SimulatedDate)
                .where(SimulatedDate.analysis_id == analysis_id)
                .order_by(SimulatedDate.created_at, SimulatedDate.ordinal)
            )
        ).scalars()
    )


async def ensure_dates(
    session: AsyncSession,
    router: TaskRouter,
    analysis: Analysis,
    user: User,
    user_snapshot_id: uuid.UUID,
) -> list[SimulatedDate]:
    """Create the dates that do not exist yet, one scenario call per candidate.

    Idempotent by construction (S11-B8): a candidate who already has their
    dates is skipped entirely, so a re-launched pipeline neither regenerates a
    scenario nor spends a call to discover that it did not need to.
    """
    candidates = list(
        (
            await session.execute(
                select(AnalysisCandidate)
                .where(AnalysisCandidate.analysis_id == analysis.id)
                .order_by(AnalysisCandidate.rank)
            )
        ).scalars()
    )
    existing = await _dates_of(session, analysis.id)
    have_dates = {d.candidate_user_id for d in existing}

    for candidate_row in candidates:
        if candidate_row.candidate_user_id in have_dates:
            continue
        # S11-B5: the hard ceiling, enforced rather than assumed. Matching caps
        # at 3 candidates so 6 is what falls out — but a cap that only holds
        # because another module happens to be small is not a cap.
        if len(existing) >= MAX_DATES_PER_ANALYSIS:
            log_event(
                logger, "date_cap_reached", level=logging.WARNING,
                analysis_id=str(analysis.id), dates=len(existing),
                cap=MAX_DATES_PER_ANALYSIS,
                skipped_candidate=str(candidate_row.candidate_user_id),
            )
            break

        candidate = (
            await session.execute(
                select(User).where(User.id == candidate_row.candidate_user_id)
            )
        ).scalar_one()
        await _progress(
            session, analysis, "scenarios",
            f"Working out where you and {candidate.display_name} would meet…",
            candidate=candidate.display_name,
        )
        try:
            settings = await generate_scenarios(
                session, router, user=user, candidate=candidate,
                shared=list(candidate_row.shared_interests or []),
            )
        except Exception as exc:  # noqa: BLE001
            # One candidate's scenarios failing must not take the other
            # candidates' dates with it. No date row is written: a `dates` row
            # needs a scenario, and a placeholder scenario would be a fabricated
            # setting sitting in the database looking real (§10).
            log_event(
                logger, "scenario_generation_failed", level=logging.ERROR,
                analysis_id=str(analysis.id),
                candidate_user_id=str(candidate.id),
                error=f"{type(exc).__name__}: {exc}"[:2000],
            )
            continue

        for ordinal, setting in enumerate(settings[:DATES_PER_CANDIDATE], start=1):
            row = SimulatedDate(
                analysis_id=analysis.id,
                candidate_user_id=candidate.id,
                ordinal=ordinal,
                scenario=setting,
                status="pending",
                user_snapshot_id=user_snapshot_id,
                candidate_snapshot_id=candidate_row.snapshot_id,
                schema_version=SCHEMA_VERSION,
            )
            session.add(row)
            existing.append(row)
        await session.commit()
        log_event(
            logger, "dates_created", analysis_id=str(analysis.id),
            candidate_user_id=str(candidate.id),
            dates=len(settings[:DATES_PER_CANDIDATE]),
            settings=[s["setting_name"] for s in settings[:DATES_PER_CANDIDATE]],
        )

    return await _dates_of(session, analysis.id)


async def run_pipeline(
    session: AsyncSession, router: TaskRouter, analysis_id: uuid.UUID
) -> None:
    """The background pipeline. Owns its own failure: an analysis that dies
    must land in `failed` with its error, never sit in `simulating` forever
    with a spinner on the other end."""
    analysis = (
        await session.execute(select(Analysis).where(Analysis.id == analysis_id))
    ).scalar_one()
    try:
        await _simulate(session, router, analysis)
    except Exception as exc:  # noqa: BLE001
        analysis.status = "failed"
        analysis.error = f"{type(exc).__name__}: {exc}"[:2000]
        await session.commit()
        log_event(
            logger, "analysis_status", level=logging.ERROR,
            analysis_id=str(analysis_id), status="failed",
            reason="pipeline_raised", error=analysis.error,
        )


async def _simulate(
    session: AsyncSession, router: TaskRouter, analysis: Analysis
) -> None:
    user = (
        await session.execute(select(User).where(User.id == analysis.user_id))
    ).scalar_one()

    if analysis.status != "simulating":
        resumed_from_failure = analysis.status == "failed"
        analysis.status = "simulating"
        # A retry after `failed` (S13-U5) must not carry the old error text
        # into its new life: a `complete` analysis still saying "openrouter
        # error 400" on the wire would be the log reconstruction test (§7)
        # failing in the other direction — an error that never happened to
        # THIS run. The failure itself is not lost; the previous
        # `analysis_status … failed` line and the `simulation_requested …
        # previous_error` line both keep it.
        analysis.error = None
        await session.commit()
        log_event(
            logger, "analysis_status", analysis_id=str(analysis.id),
            status="simulating",
            reason="resumed_after_failure" if resumed_from_failure else "simulation_started",
        )

    await _progress(
        session, analysis, "queued",
        "Waiting for a free slot — the dates start in a moment.",
    )
    # S11-B7: the global semaphore. The wait is logged with how long it lasted,
    # because "queued behind two others" and "stuck" look identical from
    # outside (AC7).
    started_waiting = time.monotonic()
    async with _SEMAPHORE:
        log_event(
            logger, "simulation_slot_acquired", analysis_id=str(analysis.id),
            waited_ms=int((time.monotonic() - started_waiting) * 1000),
            concurrency_limit=2,
        )

        snapshot = await get_current_snapshot(session, user.id)
        if snapshot is None:
            # The §11 gate. It held at match time; if it does not hold now, the
            # honest answer is to stop, not to send a half-built self on a date.
            raise AIError(
                "this account has no ready persona snapshot to send on a date",
                task=DATE_TASK,
            )

        dates = await ensure_dates(session, router, analysis, user, snapshot.id)
        if not dates:
            raise AIError(
                "no dates could be created — scenario generation failed for "
                "every candidate",
                task=SCENARIO_TASK,
            )

        prompts: dict[uuid.UUID, str] = {}
        completed = incomplete = 0
        for index, date in enumerate(dates, start=1):
            if date.status in ("complete", "incomplete", "failed"):
                # S11-B8: a finished stage is a no-op on re-run. Counted, never
                # re-run — the rows already exist, and re-running would spend a
                # date's worth of calls to arrive at the same transcript.
                completed += date.status == "complete"
                incomplete += date.status == "incomplete"
                continue

            candidate = (
                await session.execute(
                    select(User).where(User.id == date.candidate_user_id)
                )
            ).scalar_one()
            for sid in (date.user_snapshot_id, date.candidate_snapshot_id):
                if sid not in prompts:
                    prompts[sid] = (await _snapshot(session, sid)).system_prompt or ""

            await _progress(
                session, analysis, "simulating",
                f"Simulating date {index} of {len(dates)} — "
                f"{date.scenario['setting_name']}…",
                date_index=index, date_total=len(dates),
                candidate=candidate.display_name,
                setting=date.scenario["setting_name"],
            )
            status = await run_date(
                session, router,
                DateContext(
                    date=date, user=user, candidate=candidate,
                    user_prompt=prompts[date.user_snapshot_id],
                    candidate_prompt=prompts[date.candidate_snapshot_id],
                ),
            )
            completed += status == "complete"
            incomplete += status == "incomplete"

    # Nothing at all was produced: no transcript, no partial date, nothing to
    # judge or to show. That is a failed analysis, and saying otherwise would
    # be a green tick over an empty page.
    total_messages = (
        await session.execute(
            select(func.count())
            .select_from(DateMessage)
            .join(SimulatedDate, SimulatedDate.id == DateMessage.date_id)
            .where(SimulatedDate.analysis_id == analysis.id)
        )
    ).scalar_one()
    if completed == 0 and total_messages == 0:
        raise AIError("every date failed before its first line", task=DATE_TASK)

    # S12-B10: judging happens BEFORE the `complete` transition, because
    # `complete` is the word the UI uses to mean "there are results to read".
    # An analysis that flipped to complete and then scored would show a results
    # screen with no scores on it for a minute.
    #
    # Imported here rather than at module scope: judging imports this module
    # for the judgeable-message threshold, and the pipeline is the only place
    # that needs the judge.
    from app.judging import judge_analysis

    ran = completed + incomplete
    await _progress(
        session, analysis, "judging",
        f"{ran} {'date' if ran == 1 else 'dates'} ran. Scoring "
        f"{'it' if ran == 1 else 'them'} now…",
        dates_complete=completed, dates_incomplete=incomplete,
    )
    try:
        finals = await judge_analysis(session, router, analysis.id, analysis.user_id)
    except Exception as exc:  # noqa: BLE001
        # A judge that cannot run must not throw away six finished dates. The
        # analysis lands `complete` with its transcripts readable and says
        # plainly that the scores are missing — which is a much better place to
        # be than `failed` with everything hidden behind it.
        analysis.status = "complete"
        await session.commit()
        await _progress(
            session, analysis, "judging_failed",
            f"The {'date' if ran == 1 else 'dates'} ran, but scoring "
            f"{'it' if ran == 1 else 'them'} didn't finish. The "
            f"{'transcript is' if ran == 1 else 'transcripts are'} safe and "
            "you can read them.",
            dates_complete=completed, dates_incomplete=incomplete,
            judged=False, error=f"{type(exc).__name__}: {exc}"[:500],
        )
        log_event(
            logger, "analysis_status", level=logging.ERROR,
            analysis_id=str(analysis.id), status="complete",
            reason="judging_failed_dates_kept",
            error=f"{type(exc).__name__}: {exc}"[:2000],
        )
        return

    analysis.status = "complete"
    await session.commit()
    await _progress(
        session, analysis, "done",
        f"{ran} {'date' if ran == 1 else 'dates'} ran and {len(finals)} "
        f"{'person was' if len(finals) == 1 else 'people were'} scored.",
        dates_complete=completed, dates_incomplete=incomplete,
        judged=True, candidates_scored=len(finals),
    )
    log_event(
        logger, "analysis_status", analysis_id=str(analysis.id),
        status="complete", reason="all_dates_judged",
        dates_complete=completed, dates_incomplete=incomplete,
        messages=total_messages, candidates_scored=len(finals),
        final_scores={str(k): round(v, 2) for k, v in finals.items()},
    )


def start_pipeline(app, analysis_id: uuid.UUID) -> bool:
    """Launch the pipeline as a background task unless one is already running
    for this analysis IN THIS PROCESS. Returns False when it declined.

    The ONE place a simulation is started — the endpoint (S11-B11) and the
    boot-time relaunch (S11-B9) both come through here, so they cannot drift
    into two different ideas of what "already running" means (§16).
    """
    if analysis_id in _running:
        return False
    _running.add(analysis_id)

    async def _run() -> None:
        factory = async_sessionmaker(app.state.engine, expire_on_commit=False)
        try:
            async with factory() as session:
                await run_pipeline(session, app.state.ai_router, analysis_id)
        except Exception as exc:  # noqa: BLE001
            # A background task that raises has nowhere to report: no request
            # is waiting on it and asyncio swallows the traceback.
            log_event(
                logger, "simulation_crashed", level=logging.ERROR,
                analysis_id=str(analysis_id),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            _running.discard(analysis_id)

    task = asyncio.create_task(_run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True
