"""Unit tests for the parts of date simulation that are PURE CODE (S11-B4, B5).

Every rule the turn loop obeys is a function of the stored transcript and
nothing else — whose turn it is, whether an event may fire, whether the pair
have started saying goodbye, and which of the two endings finished the date.
That is not an accident of style: it is what makes resume work. A killed
server restarts, reloads the rows, and recomputes the same answers the dead
process would have reached.

So these tests are the resume guarantee, stated in the only place it can be
checked cheaply. The probe watches it happen once against a real server; these
pin the arithmetic on every run, including the boundaries a probe would have to
be extremely lucky to hit — a 30th message that happens to be an environment
row, an event roll landing exactly on the threshold, a date where one person
wants to leave and the other does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.schemas.date_scenarios import SETTINGS_PER_CANDIDATE
from app.simulation import (
    CLOSING_TURNS,
    DATES_PER_CANDIDATE,
    EVENT_PROBABILITY,
    MAX_DATES_PER_ANALYSIS,
    MAX_EVENTS_PER_DATE,
    MESSAGE_CAP,
    DateContext,
    TurnView,
    build_scenario_request,
    closing_turns_taken,
    compose_messages,
    compose_system_prompt,
    ended_by,
    event_count,
    is_closing,
    next_speaker,
    pick_event,
    should_inject_event,
    to_views,
)

USER = "user_agent"
CANDIDATE = "candidate_agent"
ENV = "environment"


@dataclass
class FakeMessage:
    speaker: str
    reply: str = "something said"
    state: dict | None = None


def views(*spec: tuple[str, bool] | str) -> list[TurnView]:
    """`views(USER, (CANDIDATE, True), ENV)` — speakers, optionally with the
    `wants_to_end` they reported."""
    out = []
    for item in spec:
        if isinstance(item, tuple):
            out.append(TurnView(speaker=item[0], wants_to_end=item[1]))
        else:
            out.append(TurnView(speaker=item))
    return out


# --- Whose turn it is (S11-B4) ---------------------------------------------


def test_the_requesters_agent_opens_the_date():
    assert next_speaker([]) == USER


def test_speakers_alternate():
    assert next_speaker(views(USER)) == CANDIDATE
    assert next_speaker(views(USER, CANDIDATE)) == USER
    assert next_speaker(views(USER, CANDIDATE, USER)) == CANDIDATE


def test_an_environment_row_does_not_take_a_turn():
    """The event is the world doing something, not a participant speaking. If
    it consumed a turn the two agents would swap sides mid-date."""
    assert next_speaker(views(USER, ENV)) == CANDIDATE
    assert next_speaker(views(USER, ENV, ENV, ENV)) == CANDIDATE


def test_to_views_reads_wants_to_end_out_of_the_stored_state():
    rows = [
        FakeMessage(USER, state={"wants_to_end": True}),
        FakeMessage(CANDIDATE, state={"wants_to_end": False}),
        FakeMessage(ENV, state=None),
    ]
    assert [v.wants_to_end for v in to_views(rows)] == [True, False, False]


# --- Event injection: the three locked rules (S11-B4.4) ---------------------


def test_no_event_before_anyone_has_spoken():
    inject, reason = should_inject_event(0.0, [])
    assert not inject and reason == "no_messages_yet"


def test_never_two_events_in_a_row():
    """Locked by date_simulation.md. A roll of 0.0 always hits, so this proves
    the rule blocks it rather than the probability sparing us."""
    inject, reason = should_inject_event(0.0, views(USER, ENV))
    assert not inject and reason == "no_consecutive_events"


def test_at_most_three_events_per_date():
    transcript = views(USER, ENV, CANDIDATE, ENV, USER, ENV, CANDIDATE)
    assert event_count(transcript) == MAX_EVENTS_PER_DATE
    inject, reason = should_inject_event(0.0, transcript)
    assert not inject and reason == "event_cap_reached"


def test_the_third_event_is_still_allowed():
    """The boundary on the other side: two fired, so a third may (§18)."""
    transcript = views(USER, ENV, CANDIDATE, ENV, USER)
    inject, reason = should_inject_event(0.0, transcript)
    assert inject and reason == "roll_hit"


def test_the_probability_boundary_is_exclusive():
    """p = 0.15 means a roll BELOW 0.15 fires. A roll of exactly 0.15 does
    not — stated here so nobody 'fixes' it into <= and quietly changes the
    event rate."""
    assert should_inject_event(EVENT_PROBABILITY - 1e-9, views(USER))[0] is True
    assert should_inject_event(EVENT_PROBABILITY, views(USER))[0] is False
    assert should_inject_event(0.9, views(USER)) == (False, "roll_missed")


def test_events_are_used_in_order_and_never_twice():
    possible = ["the band starts up", "a dog gets loose", "it starts raining"]
    assert pick_event(possible, []) == "the band starts up"
    assert pick_event(possible, ["the band starts up"]) == "a dog gets loose"
    assert pick_event(possible, possible) is None


# --- How a date ends (S11-B4.5, AC4) ---------------------------------------


def test_a_date_in_full_flow_has_not_ended():
    assert ended_by(views(USER, CANDIDATE, USER)) is None
    assert closing_turns_taken(views(USER, CANDIDATE)) is None


def test_one_person_wanting_to_leave_is_not_an_ending():
    """It takes both. Otherwise every date ends the moment one agent has a
    quiet moment, and `wants_to_end` becomes a hair trigger."""
    transcript = views((USER, True), (CANDIDATE, False), (USER, True))
    assert closing_turns_taken(transcript) is None
    assert ended_by(transcript) is None


def test_mutual_wants_to_end_opens_one_closing_exchange():
    transcript = views((USER, True), (CANDIDATE, True))
    assert closing_turns_taken(transcript) == 0
    assert is_closing(transcript)
    # Not over yet — the closing exchange still has to happen.
    assert ended_by(transcript) is None


def test_the_date_ends_after_the_closing_exchange():
    transcript = views((USER, True), (CANDIDATE, True), (USER, True), (CANDIDATE, True))
    assert closing_turns_taken(transcript) == CLOSING_TURNS
    assert not is_closing(transcript)
    assert ended_by(transcript) == "mutual_wants_to_end"


def test_an_event_during_the_goodbye_is_not_a_closing_turn():
    """Only the agents' lines count towards the closing exchange — otherwise a
    single event would end the date one line early."""
    transcript = views((USER, True), (CANDIDATE, True), ENV, (USER, True))
    assert closing_turns_taken(transcript) == 1
    assert ended_by(transcript) is None


def test_changing_your_mind_after_the_trigger_does_not_cancel_the_goodbye():
    """Both said they were done; the closing exchange runs. A model that
    flips the flag back on its farewell line must not restart the date."""
    transcript = views((USER, True), (CANDIDATE, True), (USER, False), (CANDIDATE, False))
    assert ended_by(transcript) == "mutual_wants_to_end"


def test_the_cap_ends_a_date_that_never_wound_down():
    transcript = views(*[USER if i % 2 == 0 else CANDIDATE for i in range(MESSAGE_CAP)])
    assert ended_by(transcript) == "cap"
    assert ended_by(transcript[:-1]) is None


def test_the_cap_counts_environment_rows_as_messages():
    """development_principles.md §18 writes this scope down because it is
    exactly the rule someone later 'fixes' into counting only what was said."""
    spoken = [USER if i % 2 == 0 else CANDIDATE for i in range(MESSAGE_CAP - 3)]
    transcript = views(*spoken, ENV, ENV, ENV)
    assert len(transcript) == MESSAGE_CAP
    assert ended_by(transcript) == "cap"


# --- Composing what the agent actually sees (S11-B4.1) ---------------------


def test_the_transcript_is_told_from_the_speakers_point_of_view():
    """The same two rows, read by each agent in turn. Each transcript is built
    at the moment that agent is about to speak, so it ends on the OTHER one's
    line both times."""
    rows = [FakeMessage(USER, "I got here early."), FakeMessage(CANDIDATE, "Of course you did.")]
    mine = compose_messages(rows, USER)
    assert [(m.role, m.content) for m in mine] == [
        ("assistant", "I got here early."),
        ("user", "Of course you did."),
    ]
    theirs = compose_messages(rows[:1], CANDIDATE)
    assert [(m.role, m.content) for m in theirs] == [("user", "I got here early.")]


def test_a_transcript_ending_on_the_speakers_own_line_gets_a_cue():
    """Strict alternation forbids this state, so the cue is a guard rather
    than a path — but a model handed a conversation ending in its own voice
    replies to itself, and that is worth one sentence of insurance."""
    rows = [FakeMessage(USER, "I got here early.")]
    messages = compose_messages(rows, USER)
    assert [(m.role, m.content) for m in messages] == [
        ("assistant", "I got here early."),
        ("user", "(Go on.)"),
    ]


def test_the_opening_turn_gets_a_cue_rather_than_an_empty_conversation():
    messages = compose_messages([], USER)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert "first to speak" in messages[0].content


def test_an_event_is_marked_as_the_world_not_as_the_other_person():
    rows = [FakeMessage(USER, "So."), FakeMessage(ENV, "the power cuts out")]
    messages = compose_messages(rows, USER)
    assert messages[-1].role == "user"
    assert "(Around you: the power cuts out)" in messages[-1].content


def test_consecutive_same_role_messages_are_merged():
    """An environment row followed by the other person's line is two `user`
    messages in a row, which some providers reject outright."""
    rows = [
        FakeMessage(USER, "So."),
        FakeMessage(ENV, "the power cuts out"),
        FakeMessage(CANDIDATE, "Well, that's atmospheric."),
    ]
    messages = compose_messages(rows, USER)
    assert [m.role for m in messages] == ["assistant", "user"]
    assert "power cuts out" in messages[1].content
    assert "atmospheric" in messages[1].content


# --- The system prompt for a turn ------------------------------------------


@dataclass
class FakeUser:
    display_name: str
    birth_date: date = date(1992, 6, 15)


@dataclass
class FakeDate:
    scenario: dict = field(
        default_factory=lambda: {
            "setting_name": "the Sunday car meet at the docks",
            "description": "Rows of half-finished projects on the tarmac.",
            "sensory_details": "Cold air, petrol, someone's radio.",
            "anchored_in_interest": "restores old cars",
            "possible_events": ["a vintage Mustang pulls up"],
        }
    )


def make_ctx() -> DateContext:
    return DateContext(
        date=FakeDate(),
        user=FakeUser("Alice"),
        candidate=FakeUser("Dan"),
        user_prompt="YOU ARE ALICE. Alice's own words go here.",
        candidate_prompt="YOU ARE DAN. Dan's own words go here.",
    )


def test_the_persona_prompt_comes_first_and_unedited():
    """Anything prepended to the snapshot starts competing with the user's own
    sentences for the model's attention — and that voice is the product."""
    prompt = compose_system_prompt(make_ctx(), USER, closing=False)
    assert prompt.startswith("YOU ARE ALICE. Alice's own words go here.")
    assert "YOU ARE DAN" not in prompt


def test_each_agent_is_told_who_they_are_sitting_with_and_nothing_more():
    prompt = compose_system_prompt(make_ctx(), USER, closing=False)
    assert "first date with Dan" in prompt
    assert "the Sunday car meet at the docks" in prompt
    # The other person's persona, traits and answers stay on the server side of
    # this line — an agent knows what it can see and what it is told.
    assert "Dan's own words" not in prompt


def test_the_closing_instruction_appears_only_while_closing():
    ctx = make_ctx()
    assert "winding down" not in compose_system_prompt(ctx, USER, closing=False)
    assert "winding down" in compose_system_prompt(ctx, USER, closing=True)


# --- The scenario request, both branches (S11-B2, B3) ----------------------


@dataclass
class FakeTrait:
    label: str
    description: str = "they do this a lot"


def test_a_shared_interest_drives_the_setting():
    body = build_scenario_request(
        user_name="Alice", candidate_name="Dan",
        user_interests=[FakeTrait("restores old bicycles")],
        candidate_interests=[FakeTrait("restores old bicycles")],
        shared=["restores old bicycles"],
    )
    assert "They share these interests: restores old bicycles" in body
    assert "Build the setting around something they share" in body
    assert "Produce exactly 1 setting." in body


def test_the_empty_intersection_fallback_anchors_on_the_candidate():
    """S11-B3, closed in date_simulation.md §2 and NOT left to the model's
    judgement: handed two lists and no instruction, a model averages them into
    a setting that belongs to neither person.

    REVISED 2026-09-01 with the move to one date per candidate. The old rule
    was "one setting hers, one his", which needs two settings. With one
    evening it is built around the CANDIDATE's world, so that a requester
    working through a full pool is shown three different lives rather than
    three versions of their own.
    """
    body = build_scenario_request(
        user_name="Alice", candidate_name="Dan",
        user_interests=[FakeTrait("sea swimming")],
        candidate_interests=[FakeTrait("competitive chess")],
        shared=[],
    )
    assert "share NO interests" in body
    assert "Build the setting around one of Dan's" in body
    assert "Dan gets the evening in their own" in body


def test_the_fallback_offers_only_the_candidates_interests_to_anchor_on():
    """The anchor vocabulary is the instruction's teeth. Leaving the
    requester's interests in the list would quietly re-open the choice the
    fallback exists to make, and the model would take it."""
    body = build_scenario_request(
        user_name="Alice", candidate_name="Dan",
        user_interests=[FakeTrait("sea swimming")],
        candidate_interests=[FakeTrait("competitive chess")],
        shared=[],
    )
    vocabulary = body.split("copied word for word from this list")[1]
    assert "competitive chess" in vocabulary
    assert "sea swimming" not in vocabulary


def test_the_anchor_vocabulary_is_the_shared_list_when_there_is_one():
    body = build_scenario_request(
        user_name="Alice", candidate_name="Dan",
        user_interests=[FakeTrait("sea swimming"), FakeTrait("restores old bicycles")],
        candidate_interests=[FakeTrait("restores old bicycles")],
        shared=["restores old bicycles"],
    )
    vocabulary = body.split("copied word for word from this list")[1]
    assert "restores old bicycles" in vocabulary
    # `sea swimming` is Alice's alone, so anchoring a "shared" setting on it
    # would be the fabricated common ground the matching module forbids.
    assert "sea swimming" not in vocabulary


# --- the caps themselves (revised 2026-09-01) ------------------------------


def test_one_date_per_candidate_and_three_per_analysis():
    """The owner's revision, pinned. Two constants used to carry this number
    -- the schema's `SETTINGS_PER_CANDIDATE` and simulation's
    `DATES_PER_CANDIDATE` -- and they had to agree by hand. They are now the
    same object, which is what this first assertion is really checking."""
    assert DATES_PER_CANDIDATE is SETTINGS_PER_CANDIDATE
    assert DATES_PER_CANDIDATE == 1
    # Matching caps the pool at 3, so 3 dates is the ceiling -- but it is
    # written down here rather than inherited, so it still binds if matching
    # ever returns more.
    assert MAX_DATES_PER_ANALYSIS == 3
