"""S17 rejection: the three decisions that can be checked without a database.

The endpoint's behaviour is mostly database work, but the parts that decide
WHETHER a rejection is allowed and WHAT ORDER the seats end up in are pure —
and they are the parts a future edit is most likely to get wrong. Pinning them
here means the probe is witnessing plumbing, not logic (§18).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.matching import (
    MAX_CANDIDATES,
    Scored,
    _candidate_row,
    ranked_order,
    rejection_refusal,
    would_leave_nobody,
)

# --- the state gate --------------------------------------------------------


def test_only_a_matched_analysis_can_be_recast():
    assert rejection_refusal("matched") is None


def test_every_other_state_is_refused_with_its_own_sentence():
    messages = {}
    for status in ("matching", "no_candidates", "simulating", "complete", "failed"):
        refusal = rejection_refusal(status)
        assert refusal is not None, status
        code, message = refusal
        assert code == "cannot_reject_now"
        assert message.strip()
        messages[status] = message
    # Named states, not one generic refusal: "the dates are running" and
    # "this one already finished" are different things to be told.
    assert len(set(messages.values())) == len(messages)


def test_simulating_refusal_says_why_rather_than_just_no():
    _, message = rejection_refusal("simulating")
    assert "running" in message.lower()


def test_an_unknown_future_status_still_refuses():
    # A status this function has never heard of must not fall through to
    # "allowed" — a new state is a reason to think, not a reason to permit.
    refusal = rejection_refusal("archived")
    assert refusal is not None and refusal[0] == "cannot_reject_now"


# --- the last-candidate gate -----------------------------------------------


def test_rejecting_the_last_candidate_with_no_replacement_is_refused():
    assert would_leave_nobody(active_count=1, replacement_found=False) is True


def test_rejecting_the_last_candidate_is_fine_when_someone_can_take_the_seat():
    assert would_leave_nobody(active_count=1, replacement_found=True) is False


def test_rejecting_one_of_several_is_always_fine():
    for n in range(2, MAX_CANDIDATES + 1):
        assert would_leave_nobody(active_count=n, replacement_found=False) is False


# --- re-ranking ------------------------------------------------------------


@dataclass
class _Row:
    """Just enough of an AnalysisCandidate for the ordering rule."""

    compatibility: float
    candidate_user_id: uuid.UUID
    rank: int = 0


def test_rank_follows_compatibility_after_a_swap():
    a = _Row(0.61, uuid.UUID(int=1))
    b = _Row(0.72, uuid.UUID(int=2))
    c = _Row(0.66, uuid.UUID(int=3))
    assert [r.compatibility for r in ranked_order([a, b, c])] == [0.72, 0.66, 0.61]


def test_ties_break_on_user_id_exactly_as_the_scorer_does():
    # `score_pool` sorts by (-compatibility, str(user_id)). If these two rules
    # disagreed, a replacement could silently re-order two tied people who
    # were already on screen in the other order.
    low = _Row(0.5, uuid.UUID("00000000-0000-0000-0000-00000000000a"))
    high = _Row(0.5, uuid.UUID("00000000-0000-0000-0000-00000000000b"))
    assert ranked_order([high, low]) == [low, high]


def test_ordering_does_not_mutate_the_rows_it_is_given():
    rows = [_Row(0.2, uuid.UUID(int=7), rank=1), _Row(0.9, uuid.UUID(int=8), rank=2)]
    ranked_order(rows)
    assert [r.rank for r in rows] == [1, 2]


# --- the replacement row ---------------------------------------------------


def test_a_replacement_is_built_by_the_same_function_as_the_original_three():
    analysis_id = uuid.uuid4()
    s = Scored(
        user_id=uuid.uuid4(),
        snapshot_id=uuid.uuid4(),
        fit_forward=0.8,
        fit_backward=0.6,
        shared=["live music"],
    )
    row = _candidate_row(analysis_id, s, rank=2)
    assert row.status == "active"
    assert row.rank == 2
    assert row.compatibility == s.compatibility == 0.7
    assert row.shared_interests == ["live music"]
    # The reason sentence is computed, never generated — and a replacement
    # gets one for the same reason the first three do.
    assert "live music" in row.reason_summary
