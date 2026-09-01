"""Unit tests for the judge's arithmetic (S12-B5, B6, B7).

This is the module's central claim in test form: **the number is computed in
code, not asked from the model.** Everything below is a pure function of the
four criteria the judge returned, so all of it can be pinned without a model,
a database, or a transcript — and the probe's job reduces to checking that the
stored score matches what these functions say it should be.

The boundaries get their own tests on purpose. `clash_severity` is the one
criterion where high is bad, and the ten-TURN judging threshold is named in
`development_principles.md` §14 as an accretion risk. Both are exactly the kind
of rule that survives a refactor in spirit and dies in arithmetic.
"""

from __future__ import annotations

from app.judging import (
    PARTIAL_WEIGHT,
    WEIGHTS,
    candidate_score,
    date_score,
    is_judgeable,
)
from app.simulation import JUDGEABLE_MIN_TURNS, MAX_EVENTS_PER_DATE


def criteria(alignment=0, flow=0, engagement=0, clash=0) -> dict:
    return {
        "trait_alignment": alignment,
        "conversational_flow": flow,
        "mutual_engagement": engagement,
        "clash_severity": clash,
    }


# --- date_score (S12-B5) ---------------------------------------------------


def test_the_weights_are_the_ones_the_module_plan_locked():
    """Written out longhand rather than read from WEIGHTS, so that changing
    the constants fails this test instead of silently redefining the score
    every stored evaluation was computed under."""
    assert WEIGHTS["trait_alignment"] == 0.30
    assert WEIGHTS["conversational_flow"] == 0.30
    assert WEIGHTS["mutual_engagement"] == 0.25
    assert WEIGHTS["clash_severity"] == 0.15
    assert sum(WEIGHTS.values()) == 1.0


def test_a_perfect_date_scores_100():
    """Perfect means all three positives at 100 AND no clash at all — which is
    `clash_severity: 0`, the value that reads like a failure until you
    remember it is the inverted one."""
    assert date_score(criteria(100, 100, 100, clash=0)) == 100.0


def test_the_worst_possible_date_scores_zero():
    assert date_score(criteria(0, 0, 0, clash=100)) == 0.0


def test_the_score_is_recomputable_by_hand():
    """The exact arithmetic AC1's probe re-does against a stored row."""
    got = date_score(criteria(alignment=80, flow=60, engagement=40, clash=20))
    #  0.30*80 + 0.30*60 + 0.25*40 + 0.15*(100-20)
    #    = 24   +   18   +   10    +   12          = 64
    assert got == 64.0


def test_clash_severity_is_inverted_and_nothing_else_is():
    """Raising a positive criterion raises the score; raising clash LOWERS it.
    If someone ever 'simplifies' the (100 - clash) away, this is the test that
    notices."""
    base = criteria(50, 50, 50, clash=50)
    better_flow = date_score(criteria(50, 90, 50, clash=50))
    worse_clash = date_score(criteria(50, 50, 50, clash=90))
    assert better_flow > date_score(base)
    assert worse_clash < date_score(base)


def test_no_clash_is_worth_a_full_fifteen_points():
    """An empty `clashes` array with `clash_severity: 0` is a valid and common
    verdict (S12-B8) — it must be REWARDED, not treated as a missing score."""
    assert date_score(criteria(clash=0)) - date_score(criteria(clash=100)) == 15.0


# --- candidate_score (S12-B6) ----------------------------------------------


def test_one_complete_date_is_its_own_score():
    assert candidate_score([(64.0, False)]) == 64.0


def test_two_complete_dates_average():
    assert candidate_score([(60.0, False), (80.0, False)]) == 70.0


def test_a_partial_date_counts_half():
    """The locked rule (date_simulation.md #3), by hand:
        (80*1 + 40*0.5) / (1 + 0.5) = 100/1.5 = 66.67
    A plain mean would give 60 — the partial date would drag the score down
    as hard as a full one, which is precisely what the halving exists to
    prevent."""
    assert candidate_score([(80.0, False), (40.0, True)]) == 100 / 1.5
    assert candidate_score([(80.0, False), (40.0, True)]) != 60.0


def test_all_partial_dates_still_average_normally():
    """Halving every weight equally cancels out — a candidate seen only
    through cut-short dates is not additionally penalised for it."""
    assert candidate_score([(60.0, True), (80.0, True)]) == 70.0


def test_the_partial_weight_is_a_half():
    assert PARTIAL_WEIGHT == 0.5


def test_no_judgeable_dates_is_None_and_never_zero():
    """0.0 is a SCORE and it means 'they were terrible together'. A candidate
    whose every date died before it started has no score at all, and the
    difference has to survive all the way to the screen (§10, §11)."""
    assert candidate_score([]) is None


# --- the incomplete-date policy (S12-B7, the §14 accretion risk) -----------


def test_a_complete_date_is_always_judged():
    assert is_judgeable("complete", 16)
    # Even a short complete date: `complete` means it ENDED, by cap or by both
    # of them wanting to. A short real ending is a real date.
    assert is_judgeable("complete", 4)


def test_the_ten_turn_boundary_both_ways():
    assert is_judgeable("incomplete", JUDGEABLE_MIN_TURNS) is True
    assert is_judgeable("incomplete", JUDGEABLE_MIN_TURNS - 1) is False
    assert JUDGEABLE_MIN_TURNS == 10


def test_events_cannot_pad_a_thin_date_over_the_threshold():
    """The reason this threshold counts TURNS (revised 2026-09-01).

    While it counted rows, these two dates were treated in opposite ways —
    and the one with LESS conversation in it was the one that got scored,
    purely because the dice put events in it:

        7 turns + 3 events = 10 rows  ->  judged
        9 turns + 0 events =  9 rows  ->  excluded

    Both are now decided on what was actually said. This test is written in
    the shape of the bug rather than the shape of the rule, because the rule
    reads fine either way and only the example shows the difference.
    """
    thin_but_eventful = 7
    longer_but_quiet = 9
    assert thin_but_eventful + MAX_EVENTS_PER_DATE >= JUDGEABLE_MIN_TURNS  # 10 ROWS
    assert is_judgeable("incomplete", thin_but_eventful) is False
    assert is_judgeable("incomplete", longer_but_quiet) is False
    # ...and the one that genuinely reached ten turns is judged whether or not
    # anything happened around them.
    assert is_judgeable("incomplete", 10) is True


def test_a_date_that_never_ran_is_not_judgeable():
    for status in ("pending", "running", "failed"):
        assert is_judgeable(status, 16) is False, status
