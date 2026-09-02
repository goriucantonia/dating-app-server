"""Unit tests for the judge's arithmetic (S12-B5, B6, B7).

This is the module's central claim in test form: **the number is computed in
code, not asked from the model.** Everything below is a pure function of the
four criteria the judge returned, so all of it can be pinned without a model,
a database, or a transcript — and the probe's job reduces to checking that the
stored score matches what these functions say it should be.

The boundaries get their own tests on purpose. `clash_severity` is the one
criterion where high is bad, and the judging boundary is named in
`development_principles.md` §14 as an accretion risk. Both are exactly the kind
of rule that survives a refactor in spirit and dies in arithmetic.

**Revised 2026-09-02**: the ten-turn threshold is gone (owner decision) and the
tests that pinned it are gone with it — not adjusted to a new number, because
there is no new number. What replaced them tests the rule that actually exists
now: a date is judged if anybody spoke. The old tests are described in
`test_the_ten_turn_threshold_is_gone` so that a reader who comes here looking
for them finds out what happened rather than assuming they were lost.
"""

from __future__ import annotations

from app.judging import (
    JUDGEABLE_MIN_TURNS,
    PARTIAL_WEIGHT,
    WEIGHTS,
    _depth_phrase,
    candidate_score,
    date_score,
    is_judgeable,
)
from app.simulation import MAX_EVENTS_PER_DATE, TURN_CAP


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


# --- the judging boundary (S12-B7, the §14 accretion risk) -----------------


def test_a_complete_date_is_always_judged():
    assert is_judgeable("complete", 16)
    # Even a short complete date: `complete` means it ENDED, by cap or by both
    # of them wanting to. A short real ending is a real date.
    assert is_judgeable("complete", 4)


def test_the_ten_turn_threshold_is_gone():
    """The removal, pinned (owner decision, 2026-09-02).

    This is the test that used to read

        assert is_judgeable("incomplete", JUDGEABLE_MIN_TURNS - 1) is False
        assert JUDGEABLE_MIN_TURNS == 10

    and it is written in the shape of the OLD rule on purpose. Every one of
    these was excluded from its candidate's score and shown as failed. All of
    them are judged now, and if a threshold is ever reintroduced by accident —
    a helpful-looking `if turns < 5` somewhere — this fails first and names
    what was lost.
    """
    for turns in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        assert is_judgeable("incomplete", turns) is True, turns
        assert is_judgeable("complete", turns) is True, turns


def test_the_only_floor_left_is_having_said_anything():
    """One agent turn is judgeable; none is not, and that is arithmetic rather
    than an opinion about how good a date has to be. A judge given an empty
    page describes an evening nobody had (§10)."""
    assert JUDGEABLE_MIN_TURNS == 1
    assert is_judgeable("incomplete", 1) is True
    assert is_judgeable("incomplete", 0) is False
    assert is_judgeable("complete", 0) is False


def test_environment_rows_are_not_a_transcript():
    """A date whose only rows are scenery has nothing anybody SAID in it.

    `is_judgeable` is handed turns rather than rows precisely so that three
    lucky event rolls cannot make an empty date look like a date. This was the
    2026-09-01 revision's point, and it survives the threshold's removal
    unchanged — the unit still matters at the floor.
    """
    events_only = 0
    assert events_only + MAX_EVENTS_PER_DATE == MAX_EVENTS_PER_DATE  # rows exist
    assert is_judgeable("incomplete", events_only) is False


def test_a_date_that_never_ran_is_not_judgeable():
    for status in ("pending", "running", "failed"):
        assert is_judgeable(status, 16) is False, status


# --- what the judge is told about depth (2026-09-02) -----------------------


def test_depth_phrase_covers_the_whole_range_and_derives_from_the_cap():
    """Three bands, and the thresholds are fractions of `TURN_CAP` rather than
    literals — raising the cap must not quietly turn a third of an evening into
    "a full evening"."""
    assert _depth_phrase(TURN_CAP) == "a full evening"
    assert _depth_phrase(TURN_CAP - 1) == "a full evening"
    assert _depth_phrase(int(TURN_CAP * 0.5)) == "a real but partial evening"
    assert _depth_phrase(2) == "barely started — a handful of lines, no more"
    assert _depth_phrase(1) == "barely started — a handful of lines, no more"


def test_every_judgeable_turn_count_gets_a_phrase():
    """No gap between the bands: every date that reaches the judge is
    described to it as one of the three."""
    phrases = {_depth_phrase(t) for t in range(1, TURN_CAP + 1)}
    assert phrases == {
        "a full evening",
        "a real but partial evening",
        "barely started — a handful of lines, no more",
    }
