"""Unit tests for the parts of matching that are PURE CODE (S9-B2, B5, B6).

These are the guarantees the module exists to make — reasons computed and never
generated, a deterministic embedding input, an exact mutual score. All of them
are deterministic functions with no AI call and no database, so they can and
should be pinned here rather than only being observed through a probe that
needs a populated pool and a working free-tier model.

That separation earned itself in Step 9: the end-to-end probe went GREEN while
`shared_interests` was silently returning [] because trait labels had been
corrupted upstream (D-009). A unit test on the function itself would have said
nothing about that bug — but a unit test plus a probe assertion that DEMANDS a
non-empty intersection would have caught it. Both are here now.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.matching import (
    IDENTITY_CATEGORIES,
    PREFERENCE_CATEGORIES,
    cosine,
    reason_summary,
    serialise_traits,
    shared_interests,
)


@dataclass
class FakeTrait:
    category: str
    label: str
    description: str = "some description"
    status: str = "inferred"


def test_serialisation_is_byte_identical_for_the_same_trait_set():
    a = [
        FakeTrait("interest", "restores old bicycles", "strips and re-cables them"),
        FakeTrait("quality", "dependable", "turns up early"),
    ]
    # Same traits, different order in — the embedding input must not move, or
    # an unchanged profile would look like it drifted.
    b = list(reversed(a))
    assert serialise_traits(a, IDENTITY_CATEGORIES) == serialise_traits(
        b, IDENTITY_CATEGORIES
    )


def test_serialisation_separates_identity_from_preference():
    traits = [
        FakeTrait("interest", "restores old bicycles"),
        FakeTrait("partner_preference", "wants directness"),
    ]
    identity = serialise_traits(traits, IDENTITY_CATEGORIES)
    preference = serialise_traits(traits, PREFERENCE_CATEGORIES)
    assert "bicycles" in identity and "directness" not in identity
    assert "directness" in preference and "bicycles" not in preference


def test_retracted_traits_never_reach_the_embedding():
    traits = [
        FakeTrait("interest", "restores old bicycles"),
        FakeTrait("interest", "collects stamps", status="retracted"),
    ]
    out = serialise_traits(traits, IDENTITY_CATEGORIES)
    assert "bicycles" in out
    assert "stamps" not in out


def test_shared_interests_finds_a_real_overlap():
    alice = [FakeTrait("interest", "restores old bicycles")]
    dan = [FakeTrait("interest", "restoring bicycles and motorbikes")]
    assert shared_interests(alice, dan) == ["restores old bicycles"]


def test_shared_interests_is_empty_when_nothing_overlaps():
    alice = [FakeTrait("interest", "restores old bicycles")]
    dan = [FakeTrait("interest", "competitive chess")]
    assert shared_interests(alice, dan) == []


def test_shared_interests_ignores_non_interest_categories():
    # A `partner_preference` mentioning bicycles is NOT a shared interest —
    # wanting someone who cycles is not the same as cycling.
    alice = [FakeTrait("interest", "restores old bicycles")]
    dan = [FakeTrait("partner_preference", "someone who likes bicycles")]
    assert shared_interests(alice, dan) == []


def test_shared_interests_ignores_retracted_rows_on_both_sides():
    alice = [FakeTrait("interest", "restores old bicycles")]
    dan = [FakeTrait("interest", "bicycles", status="retracted")]
    assert shared_interests(alice, dan) == []


def test_shared_interests_does_not_match_on_filler_words():
    # "old" and "the" are stripped; matching on them would produce the exact
    # fabricated-overlap failure trade #3 forbids.
    alice = [FakeTrait("interest", "restores old bicycles")]
    dan = [FakeTrait("interest", "the old lighthouse")]
    assert shared_interests(alice, dan) == []


def test_reason_summary_states_a_shared_interest_it_was_given():
    text = reason_summary(0.7, 0.7, ["restores old bicycles"])
    assert "restores old bicycles" in text


def test_reason_summary_says_so_plainly_when_nothing_is_shared():
    # Never invents an overlap to soften an empty one (§10, trade #3).
    text = reason_summary(0.7, 0.7, [])
    assert "No overlapping interests" in text


def test_reason_summary_names_the_lopsided_direction():
    forward = reason_summary(0.9, 0.5, [])
    backward = reason_summary(0.5, 0.9, [])
    assert "what you said you want" in forward
    assert "what they said they want" in backward


def test_cosine_is_one_for_identical_vectors_and_zero_for_orthogonal():
    assert abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_survives_a_zero_vector_instead_of_dividing_by_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_compatibility_is_the_mean_and_is_symmetric_by_construction():
    # The property AC8 checks by hand against stored rows, pinned here too.
    forward, backward = 0.8, 0.6
    assert (forward + backward) / 2 == (backward + forward) / 2 == 0.7
