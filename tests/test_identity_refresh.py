"""D-017: a rename must reach the agent, and must not touch anything else.

These build a REAL prompt with `_assemble_prompt` and then operate on it, so
the day someone edits the prompt's shape and the `WHO YOU ARE` seam moves,
these fail here rather than silently in production — where the symptom is an
agent politely using a name its owner abandoned weeks ago.
"""

from __future__ import annotations

from datetime import date

from app.models import User
from app.persona import (
    Excerpt,
    _assemble_prompt,
    _build_facts,
    facts_block_of,
    identity_drifted,
    rewrite_identity,
)

DIGEST = {
    "in_tense_moments": "goes quiet, then says the sharp thing.",
    "when_flirting": "teases, then over-explains the tease.",
    "when_supporting": "turns up with food.",
    "when_opening_up": "late, and all at once.",
}


def _user(name: str = "Ana", city: str | None = "Cluj-Napoca") -> User:
    return User(
        email="a@example.com",
        password_hash="x",
        display_name=name,
        birth_date=date(1996, 4, 18),
        gender="woman",
        interested_in=["man", "woman"],
        age_pref_min=24,
        age_pref_max=45,
        city=city,
        country="Romania",
    )


def _prompt(user: User, *, excerpt: str = "I bake bread on Sundays.") -> str:
    return _assemble_prompt(
        user,
        "WHAT YOU ARE LIKE\ndirect: says the thing",
        [Excerpt("BQ1", "interests", excerpt)],
        DIGEST,
        [],
    )


def test_the_facts_block_can_be_read_back_out_of_a_real_prompt():
    user = _user()
    assert facts_block_of(_prompt(user)) == _build_facts(user)


def test_a_prompt_without_the_seam_is_not_guessed_at():
    assert facts_block_of("You are Ana. Nothing else.") is None
    assert rewrite_identity("You are Ana. Nothing else.", _user()) is None


def test_an_unchanged_profile_has_not_drifted():
    user = _user()
    assert identity_drifted(user, _prompt(user)) is False


def test_a_rename_is_drift():
    prompt = _prompt(_user("Ana"))
    assert identity_drifted(_user("Antonia"), prompt) is True


def test_moving_city_is_drift_too():
    prompt = _prompt(_user("Ana", city="Cluj-Napoca"))
    assert identity_drifted(_user("Ana", city="Iasi"), prompt) is True


def test_rewriting_updates_both_places_the_name_appears():
    old = _prompt(_user("Ana"))
    assert old.startswith("You are Ana. You are not an assistant playing Ana;")
    new = rewrite_identity(old, _user("Antonia"))
    assert new is not None
    assert new.startswith("You are Antonia. You are not an assistant playing Antonia;")
    assert "Name: Antonia" in new
    assert "Name: Ana" not in new


def test_rewriting_leaves_the_person_s_own_words_alone():
    # The excerpt is what they WROTE. If they wrote their old name in an
    # answer, it stays: editing someone's words to match their new profile
    # would be forging the voice sample the persona is built from.
    old = _prompt(_user("Ana"), excerpt="My sister still calls me Ana-Maria.")
    new = rewrite_identity(old, _user("Antonia"))
    assert new is not None
    assert "My sister still calls me Ana-Maria." in new


def test_rewriting_keeps_everything_below_the_facts_byte_for_byte():
    user = _user("Ana")
    old = _prompt(user)
    new = rewrite_identity(old, _user("Antonia"))
    tail_of_old = old.split("\n\n", 2)[2]
    tail_of_new = new.split("\n\n", 2)[2]
    assert tail_of_new == tail_of_old


def test_rewriting_is_idempotent():
    renamed = _user("Antonia")
    once = rewrite_identity(_prompt(_user("Ana")), renamed)
    assert identity_drifted(renamed, once) is False
    assert rewrite_identity(once, renamed) == once


def test_a_rewritten_prompt_matches_what_the_compiler_would_have_written():
    # The strongest form of the claim: repairing a prompt in place produces
    # the same text as compiling it fresh for the renamed person would have —
    # so the free repair is not a second, subtly different code path.
    renamed = _user("Antonia")
    repaired = rewrite_identity(_prompt(_user("Ana")), renamed)
    assert repaired == _prompt(renamed)
