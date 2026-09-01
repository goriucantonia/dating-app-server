"""The cascade graph is verified, not assumed (S15-B1, §13).

`cascade_tables()` walks the ORM metadata from `users` along ON DELETE
CASCADE foreign keys. These tests pin that every table it reaches is counted
before deletion, that the global question rows are NOT on any path from
`users`, and that the demo fixture obeys the A1 rules and the 50-char floor —
the two data-hygiene mechanisms that can be checked without a database.
"""

from __future__ import annotations

from app.deletion import (
    SURVIVOR_QUERIES,
    USER_ROW_COUNTS,
    cascade_tables,
    counted_tables,
)
from app.demo import BASELINE_CODES, MIN_ANSWER_CHARS, load_demo_profiles
from app.models import Base


def test_cascade_reaches_every_table_that_hangs_off_users():
    reached = cascade_tables(Base.metadata)
    # The whole graph, named — a new table that joins it must be added HERE
    # as well as to the count list, which is the point of listing it.
    assert reached == {
        "answers", "questions", "traits", "trait_events", "profile_embeddings",
        "persona_snapshots", "calibration_sessions", "calibration_messages",
        "analyses", "analysis_candidates", "dates", "date_messages",
        "date_evaluations", "candidate_scores", "chat_sessions", "chat_messages",
    }


def test_every_cascaded_table_is_counted_before_the_delete():
    missing = cascade_tables(Base.metadata) - counted_tables()
    assert not missing, f"tables the cascade reaches but nobody counts: {missing}"


def test_counts_are_scoped_to_the_user():
    # A count without `:uid` would report the whole table as "about to go".
    for label, sql in USER_ROW_COUNTS:
        assert ":uid" in sql, label
    for key, sql in SURVIVOR_QUERIES.items():
        assert ":uid" in sql, key


def test_global_questions_are_only_counted_when_owned():
    # The one count that touches `questions` must be the per-user (dispute)
    # subset; BQ/PQ rows have user_id NULL and must survive every deletion.
    (label, sql), = [(l, s) for l, s in USER_ROW_COUNTS if "FROM questions" in s]
    assert label == "dispute_questions"
    assert "user_id = :uid" in sql


def test_snapshot_references_are_all_cleared_by_another_cascade_path():
    # persona_snapshots is referenced WITHOUT cascade by four tables. Each of
    # those tables must itself be reachable from `users` by a cascade, or a
    # deletion would fail on the FK. The set below is the audit.
    referencing = {
        t.name
        for t in Base.metadata.tables.values()
        for fk in t.foreign_keys
        if fk.column.table.name == "persona_snapshots" and (fk.ondelete or "") == ""
    }
    assert referencing == {"calibration_sessions", "analysis_candidates", "dates", "chat_sessions"}
    assert referencing <= cascade_tables(Base.metadata)


def test_demo_fixture_obeys_the_form_rules_and_the_floor():
    password, profiles = load_demo_profiles()
    assert len(password) >= 8
    assert len(profiles) >= 2
    emails = [p.email for p in profiles]
    assert len(set(emails)) == len(emails)
    for p in profiles:
        assert set(p.answers) >= set(BASELINE_CODES)
        for code in BASELINE_CODES:
            assert len(p.answers[code]) >= MIN_ANSWER_CHARS, (p.email, code)
        assert 18 <= p.age_pref_min <= p.age_pref_max
        assert p.opt_in is True  # demo profiles exist to be matched against
