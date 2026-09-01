"""Unit tests for the one-place traits_hash helper (S3-B7)."""

from __future__ import annotations

from dataclasses import dataclass

from app.traits_hash import compute_traits_hash


@dataclass
class FakeTrait:
    id: str
    category: str = "interest"
    label: str = "hiking"
    description: str = "hikes a lot"
    status: str = "inferred"
    confidence: float = 0.8


def test_deterministic_and_order_independent():
    a, b = FakeTrait("1"), FakeTrait("2", label="cooking")
    assert compute_traits_hash([a, b]) == compute_traits_hash([b, a])


def test_any_consumed_field_changes_the_hash():
    base = compute_traits_hash([FakeTrait("1")])
    assert compute_traits_hash([FakeTrait("1", label="x")]) != base
    assert compute_traits_hash([FakeTrait("1", status="confirmed")]) != base
    assert compute_traits_hash([FakeTrait("1", confidence=0.9)]) != base


def test_retracted_traits_leave_the_active_set():
    active_only = compute_traits_hash([FakeTrait("1")])
    with_retracted = compute_traits_hash([FakeTrait("1"), FakeTrait("2", status="retracted")])
    # A retracted row is not part of what downstream consumes...
    assert with_retracted == active_only
    # ...but RETRACTING an active row changes the hash (the set shrank).
    assert compute_traits_hash([FakeTrait("1", status="retracted")]) != active_only
