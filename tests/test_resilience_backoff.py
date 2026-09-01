"""Unit tests for the retry schedule (S2-B7, revised in Step 11).

The schedule is not a tuning knob, it is a claim about the wire (§15). This
project's providers fail in two different ways and only one of them clears in
seconds:

- a 5xx or a dropped connection is over almost immediately;
- a 429 is a QUOTA WINDOW, and every window this project actually meets is
  per-minute or per-day.

Waiting 2 seconds and then 4 against a per-minute window is three requests
inside the same blocked window pretending to be a retry policy. That is not
hypothetical — it took a whole analysis down in Step 11 when matching embedded
four people within a few seconds of each other and met google's per-minute
embedding cap.
"""

from __future__ import annotations

from app.ai.resilience import (
    _BACKOFF_CAP_S,
    _RATE_LIMIT_BASE_S,
    MAX_ATTEMPTS,
    _backoff_delay,
)


def test_a_transient_failure_retries_almost_immediately():
    assert _backoff_delay(1, None) == 2.0
    assert _backoff_delay(2, None) == 4.0


def test_a_rate_limit_waits_long_enough_to_leave_the_window():
    """The whole point: the first rate-limit retry must not land inside the
    same minute that just rejected the call."""
    assert _backoff_delay(1, None, rate_limited=True) >= _RATE_LIMIT_BASE_S
    assert _backoff_delay(1, None, rate_limited=True) > _backoff_delay(1, None)


def test_the_provider_hint_always_wins():
    """A provider that says when to come back knows better than any schedule
    here — including when that is SOONER than our own backoff."""
    assert _backoff_delay(1, 0.5, rate_limited=True) == 0.5
    assert _backoff_delay(3, 9.0) == 9.0


def test_no_delay_ever_exceeds_the_cap():
    """Including the provider's hint: a provider asking us to wait an hour
    gets a capped wait and then the typed give-up, rather than a request that
    hangs for an hour with a user on the other end."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        assert _backoff_delay(attempt, None) <= _BACKOFF_CAP_S
        assert _backoff_delay(attempt, None, rate_limited=True) <= _BACKOFF_CAP_S
    assert _backoff_delay(1, 3600.0) == _BACKOFF_CAP_S


def test_the_whole_rate_limit_ladder_outlasts_a_per_minute_window():
    """Three attempts have two waits between them. Together they have to be
    worth more than the window that is blocking us, or the give-up is just a
    faster way to fail."""
    total = sum(
        _backoff_delay(a, None, rate_limited=True) for a in range(1, MAX_ATTEMPTS)
    )
    assert total >= 45.0, f"the rate-limit ladder only waits {total}s in total"
