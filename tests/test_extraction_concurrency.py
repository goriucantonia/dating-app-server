"""Unit tests for the extraction give-up (S6-B5, §17, Step 6 AC6).

Why these are unit tests and not only a live probe: the property under test is
"never two concurrent runs, and never a pile-up of queued ones". Proving that
live requires forcing a real race against a real model call, which makes the
witness depend on model latency and free-tier quota — it fails for reasons that
have nothing to do with the property. Here the "model call" is a controlled
sleep, so the race is deterministic and the assertion is about the lock alone.

The live half is still witnessed in the api logs: a second request arriving
during a real run logs `extraction_queued` with `newly_queued: true` and never
a second `extraction_start`.
"""

from __future__ import annotations

import asyncio

import pytest

from app import extraction


@pytest.fixture(autouse=True)
def _clean_state():
    """The lock table and queued set are module-level (the named trade in
    extraction.py: one worker is the whole system this phase). Tests must not
    inherit each other's state."""
    extraction._locks.clear()
    extraction._queued.clear()
    yield
    extraction._locks.clear()
    extraction._queued.clear()


USER = __import__("uuid").UUID("11111111-1111-1111-1111-111111111111")


async def test_second_request_during_a_run_is_queued_not_run(monkeypatch):
    starts = 0

    async def fake_run(session, router, user_id):
        nonlocal starts
        starts += 1
        await asyncio.sleep(0.05)
        return extraction.ExtractionOutcome(kept=1)

    monkeypatch.setattr(extraction, "run_extraction", fake_run)

    first, second = await asyncio.gather(
        extraction.extract_once(None, None, USER),
        # Yield first so the race is the real one: request B arrives while A
        # holds the lock, rather than before A has taken it.
        _after(0.01, extraction.extract_once(None, None, USER)),
    )

    assert second is None, "the second caller must be told 'queued', not given a result"
    assert first is not None
    # Two runs happened, but SEQUENTIALLY: the original, then its one follow-up.
    assert starts == 2


async def test_many_requests_during_one_run_collapse_into_one_follow_up(monkeypatch):
    starts = 0

    async def fake_run(session, router, user_id):
        nonlocal starts
        starts += 1
        await asyncio.sleep(0.05)
        return extraction.ExtractionOutcome(kept=1)

    monkeypatch.setattr(extraction, "run_extraction", fake_run)

    results = await asyncio.gather(
        extraction.extract_once(None, None, USER),
        *[_after(0.01, extraction.extract_once(None, None, USER)) for _ in range(5)],
    )

    assert sum(r is None for r in results) == 5, "all five late callers are queued"
    # The whole point of §17: five requests do not become five runs. One
    # original plus exactly ONE follow-up, no matter how many piled up.
    assert starts == 2


async def test_a_later_request_after_the_run_finished_runs_normally(monkeypatch):
    starts = 0

    async def fake_run(session, router, user_id):
        nonlocal starts
        starts += 1
        return extraction.ExtractionOutcome(kept=1)

    monkeypatch.setattr(extraction, "run_extraction", fake_run)

    assert await extraction.extract_once(None, None, USER) is not None
    assert starts == 1
    # The give-up is not sticky: once nothing is in flight, the next request is
    # an ordinary run again.
    assert await extraction.extract_once(None, None, USER) is not None
    assert starts == 2


async def test_queue_flag_is_consumed_exactly_once():
    assert extraction.queue_follow_up(USER) is True
    assert extraction.queue_follow_up(USER) is False, "already queued, collapses"
    assert extraction.take_queued(USER) is True
    assert extraction.take_queued(USER) is False, "consumed; nothing left owed"


async def _after(delay: float, coro):
    await asyncio.sleep(delay)
    return await coro
