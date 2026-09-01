"""Resilience layer (S2-B7): backoff, per-provider rate limiting, capped
retries, and the mandatory per-call log line (S2-B8, ai_interaction.md §5).

Free tiers throttle aggressively — the wire dictates this design (§15).

Logging contract:
- `ai_call` — ONE line per provider call with its FINAL outcome
  (task, provider, model, attempts, latency_ms, outcome).
- `ai_call_retry` — one line per intermediate failed attempt, with the error
  and the backoff delay, so a retry storm is visible while it happens.

The Guard in structured.py adds its own `ai_call` lines with outcomes
`malformed` / `gave_up`, which only exist after validation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.ai.base import RateLimitedError, RefusedError, TransientAIError
from app.logging_setup import log_event

logger = logging.getLogger("app.ai")

T = TypeVar("T")

# Give-up after 3 attempts (§17) — committed in the same module as the loop.
MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 2.0
_BACKOFF_CAP_S = 30.0


class RateLimiter:
    """Minimum spacing between calls to one provider. Coarse by design — the
    published per-minute caps arrive with the model choice at the quota gate;
    until then this just keeps bursts polite."""

    def __init__(self, min_interval_s: float = 1.0):
        self._min_interval = min_interval_s
        self._lock = asyncio.Lock()
        self._next_free = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_free - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_free = max(now, self._next_free) + self._min_interval


def _backoff_delay(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, _BACKOFF_CAP_S)
    return min(_BACKOFF_BASE_S * (2 ** (attempt - 1)), _BACKOFF_CAP_S)


async def execute(
    fn: Callable[[], Awaitable[T]],
    *,
    task: str,
    provider: str,
    model: str,
    limiter: RateLimiter,
) -> T:
    """Run one provider call with rate limiting, backoff on 429/5xx, and the
    mandatory log line. Raises the typed error after the capped retries."""
    started = time.monotonic()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        await limiter.acquire()
        try:
            result = await fn()
        except (RateLimitedError, TransientAIError) as exc:
            outcome = "rate_limited" if isinstance(exc, RateLimitedError) else "gave_up"
            if attempt == MAX_ATTEMPTS:
                # Final failure: the one mandatory line, then the typed error out.
                log_event(
                    logger, "ai_call", level=logging.WARNING,
                    task=task, provider=provider, model=model, attempts=attempt,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    outcome=outcome,
                    error=str(exc),
                )
                raise
            delay = _backoff_delay(
                attempt, exc.retry_after if isinstance(exc, RateLimitedError) else None
            )
            log_event(
                logger, "ai_call_retry", level=logging.WARNING,
                task=task, provider=provider, model=model, attempt=attempt,
                error=str(exc), backoff_s=round(delay, 1),
            )
            await asyncio.sleep(delay)
        except RefusedError as exc:
            # Refusals are never retried — same content, same refusal.
            log_event(
                logger, "ai_call", level=logging.WARNING,
                task=task, provider=provider, model=model, attempts=attempt,
                latency_ms=int((time.monotonic() - started) * 1000),
                outcome="refused", error=str(exc),
            )
            raise
        else:
            log_event(
                logger, "ai_call",
                task=task, provider=provider, model=model, attempts=attempt,
                latency_ms=int((time.monotonic() - started) * 1000), outcome="ok",
            )
            return result
    raise AssertionError("unreachable: the retry loop returns or raises")
