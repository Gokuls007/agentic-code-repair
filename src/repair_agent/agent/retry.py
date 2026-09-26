"""Retry LLM calls with exponential backoff and full jitter."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TypeVar

from repair_agent.llm.base import LLMError

T = TypeVar("T")


def backoff_delay(
    attempt: int,
    *,
    base_s: float,
    cap_s: float,
    retry_after_s: float | None = None,
    rng: Callable[[], float] = random.random,
) -> float:
    """Delay before retry number ``attempt`` (0-based).

    Full jitter: uniform in [0, min(cap, base * 2**attempt)]. A server-provided
    ``retry-after`` is a floor, since retrying earlier is guaranteed to fail.
    """
    delay = min(cap_s, base_s * (2**attempt)) * rng()
    if retry_after_s is not None:
        delay = max(delay, retry_after_s)
    return delay


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_s: float,
    cap_s: float,
    time_left: Callable[[], float],
    sleep: Callable[[float], None],
    on_retry: Callable[[int, LLMError, float], None] | None = None,
    rng: Callable[[], float] = random.random,
) -> T:
    """Call ``fn``, retrying retryable :class:`LLMError`s.

    Gives up (re-raising the last error) when the error is not retryable, retries are
    exhausted, or the next wait would run past the wall-clock deadline.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except LLMError as exc:
            if not exc.retryable or attempt >= max_retries:
                raise
            delay = backoff_delay(
                attempt, base_s=base_s, cap_s=cap_s, retry_after_s=exc.retry_after_s, rng=rng
            )
            if delay >= time_left():
                raise
            if on_retry:
                on_retry(attempt + 1, exc, delay)
            sleep(delay)
            attempt += 1
