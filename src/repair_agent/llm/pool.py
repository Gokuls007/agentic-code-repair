"""A provider that balances requests across several backends serving the same model.

Free tiers cap tokens per minute and per day, so one key finishes only a handful of
eval attempts a day. A pool puts several endpoints for the *same* model behind one
provider (e.g. gpt-oss-120b on Groq and on Cerebras):

- ``round_robin`` rotates the first backend tried on every request, spreading
  per-minute limits; ``failover`` always tries backends in configured order.
- A backend whose quota is exhausted (429 with a long or daily retry, 402), whose key is
  rejected (401/403), or that does not serve the model (404) is **benched**: skipped
  until its cooldown ends (for the rest of the process for key/model errors).
- A transient failure (5xx, connection error, short 429) benches the backend briefly
  and the same request goes to the next one immediately.
- A prompt too large for one backend's per-minute cap (a non-retryable 413) is tried on
  the next backend, which may have a larger cap.
- Errors about the request itself (400s, a required tool call not made) are raised at
  once: another backend serving the same model would answer the same way.
- When every backend is benched, the error is retryable with ``retry_after_s`` set to
  the earliest cooldown end. The retry layer then gives up if that wait is long, and the
  eval runner records the attempt as infrastructure and stops, as for a single provider.

Messages are provider-neutral, so switching backends mid-attempt is safe for
OpenAI-style backends (they keep no provider-specific state between turns).
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from repair_agent.config import LLMSettings
from repair_agent.llm.base import (
    NO_TOOL_CALL,
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    ToolSpec,
)

# LLMError.kind when every backend is benched.
POOL_EXHAUSTED = "pool_exhausted"

_DAILY = re.compile(
    r"per[ -]day|daily|tokens per day|requests per day|\bTPD\b|\bRPD\b|quota|credit",
    re.IGNORECASE,
)
# Below this, a per-backend request is not worth sending (the caller's timeout is nearly spent).
_MIN_REQUEST_S = 1.0


@dataclass
class Backend:
    name: str
    provider: LLMProvider
    free_tier: bool = True
    benched_until: float = 0.0
    bench_reason: str | None = None
    # HTTP status that caused the bench (None = connection error); only 429/402 is a quota.
    bench_status: int | None = None
    requests: int = 0
    failures: int = 0


class PoolProvider(LLMProvider):
    """Routes each request to one backend of a pool; see the module docstring."""

    name = "pool"

    def __init__(
        self,
        backends: list[Backend],
        settings: LLMSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not backends:
            raise ValueError("a provider pool needs at least one backend")
        self.backends = backends
        self._settings = settings
        self._clock = clock
        self._next = 0
        self._lock = threading.Lock()

    @property
    def free_backends(self) -> set[str]:
        return {b.name for b in self.backends if b.free_tier}

    def status(self) -> list[dict[str, object]]:
        """Per-backend counters and bench state (for logs and the CLI)."""
        now = self._clock()
        return [
            {
                "name": b.name,
                "requests": b.requests,
                "failures": b.failures,
                "benched_for_s": (
                    None
                    if b.benched_until <= now
                    else math.inf
                    if math.isinf(b.benched_until)
                    else round(b.benched_until - now, 1)
                ),
                "bench_reason": b.bench_reason if b.benched_until > now else None,
            }
            for b in self.backends
        ]

    def _order(self) -> list[Backend]:
        with self._lock:
            if self._settings.pool_strategy == "round_robin":
                start = self._next % len(self.backends)
                self._next += 1
            else:
                start = 0
        return self.backends[start:] + self.backends[:start]

    def _bench(self, backend: Backend, seconds: float, reason: str, status: int | None) -> None:
        with self._lock:
            backend.benched_until = max(backend.benched_until, self._clock() + seconds)
            backend.bench_reason = reason
            backend.bench_status = status

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        started = self._clock()
        available = [b for b in self._order() if b.benched_until <= started]
        if not available:
            raise self._exhausted_error()

        failed: list[dict[str, object]] = []
        errors: list[tuple[Backend, LLMError]] = []
        for backend in available:
            remaining = None if timeout_s is None else timeout_s - (self._clock() - started)
            if remaining is not None and remaining < _MIN_REQUEST_S:
                break
            backend.requests += 1
            try:
                response = backend.provider.complete(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_output_tokens=max_output_tokens,
                    timeout_s=remaining,
                )
            except LLMError as exc:
                backend.failures += 1
                if _about_the_request(exc):
                    raise
                self._handle_failure(backend, exc)
                failed.append(
                    {"backend": backend.name, "status": exc.status_code, "error": str(exc)[:300]}
                )
                errors.append((backend, exc))
                continue
            response.provider = backend.name
            response.fallbacks = failed
            return response

        raise self._combined_error(errors)

    def _handle_failure(self, backend: Backend, exc: LLMError) -> None:
        s = self._settings
        status = exc.status_code
        if status in (401, 403):
            self._bench(backend, math.inf, f"key rejected ({status})", status)
        elif status == 404:
            self._bench(backend, math.inf, "model not found on this backend (404)", status)
        elif status == 402 or (status == 429 and _is_quota(exc, s.max_retry_wait_s)):
            self._bench(
                backend,
                exc.retry_after_s or s.pool_quota_cooldown_s,
                f"quota exhausted ({status})",
                status,
            )
        elif status == 413 and not exc.retryable:
            pass  # too big for this backend's cap; another backend may take it
        else:  # 5xx, connection errors, per-minute 429/413
            self._bench(
                backend,
                exc.retry_after_s or s.pool_transient_cooldown_s,
                f"transient failure ({status or 'connection'})",
                status,
            )

    def _exhausted_error(self) -> LLMError:
        now = self._clock()
        wait = min(b.benched_until for b in self.backends) - now
        reasons = "; ".join(f"{b.name}: {b.bench_reason}" for b in self.backends)
        # Report what actually happened: 429 only if some backend is out of quota or rate
        # limited, otherwise the servers' own status (e.g. 500/503), so traces stay truthful.
        statuses = [b.bench_status for b in self.backends]
        status = (
            429
            if any(s in (402, 429) for s in statuses)
            else next((s for s in statuses if s is not None), 503)
        )
        return LLMError(
            f"every backend in the pool is unavailable ({reasons})",
            retryable=not math.isinf(wait),
            status_code=status,
            retry_after_s=None if math.isinf(wait) else max(0.0, wait),
            kind=POOL_EXHAUSTED,
        )

    def _combined_error(self, errors: list[tuple[Backend, LLMError]]) -> LLMError:
        if not errors:  # ran out of time before any backend answered
            return LLMError("request timed out before a backend answered", retryable=True)
        if all(e.status_code == 413 and not e.retryable for _, e in errors):
            return errors[-1][1]  # too big everywhere: let the loop trim context
        if all(b.benched_until > self._clock() for b in self.backends):
            return self._exhausted_error()
        last = errors[-1][1]
        detail = "; ".join(f"{b.name}: {e}" for b, e in errors)
        return LLMError(
            f"all available backends failed: {detail}"[:2000],
            retryable=any(e.retryable for _, e in errors),
            status_code=last.status_code,
            retry_after_s=last.retry_after_s,
        )


def _about_the_request(exc: LLMError) -> bool:
    """Errors another backend serving the same model would repeat."""
    return exc.kind == NO_TOOL_CALL or exc.status_code == 400


def _is_quota(exc: LLMError, max_wait_s: float) -> bool:
    """A 429 that will not clear within a normal retry wait (daily/credit limits)."""
    if exc.retry_after_s is not None and exc.retry_after_s > max_wait_s:
        return True
    return bool(_DAILY.search(str(exc)))
