"""Append-only JSONL trace of every agent step.

Each task run writes ``<runs_dir>/<run_id>/<task_id>.jsonl``. One line per event, flushed
immediately so a crashed or killed run still leaves a readable trace. All string content
is scrubbed for secrets before it touches disk.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from pydantic import BaseModel, Field

REDACTED = "[REDACTED]"

# Well-known credential shapes, redacted even if not present in config.
_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),  # Anthropic
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),  # Groq
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub classic tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),  # GitHub fine-grained tokens
]

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")


class EventKind(StrEnum):
    """Kinds of trace events emitted by the agent."""

    TASK_START = "task_start"
    LLM_CALL = "llm_call"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    TASK_END = "task_end"


class TraceEvent(BaseModel):
    """One line in a trace file."""

    step: int = Field(ge=0)
    kind: EventKind
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)
    tokens_in: int | None = None
    tokens_out: int | None = None


def new_run_id(now: datetime | None = None) -> str:
    """A sortable, collision-resistant run id such as ``20260925-141503-a1b2c3``."""
    now = now or datetime.now(UTC)
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


class Redactor:
    """Replaces known secret values and credential-shaped strings with ``[REDACTED]``."""

    def __init__(self, secret_values: list[str] | None = None):
        # Longest first so a secret that contains another is fully replaced.
        self._values = sorted({v for v in secret_values or [] if v}, key=len, reverse=True)

    def scrub(self, value: Any) -> Any:
        """Return a copy of ``value`` with secrets removed from every nested string."""
        if isinstance(value, str):
            for secret in self._values:
                value = value.replace(secret, REDACTED)
            for pattern in _SECRET_PATTERNS:
                value = pattern.sub(REDACTED, value)
            return value
        if isinstance(value, dict):
            return {k: self.scrub(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self.scrub(v) for v in value]
        return value


class Tracer:
    """Writes trace events for one task. Use as a context manager."""

    def __init__(
        self,
        runs_dir: Path,
        run_id: str,
        task_id: str,
        *,
        secret_values: list[str] | None = None,
    ):
        for label, ident in (("run_id", run_id), ("task_id", task_id)):
            if not _SAFE_ID.match(ident):
                raise ValueError(f"{label} must be filesystem-safe, got {ident!r}")
        self.path = Path(runs_dir) / run_id / f"{task_id}.jsonl"
        self._redactor = Redactor(secret_values)
        self._file: IO[str] | None = None
        self._step = 0

    @property
    def step(self) -> int:
        """The current step number (incremented by :meth:`next_step`)."""
        return self._step

    def next_step(self) -> int:
        """Advance to the next agent iteration and return its number."""
        self._step += 1
        return self._step

    def open(self) -> Tracer:
        """Create the run directory and open the trace file for appending."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        return self

    def close(self) -> None:
        """Close the trace file. Safe to call more than once."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> Tracer:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def log(
        self,
        kind: EventKind,
        data: dict[str, Any] | None = None,
        *,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> TraceEvent:
        """Record one event at the current step and flush it to disk."""
        if self._file is None:
            raise RuntimeError("Tracer is not open; use it as a context manager")
        event = TraceEvent(
            step=self._step,
            kind=kind,
            timestamp=datetime.now(UTC),
            data=self._redactor.scrub(data or {}),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        self._file.write(event.model_dump_json() + "\n")
        self._file.flush()
        return event


def read_trace(path: Path) -> list[TraceEvent]:
    """Load every event from a trace file, in order."""
    with Path(path).open(encoding="utf-8") as fh:
        return [TraceEvent.model_validate(json.loads(line)) for line in fh if line.strip()]
