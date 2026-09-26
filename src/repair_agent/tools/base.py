"""Tool contract and registry.

Every tool call returns a :class:`ToolOutput`; the registry never raises. Expected
failures (bad path, no match, ...) become ``is_error=True`` results with an actionable
message, so the model can correct itself instead of the loop crashing.
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from repair_agent.llm.base import INVALID_JSON_KEY, ToolCall, ToolResult, ToolSpec


class ToolError(Exception):
    """An expected tool failure. The message is shown to the model verbatim.

    ``metadata`` is passed through to the ToolOutput for the loop (never shown to the model).
    """

    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.metadata = metadata or {}


class ToolArgs(BaseModel):
    """Base for tool argument models: unknown arguments are rejected."""

    model_config = ConfigDict(extra="forbid")


class ToolOutput(BaseModel):
    """Result of one tool call.

    ``content`` is exactly what the model sees. ``metadata`` is for the agent loop and
    tracing only and is never shown to the model.
    """

    content: str
    is_error: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_result(self, tool_call_id: str) -> ToolResult:
        """Convert to the neutral LLM ``ToolResult`` block."""
        return ToolResult(tool_call_id=tool_call_id, content=self.content, is_error=self.is_error)


class Tool(ABC):
    """One capability exposed to the model."""

    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[ToolArgs]]

    def spec(self) -> ToolSpec:
        """The tool definition sent to the LLM."""
        schema = self.Args.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, input_schema=schema)

    @abstractmethod
    def run(self, args: Any) -> ToolOutput:
        """Execute with validated ``args``. Raise ToolError for expected failures."""


def truncate_middle(text: str, limit: int) -> tuple[str, bool]:
    """Keep the first third and last two thirds of ``limit`` chars, noting the cut.

    The tail gets more room because the end of long outputs (test summaries,
    tracebacks) is usually the most informative part.
    """
    if len(text) <= limit:
        return text, False
    head = limit // 3
    tail = limit - head
    omitted = len(text) - head - tail
    note = (
        f"\n[... output truncated: {omitted:,} of {len(text):,} chars omitted. "
        "Narrow the request to see more. ...]\n"
    )
    return text[:head] + note + text[-tail:], True


class ToolRegistry:
    """Dispatches tool calls by name, validates arguments, and applies truncation."""

    def __init__(self, tools: list[Tool], max_output_chars: int):
        self._tools = {tool.name: tool for tool in tools}
        self._max_output_chars = max_output_chars

    @property
    def names(self) -> list[str]:
        """Registered tool names, in registration order."""
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        """Tool definitions for the LLM request."""
        return [tool.spec() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolOutput:
        """Run one tool call. Never raises; failures come back as ``is_error`` outputs."""
        tool = self._tools.get(call.name)
        if tool is None:
            return _error(f"unknown tool {call.name!r}. Available: {', '.join(self.names)}")
        if INVALID_JSON_KEY in call.arguments:
            raw = str(call.arguments[INVALID_JSON_KEY])[:200]
            return _error(f"arguments for {call.name} were not valid JSON: {raw!r}")
        try:
            args = tool.Args.model_validate(call.arguments)
        except ValidationError as exc:
            return _error(f"invalid arguments for {call.name}: {_format_validation(exc)}")
        try:
            output = tool.run(args)
        except ToolError as exc:
            return _error(str(exc), metadata=exc.metadata)
        except Exception as exc:
            return _error(
                f"internal tool failure ({type(exc).__name__}: {exc})",
                metadata={"traceback": traceback.format_exc()},
            )
        content, cut = truncate_middle(output.content, self._max_output_chars)
        return output.model_copy(update={"content": content, "truncated": output.truncated or cut})


def _error(message: str, metadata: dict[str, Any] | None = None) -> ToolOutput:
    return ToolOutput(content=f"Error: {message}", is_error=True, metadata=metadata or {})


def _format_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)
