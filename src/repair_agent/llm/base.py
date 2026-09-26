"""Provider-agnostic LLM interface.

The agent loop only ever sees the types defined here. Each provider translates them
to and from its own wire format, so swapping Anthropic for Groq (or anything else
that supports tool calling) never touches agent code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# Providers that receive tool arguments as a JSON string put unparseable input under this key,
# so the tool registry can tell the model its JSON was malformed.
INVALID_JSON_KEY = "_invalid_json"


class TextBlock(BaseModel):
    """Plain text produced by the model or the user."""

    type: Literal["text"] = "text"
    text: str


class ToolCall(BaseModel):
    """A request from the model to run a tool."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of running a tool, sent back to the model."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[TextBlock | ToolCall | ToolResult, Field(discriminator="type")]


class Message(BaseModel):
    """One conversation turn.

    ``provider_raw`` optionally holds the provider's native content blocks for an
    assistant turn. Some providers require blocks the neutral format does not model
    (e.g. Anthropic thinking blocks with signatures) to be replayed verbatim; a
    provider uses ``provider_raw`` only when ``provider`` matches its own name.
    """

    role: Literal["user", "assistant"]
    content: list[ContentBlock]
    provider: str | None = None
    provider_raw: list[dict[str, Any]] | None = None

    @property
    def text(self) -> str:
        """Concatenated text blocks (the model's visible 'thought')."""
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Tool calls requested in this turn, in order."""
        return [b for b in self.content if isinstance(b, ToolCall)]


class ToolSpec(BaseModel):
    """A tool the model may call, described by a JSON Schema for its arguments."""

    name: str
    description: str
    input_schema: dict[str, Any]


class StopReason(StrEnum):
    """Why the model stopped generating, normalised across providers."""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"


class Usage(BaseModel):
    """Token accounting for a single request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )

    @property
    def total_tokens(self) -> int:
        """All tokens billed for this request (input incl. cache, plus output)."""
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
            + self.output_tokens
        )


class LLMResponse(BaseModel):
    """A single model completion in neutral form."""

    message: Message
    stop_reason: StopReason
    usage: Usage
    model: str
    latency_s: float


class LLMError(RuntimeError):
    """A provider call failed. ``retryable`` hints whether trying again may help."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        retry_after_s: float | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_s = retry_after_s


class LLMProvider(ABC):
    """Interface every LLM backend implements."""

    name: str

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Send the conversation and return the model's next turn.

        ``timeout_s`` caps this one request (e.g. to the task's remaining wall-clock time).

        Raises:
            LLMError: if the provider call fails after the client's own retries.
        """
