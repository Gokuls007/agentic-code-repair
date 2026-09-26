"""Anthropic Messages API provider (tool use)."""

from __future__ import annotations

import time
from typing import Any

import anthropic

from repair_agent.config import LLMSettings
from repair_agent.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "stop_sequence": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "refusal": StopReason.REFUSAL,
}
# 408 timeout, 409 conflict, 429 rate limit; all 5xx (incl. 529 overloaded) are retryable too.
_RETRYABLE_STATUS = frozenset({408, 409, 429})


class AnthropicProvider(LLMProvider):
    """Calls Claude via the official ``anthropic`` SDK."""

    name = "anthropic"

    def __init__(self, settings: LLMSettings, api_key: str, client: Any | None = None):
        """Create the provider. ``client`` may be injected for testing."""
        self._settings = settings
        # SDK retries are off: the agent retries itself so every attempt is traced
        # and backoff respects the task's wall-clock budget.
        self._client = client or anthropic.Anthropic(
            api_key=api_key,
            timeout=settings.request_timeout_s,
            max_retries=0,
        )

    def build_params(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """The exact ``messages.create`` keyword arguments for this request."""
        system_block: dict[str, Any] = {"type": "text", "text": system}
        params: dict[str, Any] = {
            "model": self._settings.model,
            "max_tokens": max_output_tokens or self._settings.max_output_tokens,
            "system": [system_block],
            "messages": [self._to_wire(m) for m in messages],
        }
        if tools:
            params["tools"] = [t.model_dump() for t in tools]
            if self._settings.tool_choice_for(self.name) == "required":
                # Anthropic's "must call some tool". Not accepted by every model (e.g. with
                # thinking on); the default for this provider is therefore "auto".
                params["tool_choice"] = {"type": "any"}
        if self._settings.effort:
            params["output_config"] = {"effort": self._settings.effort}
        if self._settings.temperature is not None:
            # SDK 1.x dropped the argument; send it raw and let the API accept or reject it.
            params["extra_body"] = {"temperature": self._settings.temperature}
        if self._settings.prompt_caching:
            # Tools render before system, so this marker caches tools + system prompt.
            system_block["cache_control"] = {"type": "ephemeral"}
            # Automatic breakpoint on the last block caches the growing history.
            params["cache_control"] = {"type": "ephemeral"}
        return params

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Send one Messages API request and normalise the reply."""
        params = self.build_params(
            system=system, messages=messages, tools=tools, max_output_tokens=max_output_tokens
        )
        if timeout_s is not None:
            params["timeout"] = timeout_s

        started = time.perf_counter()
        try:
            response = self._client.messages.create(**params)
        except anthropic.APIStatusError as exc:
            status = exc.status_code
            retryable = status in _RETRYABLE_STATUS or status >= 500
            raise LLMError(
                str(exc),
                retryable=retryable,
                status_code=status,
                retry_after_s=_retry_after(exc),
            ) from exc
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise LLMError(str(exc), retryable=True) from exc
        latency = time.perf_counter() - started

        return LLMResponse(
            message=self._from_wire(response.content),
            stop_reason=_STOP_REASONS.get(response.stop_reason or "", StopReason.OTHER),
            usage=_usage(response.usage),
            model=response.model,
            latency_s=latency,
        )

    def _to_wire(self, message: Message) -> dict[str, Any]:
        """Convert a neutral message to Anthropic's ``MessageParam`` shape."""
        if message.role == "assistant" and message.provider == self.name and message.provider_raw:
            return {"role": "assistant", "content": message.provider_raw}

        blocks: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolCall):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    }
                )
            elif isinstance(block, ToolResult):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_call_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    }
                )
        return {"role": message.role, "content": blocks}

    def _from_wire(self, content: list[Any]) -> Message:
        """Convert response content blocks to a neutral assistant message.

        Thinking blocks are not surfaced as neutral content but are kept in
        ``provider_raw`` so they can be replayed unchanged on the next turn.
        """
        blocks: list[TextBlock | ToolCall] = []
        for block in content:
            if block.type == "text":
                blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                blocks.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        raw = [b.model_dump(mode="json", exclude_none=True) for b in content]
        return Message(role="assistant", content=blocks, provider=self.name, provider_raw=raw)


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    """Seconds from a numeric ``retry-after`` header, if the server sent one."""
    value = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _usage(usage: Any) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
    )
