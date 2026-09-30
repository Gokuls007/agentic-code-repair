"""Groq provider (OpenAI-compatible chat completions with function calling).

Differences from the Anthropic provider that matter to the agent:

- No explicit prompt caching: nothing cache-related is ever sent. Groq may cache
  prompts automatically on some models; any ``cached_tokens`` it reports are recorded
  as cache reads so token accounting stays complete.
- Free-tier rate limits count prompt + ``max_completion_tokens`` against a per-minute
  token budget, and one request above it fails with 413. ``max_completion_tokens`` is
  therefore sized to fit under ``groq_tpm_limit``.
- Unparseable model output comes back as HTTP 400 ``tool_use_failed`` or
  ``output_parse_failed``; that is a sampling failure, so it is marked retryable.
"""

from __future__ import annotations

import json
import time
from typing import Any

import groq

from repair_agent.config import LLMSettings
from repair_agent.llm.base import (
    INVALID_JSON_KEY,
    NO_TOOL_CALL,
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
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
}
_RETRYABLE_STATUS = frozenset({408, 409, 429})
# 400s meaning the model's sampled output was unparseable (a malformed tool call, or text
# Groq could not parse). Resampling usually succeeds, so these are retried.
_BAD_SAMPLE_CODES = frozenset({"tool_use_failed", "output_parse_failed"})
# Our effort levels -> Groq's reasoning_effort (only sent for gpt-oss models).
_REASONING_EFFORT = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
# Rough chars-per-token for estimating prompt size before sending (errs on the high side).
_CHARS_PER_TOKEN = 3.2
# A 413 caused by the shared per-minute window clears once the window rolls over.
_TPM_WINDOW_RETRY_S = 20.0


class GroqProvider(LLMProvider):
    """Calls Groq via the official ``groq`` SDK.

    The wire format is OpenAI chat completions, so :class:`OpenAICompatibleProvider`
    reuses everything here and only swaps the client and its exception types.
    """

    name = "groq"
    _status_error: type[Exception] = groq.APIStatusError
    _connection_error: type[Exception] = groq.APIConnectionError  # includes timeouts
    _tpm_setting = "REPAIR_LLM__GROQ_TPM_LIMIT"

    def __init__(
        self,
        settings: LLMSettings,
        api_key: str,
        client: Any | None = None,
        *,
        name: str | None = None,
        send_reasoning_effort: bool = True,
    ):
        """Create the provider. ``client`` may be injected for testing; ``name`` labels
        this backend in traces and results (it is "groq" unless running inside a pool)."""
        self._settings = settings
        if name:
            self.name = name
        self._send_reasoning_effort = send_reasoning_effort
        # SDK retries off: the agent's retry layer handles backoff and traces attempts.
        self._client = client or self._make_client(settings, api_key)

    def _make_client(self, settings: LLMSettings, api_key: str) -> Any:
        return groq.Groq(api_key=api_key, timeout=settings.request_timeout_s, max_retries=0)

    def build_params(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """The exact ``chat.completions.create`` keyword arguments for this request."""
        s = self._settings
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for message in messages:
            wire.extend(self._to_wire(message))
        params: dict[str, Any] = {"model": s.model, "messages": wire}
        if tools:
            params["tools"] = [_tool(t) for t in tools]
            # "required": the model must call a tool every turn (it stops via finish).
            params["tool_choice"] = s.tool_choice_for(self.name)
        if s.temperature is not None:
            params["temperature"] = s.temperature
        if s.effort and self._send_reasoning_effort and "gpt-oss" in s.model:
            params["reasoning_effort"] = _REASONING_EFFORT[s.effort]
        params["max_completion_tokens"] = self._completion_budget(
            params, max_output_tokens or s.max_output_tokens
        )
        return params

    def _completion_budget(self, params: dict[str, Any], requested: int) -> int:
        """Shrink max_completion_tokens so prompt + completion fits the TPM limit."""
        limit = self._settings.groq_tpm_limit
        if limit is None:
            return requested
        prompt_estimate = estimate_tokens(params)
        room = limit - prompt_estimate
        if room < self._settings.min_output_tokens:
            raise LLMError(
                f"prompt is ~{prompt_estimate:,} tokens, leaving under "
                f"{self._settings.min_output_tokens:,} for the reply within the "
                f"{limit:,} tokens/minute limit ({self._tpm_setting})",
                retryable=False,
                status_code=413,
            )
        return min(requested, room)

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Send one chat completion request and normalise the reply."""
        params = self.build_params(
            system=system, messages=messages, tools=tools, max_output_tokens=max_output_tokens
        )
        if timeout_s is not None:
            params["timeout"] = timeout_s

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**params)
        except self._status_error as exc:
            raise _to_llm_error(exc) from exc
        except self._connection_error as exc:
            raise LLMError(str(exc), retryable=True) from exc
        latency = time.perf_counter() - started

        choice = response.choices[0]
        return LLMResponse(
            message=self._from_wire(choice.message),
            stop_reason=_STOP_REASONS.get(choice.finish_reason or "", StopReason.OTHER),
            usage=_usage(response.usage),
            model=response.model,
            latency_s=latency,
            provider=self.name,
        )

    def _to_wire(self, message: Message) -> list[dict[str, Any]]:
        """Neutral message -> one or more OpenAI-style chat messages.

        A user turn holding tool results becomes one ``tool`` message per result
        (which must directly follow the assistant's tool calls), then a ``user``
        message for any accompanying text (e.g. the budget line).
        """
        if message.role == "assistant":
            out: dict[str, Any] = {"role": "assistant", "content": message.text or None}
            calls = message.tool_calls
            if calls:
                out["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in calls
                ]
            return [out]

        wire: list[dict[str, Any]] = []
        texts: list[str] = []
        for block in message.content:
            if isinstance(block, ToolResult):
                wire.append(
                    {"role": "tool", "tool_call_id": block.tool_call_id, "content": block.content}
                )
            elif isinstance(block, TextBlock):
                texts.append(block.text)
        if texts:
            wire.append({"role": "user", "content": "\n\n".join(texts)})
        return wire

    def _from_wire(self, msg: Any) -> Message:
        blocks: list[TextBlock | ToolCall] = []
        if msg.content:
            blocks.append(TextBlock(text=msg.content))
        for call in msg.tool_calls or []:
            blocks.append(
                ToolCall(id=call.id, name=call.function.name, arguments=_parse_args(call))
            )
        return Message(role="assistant", content=blocks, provider=self.name)


def _tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _parse_args(call: Any) -> dict[str, Any]:
    """Tool-call arguments arrive as a JSON string; keep bad JSON visible to the registry."""
    raw = call.function.arguments or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {INVALID_JSON_KEY: raw}
    return parsed if isinstance(parsed, dict) else {INVALID_JSON_KEY: raw}


def estimate_tokens(params: dict[str, Any]) -> int:
    """Conservative prompt-size estimate from the serialised messages and tools."""
    payload = json.dumps({"m": params.get("messages", []), "t": params.get("tools", [])})
    return int(len(payload) / _CHARS_PER_TOKEN) + 1


def _error_code(exc: Any) -> str | None:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body)
    return error.get("code") if isinstance(error, dict) else None


def _to_llm_error(exc: Any) -> LLMError:
    status = exc.status_code
    code = _error_code(exc)
    retry_after = _retry_after(exc)
    message = str(exc)
    if status == 400 and code == "tool_use_failed" and "did not call a tool" in message:
        # tool_choice=required, but the model answered in text (typically a final summary).
        # Resampling the same prompt tends to give the same answer; the agent loop handles
        # this like a text-only turn and asks the model to call finish.
        return LLMError(
            f"model answered without a tool call: {message}",
            retryable=False,
            status_code=400,
            kind=NO_TOOL_CALL,
            generated_text=_failed_generation(exc),
        )
    if status == 400 and code in _BAD_SAMPLE_CODES:
        return LLMError(
            f"model output could not be parsed ({code}): {message}",
            retryable=True,
            status_code=400,
        )
    if status == 413:
        # Over the per-minute window (retryable once it rolls over) vs. simply too big.
        per_minute = "per minute" in message.lower() or "tpm" in message.lower()
        return LLMError(
            message,
            retryable=per_minute,
            status_code=413,
            retry_after_s=retry_after or (_TPM_WINDOW_RETRY_S if per_minute else None),
        )
    retryable = status in _RETRYABLE_STATUS or status >= 500
    return LLMError(message, retryable=retryable, status_code=status, retry_after_s=retry_after)


def _failed_generation(exc: Any) -> str | None:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body)
    text = error.get("failed_generation") if isinstance(error, dict) else None
    return text if isinstance(text, str) and text.strip() else None


def _retry_after(exc: Any) -> float | None:
    value = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _usage(usage: Any) -> Usage:
    """Map OpenAI-style usage. prompt_tokens includes any cached tokens; split them out."""
    if usage is None:
        return Usage()
    prompt = usage.prompt_tokens or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
    return Usage(
        input_tokens=prompt - cached,
        output_tokens=usage.completion_tokens or 0,
        cache_read_input_tokens=cached,
    )
