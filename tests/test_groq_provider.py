"""GroqProvider translation, TPM sizing, and error mapping with a fake SDK client (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest
from groq.types.chat import ChatCompletion

from repair_agent.config import LLMSettings
from repair_agent.llm.base import (
    INVALID_JSON_KEY,
    NO_TOOL_CALL,
    LLMError,
    Message,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from repair_agent.llm.groq import GroqProvider, estimate_tokens

READ_TOOL = ToolSpec(
    name="read_file",
    description="Read a file",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
)
USER = Message(role="user", content=[TextBlock(text="fix the bug")])


class FakeCompletions:
    def __init__(self, response: Any = None, error: Exception | None = None):
        self.response, self.error = response, error
        self.calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        self.calls.append(params)
        if self.error:
            raise self.error
        return self.response


def make(fake: FakeCompletions, **settings: Any) -> GroqProvider:
    client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return GroqProvider(LLMSettings(**settings), api_key="unused", client=client)


def completion(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish: str = "stop",
    prompt: int = 900,
    completion_tokens: int = 50,
    cached: int | None = None,
) -> ChatCompletion:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt + completion_tokens,
    }
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "openai/gpt-oss-120b",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                }
            ],
            "usage": usage,
        }
    )


def call_json(cid: str, name: str, args: str) -> dict[str, Any]:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


# --- request shape ----------------------------------------------------------------


def test_request_shape_and_no_caching_params() -> None:
    fake = FakeCompletions(completion(content="ok"))
    make(fake, effort="medium", prompt_caching=True).complete(
        system="be careful", messages=[USER], tools=[READ_TOOL]
    )
    params = fake.calls[0]
    assert params["model"] == "openai/gpt-oss-120b"
    assert params["messages"][0] == {"role": "system", "content": "be careful"}
    assert params["messages"][1] == {"role": "user", "content": "fix the bug"}
    assert params["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": READ_TOOL.input_schema,
            },
        }
    ]
    assert params["tool_choice"] == "required"  # Groq default: the agent stops only via finish
    assert params["reasoning_effort"] == "medium"
    assert "temperature" not in params
    assert "cache_control" not in json.dumps(params)  # prompt caching is Anthropic-only


def test_reasoning_effort_only_for_gpt_oss_and_temperature_when_set() -> None:
    fake = FakeCompletions(completion(content="ok"))
    make(fake, model="qwen/qwen3.8-27b", effort="xhigh", temperature=0.2).complete(
        system="s", messages=[USER], tools=[]
    )
    params = fake.calls[0]
    assert "reasoning_effort" not in params
    assert "tools" not in params and "tool_choice" not in params
    assert params["temperature"] == 0.2


def test_tool_results_become_tool_messages_then_user_text() -> None:
    fake = FakeCompletions(completion(content="done"))
    assistant = Message(
        role="assistant",
        content=[
            TextBlock(text="Reading."),
            ToolCall(id="c1", name="read_file", arguments={"path": "a.py"}),
        ],
        provider="groq",
    )
    results = Message(
        role="user",
        content=[
            ToolResult(tool_call_id="c1", content="1\tx = 1"),
            TextBlock(text="[budget] iteration 1/30"),
        ],
    )
    make(fake).complete(system="s", messages=[USER, assistant, results], tools=[READ_TOOL])
    wire = fake.calls[0]["messages"]
    assert wire[2] == {
        "role": "assistant",
        "content": "Reading.",
        "tool_calls": [call_json("c1", "read_file", '{"path": "a.py"}')],
    }
    assert wire[3] == {"role": "tool", "tool_call_id": "c1", "content": "1\tx = 1"}
    assert wire[4] == {"role": "user", "content": "[budget] iteration 1/30"}


def test_anthropic_raw_blocks_are_ignored() -> None:
    fake = FakeCompletions(completion(content="ok"))
    foreign = Message(
        role="assistant",
        content=[TextBlock(text="hi")],
        provider="anthropic",
        provider_raw=[{"type": "thinking", "signature": "x"}],
    )
    make(fake).complete(system="s", messages=[USER, foreign], tools=[])
    assert fake.calls[0]["messages"][2] == {"role": "assistant", "content": "hi"}


# --- responses ---------------------------------------------------------------------


def test_parses_tool_calls_usage_and_model() -> None:
    fake = FakeCompletions(
        completion(
            tool_calls=[call_json("call_1", "read_file", '{"path": "src/a.py"}')],
            finish="tool_calls",
            prompt=1200,
            completion_tokens=80,
            cached=1000,
        )
    )
    resp = make(fake).complete(system="s", messages=[USER], tools=[READ_TOOL])
    assert resp.stop_reason == StopReason.TOOL_USE
    assert resp.message.tool_calls == [
        ToolCall(id="call_1", name="read_file", arguments={"path": "src/a.py"})
    ]
    assert resp.message.provider == "groq" and resp.message.provider_raw is None
    assert (resp.usage.input_tokens, resp.usage.cache_read_input_tokens) == (200, 1000)
    assert resp.usage.output_tokens == 80
    assert resp.usage.total_tokens == 1280  # prompt_tokens + completion_tokens, nothing lost
    assert resp.model == "openai/gpt-oss-120b"


def test_invalid_tool_json_is_flagged_not_crashed() -> None:
    fake = FakeCompletions(
        completion(tool_calls=[call_json("c", "read_file", '{"path": ')], finish="tool_calls")
    )
    resp = make(fake).complete(system="s", messages=[USER], tools=[READ_TOOL])
    assert resp.message.tool_calls[0].arguments == {INVALID_JSON_KEY: '{"path": '}


@pytest.mark.parametrize(
    ("finish", "expected"),
    [
        ("stop", StopReason.END_TURN),
        ("tool_calls", StopReason.TOOL_USE),
        ("length", StopReason.MAX_TOKENS),
        ("function_call", StopReason.OTHER),
    ],
)
def test_finish_reason_mapping(finish: str, expected: StopReason) -> None:
    fake = FakeCompletions(completion(content="x", finish=finish))
    assert make(fake).complete(system="s", messages=[USER], tools=[]).stop_reason == expected


# --- free-tier TPM sizing ----------------------------------------------------------


def test_completion_budget_is_clamped_to_fit_tpm_limit() -> None:
    fake = FakeCompletions(completion(content="ok"))
    provider = make(fake, max_output_tokens=16000, groq_tpm_limit=8000)
    provider.complete(system="s", messages=[USER], tools=[READ_TOOL])
    params = fake.calls[0]
    assert params["max_completion_tokens"] == 8000 - estimate_tokens(params)


def test_small_requests_keep_the_configured_max() -> None:
    fake = FakeCompletions(completion(content="ok"))
    make(fake, max_output_tokens=2000, groq_tpm_limit=8000).complete(
        system="s", messages=[USER], tools=[]
    )
    assert fake.calls[0]["max_completion_tokens"] == 2000


def test_clamp_disabled_for_paid_tiers() -> None:
    fake = FakeCompletions(completion(content="ok"))
    make(fake, max_output_tokens=16000, groq_tpm_limit=None).complete(
        system="s", messages=[USER], tools=[]
    )
    assert fake.calls[0]["max_completion_tokens"] == 16000


def test_prompt_too_big_for_tpm_raises_non_retryable_413_before_sending() -> None:
    fake = FakeCompletions(completion(content="ok"))
    huge = Message(role="user", content=[TextBlock(text="x" * 40_000)])
    with pytest.raises(LLMError) as info:
        make(fake, groq_tpm_limit=8000).complete(system="s", messages=[huge], tools=[])
    assert info.value.status_code == 413 and not info.value.retryable
    assert fake.calls == []  # never sent


# --- errors --------------------------------------------------------------------------


def status_error(status: int, body: dict[str, Any], headers: dict[str, str] | None = None):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {}, json=body)
    message = body.get("error", {}).get("message", "error")
    return groq.APIStatusError(message, response=response, body=body)


@pytest.mark.parametrize(
    ("status", "body", "headers", "retryable", "retry_after"),
    [
        (
            429,
            {"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}},
            {"retry-after": "12"},
            True,
            12.0,
        ),
        (
            400,
            {"error": {"message": "Failed to call a function", "code": "tool_use_failed"}},
            None,
            True,
            None,
        ),
        (
            400,
            {"error": {"message": "Parsing failed.", "code": "output_parse_failed"}},
            None,
            True,
            None,
        ),
        (
            400,
            {"error": {"message": "bad request", "code": "invalid_request_error"}},
            None,
            False,
            None,
        ),
        (
            413,
            {"error": {"message": "Request too large: tokens per minute (TPM) limit 8000"}},
            None,
            True,
            20.0,
        ),
        (413, {"error": {"message": "Request Entity Too Large"}}, None, False, None),
        (401, {"error": {"message": "Invalid API Key"}}, None, False, None),
        (503, {"error": {"message": "unavailable"}}, None, True, None),
    ],
)
def test_error_mapping(status, body, headers, retryable, retry_after) -> None:
    fake = FakeCompletions(error=status_error(status, body, headers))
    with pytest.raises(LLMError) as info:
        make(fake).complete(system="s", messages=[USER], tools=[])
    assert info.value.status_code == status
    assert info.value.retryable is retryable
    assert info.value.retry_after_s == retry_after


def test_connection_errors_are_retryable() -> None:
    request = httpx.Request("POST", "https://api.groq.com")
    fake = FakeCompletions(error=groq.APIConnectionError(request=request))
    with pytest.raises(LLMError) as info:
        make(fake).complete(system="s", messages=[USER], tools=[])
    assert info.value.retryable


def test_sdk_retries_are_disabled() -> None:
    provider = GroqProvider(LLMSettings(), api_key="gsk_test_0000000000000000")
    assert provider._client.max_retries == 0


# --- tool_choice ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setting", "sent"), [(None, "required"), ("auto", "auto"), ("required", "required")]
)
def test_tool_choice_setting(setting: str | None, sent: str) -> None:
    fake = FakeCompletions(completion(content="ok"))
    make(fake, tool_choice=setting).complete(system="s", messages=[USER], tools=[READ_TOOL])
    assert fake.calls[0]["tool_choice"] == sent


def test_tool_choice_omitted_without_tools() -> None:
    fake = FakeCompletions(completion(content="pong"))
    make(fake, tool_choice="required").complete(system="s", messages=[USER], tools=[])
    assert "tool_choice" not in fake.calls[0]


def test_required_tool_but_text_answer_is_not_resampled() -> None:
    body = {
        "error": {
            "message": "Tool choice is required, but model did not call a tool",
            "code": "tool_use_failed",
            "failed_generation": "All tests pass now.",
        }
    }
    fake = FakeCompletions(error=status_error(400, body))
    with pytest.raises(LLMError) as info:
        make(fake).complete(system="s", messages=[USER], tools=[READ_TOOL])
    err = info.value
    assert (err.kind, err.retryable, err.generated_text) == (
        NO_TOOL_CALL,
        False,
        "All tests pass now.",
    )
