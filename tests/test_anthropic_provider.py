"""AnthropicProvider translation tests, using a fake SDK client (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message as WireMessage

from repair_agent.config import LLMSettings
from repair_agent.llm.anthropic import AnthropicProvider
from repair_agent.llm.base import (
    LLMError,
    Message,
    StopReason,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        self.calls.append(params)
        if self.error:
            raise self.error
        return self.response


def make_provider(messages: FakeMessages, **settings: Any) -> AnthropicProvider:
    client = SimpleNamespace(messages=messages)
    return AnthropicProvider(LLMSettings(**settings), api_key="unused", client=client)


def wire_response(content: list[dict[str, Any]], stop_reason: str = "tool_use") -> WireMessage:
    return WireMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": None,
            },
        }
    )


READ_TOOL = ToolSpec(
    name="read_file",
    description="Read a file",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
)


def test_parses_text_tool_calls_and_usage() -> None:
    fake = FakeMessages(
        wire_response(
            [
                {"type": "thinking", "thinking": "", "signature": "sig123"},
                {"type": "text", "text": "Let me read it."},
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "a.py"}},
            ]
        )
    )
    resp = make_provider(fake).complete(
        system="sys",
        messages=[Message(role="user", content=[TextBlock(text="fix bug")])],
        tools=[READ_TOOL],
    )
    assert resp.stop_reason == StopReason.TOOL_USE
    assert resp.message.text == "Let me read it."
    assert resp.message.tool_calls == [
        ToolCall(id="tu_1", name="read_file", arguments={"path": "a.py"})
    ]
    assert resp.usage.input_tokens == 120
    assert resp.usage.output_tokens == 30
    assert resp.usage.cache_read_input_tokens == 1000
    assert resp.usage.cache_creation_input_tokens == 0
    assert resp.model == "claude-opus-5"
    # thinking block is kept for verbatim replay
    assert resp.message.provider_raw is not None
    assert resp.message.provider_raw[0]["type"] == "thinking"
    assert resp.message.provider_raw[0]["signature"] == "sig123"


def test_request_shape() -> None:
    fake = FakeMessages(wire_response([{"type": "text", "text": "ok"}], "end_turn"))
    make_provider(fake, model="claude-sonnet-5", max_output_tokens=4096, effort="high").complete(
        system="be careful",
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
        tools=[READ_TOOL],
    )
    params = fake.calls[0]
    assert params["model"] == "claude-sonnet-5"
    assert params["max_tokens"] == 4096
    assert params["system"] == [
        {"type": "text", "text": "be careful", "cache_control": {"type": "ephemeral"}}
    ]
    assert params["cache_control"] == {"type": "ephemeral"}  # automatic history caching
    assert params["output_config"] == {"effort": "high"}
    assert params["tools"] == [READ_TOOL.model_dump()]
    assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert "temperature" not in params
    assert "timeout" not in params


def test_omits_optional_params_when_unset() -> None:
    fake = FakeMessages(wire_response([{"type": "text", "text": "ok"}], "end_turn"))
    make_provider(fake, prompt_caching=False).complete(
        system="s", messages=[Message(role="user", content=[TextBlock(text="hi")])], tools=[]
    )
    params = fake.calls[0]
    assert "tools" not in params
    assert "output_config" not in params
    assert "cache_control" not in params
    assert params["system"] == [{"type": "text", "text": "s"}]


def test_temperature_and_timeout_are_sent_when_set() -> None:
    fake = FakeMessages(wire_response([{"type": "text", "text": "ok"}], "end_turn"))
    make_provider(fake, temperature=0.0).complete(
        system="s",
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
        tools=[],
        timeout_s=12.5,
    )
    assert fake.calls[0]["temperature"] == 0.0
    assert fake.calls[0]["timeout"] == 12.5


def test_retry_after_header_is_parsed() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(429, request=request, headers={"retry-after": "7"}, json={})
    fake = FakeMessages(error=anthropic.APIStatusError("slow", response=response, body=None))
    with pytest.raises(LLMError) as info:
        make_provider(fake).complete(
            system="s", messages=[Message(role="user", content=[TextBlock(text="hi")])], tools=[]
        )
    assert info.value.retryable and info.value.retry_after_s == 7.0


def test_sdk_retries_are_disabled() -> None:
    provider = AnthropicProvider(LLMSettings(), api_key="sk-ant-test-0000000000")
    assert provider._client.max_retries == 0


def test_replays_own_raw_blocks_and_serialises_tool_results() -> None:
    first = FakeMessages(
        wire_response(
            [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "a.py"}},
            ]
        )
    )
    provider = make_provider(first)
    user = Message(role="user", content=[TextBlock(text="fix")])
    assistant = provider.complete(system="s", messages=[user], tools=[READ_TOOL]).message
    result = Message(
        role="user",
        content=[ToolResult(tool_call_id="tu_1", content="no such file", is_error=True)],
    )

    second = FakeMessages(wire_response([{"type": "text", "text": "done"}], "end_turn"))
    provider._client = SimpleNamespace(messages=second)
    provider.complete(system="s", messages=[user, assistant, result], tools=[READ_TOOL])

    sent = second.calls[0]["messages"]
    assert sent[1]["content"][0]["type"] == "thinking"  # replayed verbatim
    assert sent[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": "no such file",
                "is_error": True,
            }
        ],
    }


def test_foreign_assistant_messages_are_rebuilt_from_neutral_blocks() -> None:
    fake = FakeMessages(wire_response([{"type": "text", "text": "ok"}], "end_turn"))
    foreign = Message(
        role="assistant",
        content=[TextBlock(text="t"), ToolCall(id="c1", name="read_file", arguments={"path": "x"})],
        provider="groq",
        provider_raw=[{"weird": "shape"}],
    )
    make_provider(fake).complete(
        system="s",
        messages=[Message(role="user", content=[TextBlock(text="hi")]), foreign],
        tools=[READ_TOOL],
    )
    assert fake.calls[0]["messages"][1]["content"] == [
        {"type": "text", "text": "t"},
        {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "x"}},
    ]


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("end_turn", StopReason.END_TURN),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("refusal", StopReason.REFUSAL),
        ("pause_turn", StopReason.OTHER),
    ],
)
def test_stop_reason_mapping(wire: str, expected: StopReason) -> None:
    fake = FakeMessages(wire_response([{"type": "text", "text": "x"}], wire))
    resp = make_provider(fake).complete(
        system="s", messages=[Message(role="user", content=[TextBlock(text="hi")])], tools=[]
    )
    assert resp.stop_reason == expected


def _status_error(status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request, json={"error": {"message": "boom"}})
    return anthropic.APIStatusError("boom", response=response, body=None)


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (400, False),
        (401, False),
        (404, False),
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (529, True),
    ],
)
def test_api_errors_become_llm_errors(status: int, retryable: bool) -> None:
    fake = FakeMessages(error=_status_error(status))
    with pytest.raises(LLMError) as info:
        make_provider(fake).complete(
            system="s", messages=[Message(role="user", content=[TextBlock(text="hi")])], tools=[]
        )
    assert info.value.status_code == status
    assert info.value.retryable is retryable


@pytest.mark.parametrize(
    ("setting", "sent"), [(None, None), ("auto", None), ("required", {"type": "any"})]
)
def test_tool_choice_setting(setting: str | None, sent: dict | None) -> None:
    fake = FakeMessages(wire_response([{"type": "text", "text": "ok"}], "end_turn"))
    make_provider(fake, tool_choice=setting).complete(
        system="s",
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
        tools=[READ_TOOL],
    )
    assert fake.calls[0].get("tool_choice") == sent
