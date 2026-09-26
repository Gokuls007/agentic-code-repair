from __future__ import annotations

from typing import ClassVar

from repair_agent.config import Settings
from repair_agent.llm.base import ToolCall
from repair_agent.sandbox.docker import DockerSandbox
from repair_agent.sandbox.workspace import Workspace
from repair_agent.tools import build_registry
from repair_agent.tools.base import (
    Tool,
    ToolArgs,
    ToolError,
    ToolOutput,
    ToolRegistry,
    truncate_middle,
)


class Echo(Tool):
    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "echo text"

    class Args(ToolArgs):
        text: str
        repeat: int = 1

    def run(self, args: Args) -> ToolOutput:
        if args.text == "fail":
            raise ToolError("asked to fail")
        if args.text == "crash":
            raise KeyError("boom")
        return ToolOutput(content=args.text * args.repeat)


def call(registry: ToolRegistry, name: str, **arguments: object) -> ToolOutput:
    return registry.execute(ToolCall(id="c1", name=name, arguments=arguments))


def test_dispatches_and_returns_content() -> None:
    reg = ToolRegistry([Echo()], max_output_chars=100)
    out = call(reg, "echo", text="hi")
    assert (out.content, out.is_error, out.truncated) == ("hi", False, False)


def test_unknown_tool() -> None:
    out = call(ToolRegistry([Echo()], max_output_chars=100), "nope")
    assert out.is_error
    assert out.content == "Error: unknown tool 'nope'. Available: echo"


def test_invalid_and_extra_arguments() -> None:
    reg = ToolRegistry([Echo()], max_output_chars=100)
    missing = call(reg, "echo")
    assert missing.is_error and missing.content.startswith(
        "Error: invalid arguments for echo: text"
    )
    extra = call(reg, "echo", text="x", bogus=1)
    assert extra.is_error and "bogus" in extra.content


def test_tool_error_and_unexpected_exception_do_not_raise() -> None:
    reg = ToolRegistry([Echo()], max_output_chars=100)
    expected = call(reg, "echo", text="fail")
    assert expected.is_error and expected.content == "Error: asked to fail"
    crash = call(reg, "echo", text="crash")
    assert crash.is_error
    assert crash.content == "Error: internal tool failure (KeyError: 'boom')"
    assert "Traceback" in crash.metadata["traceback"]  # for the trace, not the model
    assert "Traceback" not in crash.content


def test_central_truncation_keeps_head_and_tail() -> None:
    reg = ToolRegistry([Echo()], max_output_chars=90)
    out = call(reg, "echo", text="H" * 30 + "m" * 1000 + "T" * 60)
    assert out.truncated
    assert out.content.startswith(
        "H" * 30 + "\n[... output truncated: 1,000 of 1,090 chars omitted."
    )
    assert out.content.endswith("...]\n" + "T" * 60)


def test_truncate_middle_noop_under_limit() -> None:
    assert truncate_middle("short", 10) == ("short", False)


def test_to_result_maps_fields() -> None:
    result = ToolOutput(content="x", is_error=True).to_result("id9")
    assert (result.tool_call_id, result.content, result.is_error) == ("id9", "x", True)


def test_build_registry_exposes_six_tools_with_strict_schemas(workspace: Workspace) -> None:
    settings = Settings()
    reg = build_registry(workspace.root, DockerSandbox(settings.sandbox, client=object()), settings)
    assert reg.names == [
        "list_files",
        "read_file",
        "search_code",
        "edit_file",
        "run_tests",
        "finish",
    ]
    for spec in reg.specs():
        schema = spec.input_schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert spec.description
    edit = next(s for s in reg.specs() if s.name == "edit_file")
    assert edit.input_schema["required"] == ["path", "old_str", "new_str"]
