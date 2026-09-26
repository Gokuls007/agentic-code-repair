"""Test doubles: a scripted LLM and a Docker-free sandbox."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from repair_agent.config import SandboxSettings
from repair_agent.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    StopReason,
    TextBlock,
    ToolCall,
    ToolSpec,
    Usage,
)
from repair_agent.sandbox.docker import SandboxError
from repair_agent.sandbox.junit import Outcome, TestReport

MODEL_ID = "claude-sonnet-5-scripted"
_ids = itertools.count(1)


def tc(name: str, **arguments: Any) -> ToolCall:
    """A tool call with a unique id."""
    return ToolCall(id=f"toolu_{next(_ids)}", name=name, arguments=arguments)


def reply(
    *calls: ToolCall,
    text: str = "",
    stop: StopReason | None = None,
    input_tokens: int = 1_000,
    output_tokens: int = 100,
    cache_read: int = 0,
) -> LLMResponse:
    """One scripted model turn."""
    content: list[Any] = [TextBlock(text=text)] if text else []
    content += list(calls)
    return LLMResponse(
        message=Message(role="assistant", content=content, provider="scripted"),
        stop_reason=stop or (StopReason.TOOL_USE if calls else StopReason.END_TURN),
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
        ),
        model=MODEL_ID,
        latency_s=0.5,
    )


Script = LLMResponse | Exception | Callable[[list[Message]], LLMResponse]


class ScriptedProvider(LLMProvider):
    """Returns scripted turns in order; records every request it receives."""

    name = "scripted"

    def __init__(self, script: list[Script], *, repeat_last: bool = False):
        self.script = list(script)
        self.repeat_last = repeat_last
        self.requests: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_output_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        self.requests.append(
            {
                "system": system,
                "messages": [m.model_copy(deep=True) for m in messages],
                "tools": [t.name for t in tools],
                "timeout_s": timeout_s,
            }
        )
        if not self.script:
            raise AssertionError("ScriptedProvider ran out of turns")
        item = (
            self.script[0] if (self.repeat_last and len(self.script) == 1) else self.script.pop(0)
        )
        if isinstance(item, Exception):
            raise item
        if callable(item) and not isinstance(item, LLMResponse):
            return item(messages)
        return item


Judge = Callable[[Path, list[str]], dict[str, Outcome]]

CALC_TESTS = [
    "tests/test_ops.py::test_add",
    "tests/test_ops.py::test_subtract",
    "tests/test_ops.py::test_clamp",
    "tests/test_stats.py::test_mean_basic",
    "tests/test_stats.py::test_mean_empty_raises",
    "tests/test_stats.py::test_median",
]


def calc_judge(root: Path, selectors: list[str]) -> dict[str, Outcome]:
    """Decide calc outcomes from file contents, without executing any repo code.

    test_mean_basic passes if mean is fixed, or if the test's expectation was edited
    to match the buggy result (the cheat grading must defeat).
    """
    stats = (root / "src/calc/stats.py").read_text(encoding="utf-8")
    test = (root / "tests/test_stats.py").read_text(encoding="utf-8")
    ok = "sum(xs) / len(xs)" in stats or "3.3333333333333335" in test
    outcomes = {nid: Outcome.PASSED for nid in CALC_TESTS}
    outcomes["tests/test_stats.py::test_mean_basic"] = Outcome.PASSED if ok else Outcome.FAILED
    if selectors:
        outcomes = {k: v for k, v in outcomes.items() if any(k.startswith(s) for s in selectors)}
    return outcomes


class FakeSandbox:
    """Duck-types DockerSandbox.run_pytest with a judge function instead of Docker."""

    def __init__(self, judge: Judge = calc_judge, *, fail: bool = False, **settings: Any):
        self.settings = SandboxSettings(**settings)
        self.judge = judge
        self.fail = fail
        self.calls: list[list[str]] = []

    def run_pytest(
        self, workspace_root: Path, selectors: list[str] = (), *, timeout_s: float | None = None
    ) -> TestReport:
        self.calls.append(list(selectors))
        if self.fail:
            raise SandboxError("Docker is not reachable")
        outcomes = self.judge(Path(workspace_root), list(selectors))
        failed = any(o != Outcome.PASSED for o in outcomes.values())
        return TestReport(
            outcomes=outcomes,
            exit_code=1 if failed else 0,
            duration_s=0.1,
            output="fake pytest output",
        )


class FakeClock:
    """Monotonic clock that advances only when told to (or on each read, if ``step``)."""

    def __init__(self, step: float = 0.0):
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


class RecordingSleep:
    def __init__(self, clock: FakeClock | None = None):
        self.waits: list[float] = []
        self.clock = clock

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)
        if self.clock:
            self.clock.now += seconds
