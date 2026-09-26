"""run_tests: pytest inside the Docker sandbox."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field

from repair_agent.sandbox.docker import DockerSandbox, SandboxError
from repair_agent.sandbox.junit import Outcome, TestReport
from repair_agent.tools.base import Tool, ToolArgs, ToolError, ToolOutput

# Node ids and paths only: tests/test_x.py, tests/test_x.py::TestA::test_b[param-1]
_SELECTOR_TOKEN = re.compile(r"^[\w./:\[\]\-+=,@]+$")
MAX_LISTED_FAILURES = 30
EXIT_CODE_OOM = 137


def parse_selector(selector: str | None) -> list[str]:
    """Split a selector into pytest args, rejecting flags and odd characters."""
    if not selector or not selector.strip():
        return []
    tokens = selector.split()
    for token in tokens:
        if token.startswith("-"):
            raise ToolError(
                f"test_selector accepts test paths or node ids only, not pytest options ({token!r})"
            )
        if not _SELECTOR_TOKEN.match(token):
            raise ToolError(f"invalid characters in test selector token {token!r}")
    return tokens


def summarize(report: TestReport, timeout_s: float, memory_mb: int) -> str:
    """Human/LLM-readable summary line plus failing test ids, then the pytest output."""
    passed, failed = report.count(Outcome.PASSED), report.count(Outcome.FAILED)
    errors, skipped = report.count(Outcome.ERROR), report.count(Outcome.SKIPPED)
    counts = f"{failed} failed, {passed} passed, {errors} errors"
    if skipped:
        counts += f", {skipped} skipped"
    code = report.exit_code

    if report.timed_out:
        head = f"Result: TIMEOUT: killed after {timeout_s:.0f}s (possible infinite loop or hang)"
    elif report.oom or code == EXIT_CODE_OOM:
        head = f"Result: OUT OF MEMORY: killed at {memory_mb} MB limit"
    elif code == 0:
        head = f"Result: PASSED ({passed} passed"
        head += f", {skipped} skipped" if skipped else ""
        head += f") in {report.duration_s:.1f}s"
    elif code == 1:
        head = f"Result: FAILED ({counts}) in {report.duration_s:.1f}s [exit code 1]"
    elif code == 2:
        head = f"Result: INTERRUPTED, errors during collection ({counts}) [exit code 2]"
    elif code == 5:
        head = "Result: NO TESTS COLLECTED (exit code 5)"
    else:
        head = f"Result: ERROR (pytest exit code {code}; see output)"

    parts = [head]
    bad = report.ids_with(Outcome.FAILED) + report.ids_with(Outcome.ERROR)
    if bad:
        listed = "\n".join(f"- {nid}" for nid in bad[:MAX_LISTED_FAILURES])
        more = len(bad) - MAX_LISTED_FAILURES
        parts.append("Failed:\n" + listed + (f"\n- ... and {more} more" if more > 0 else ""))
    if report.output.strip():
        parts.append("--- pytest output ---\n" + report.output.strip())
    return "\n".join(parts)


class RunTests(Tool):
    """Run the repo's tests in the sandbox."""

    name = "run_tests"
    description = (
        "Run the test suite with pytest inside an isolated sandbox (no network). "
        "Optionally pass test_selector: space-separated test files or node ids, e.g. "
        "'tests/test_ops.py::test_add'. Returns a summary line, the failing test ids, "
        "and pytest's output. Test runs are limited, so prefer targeted selectors."
    )

    class Args(ToolArgs):
        test_selector: str | None = Field(
            default=None, description="Test paths or node ids; omit to run the whole suite."
        )

    def __init__(self, root: Path, sandbox: DockerSandbox):
        self.root = root
        self.sandbox = sandbox

    def run(self, args: Args) -> ToolOutput:
        selectors = parse_selector(args.test_selector)
        try:
            report = self.sandbox.run_pytest(self.root, selectors)
        except SandboxError as exc:
            raise ToolError(f"sandbox failure, tests did not run: {exc}") from exc
        s = self.sandbox.settings
        return ToolOutput(
            content=summarize(report, s.test_timeout_s, s.memory_mb),
            metadata={
                "exit_code": report.exit_code,
                "passed": report.count(Outcome.PASSED),
                "failed": report.count(Outcome.FAILED),
                "errors": report.count(Outcome.ERROR),
                "skipped": report.count(Outcome.SKIPPED),
                "duration_s": report.duration_s,
                "timed_out": report.timed_out,
                "oom": report.oom,
                "outcomes": {k: str(v) for k, v in report.outcomes.items()},
                "selectors": selectors,
            },
        )
