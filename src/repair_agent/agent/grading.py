"""Grading with test integrity.

Before the grading run, every test file that existed in the task's baseline (the graded
test files, plus anything that looks like a test file, including conftest.py) is reset
to its original content. The agent therefore cannot "resolve" a task by editing tests.
New test files the agent added are left in place.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from repair_agent.agent.task import Task, is_test_file
from repair_agent.sandbox.docker import DockerSandbox
from repair_agent.sandbox.junit import Outcome, TestReport
from repair_agent.sandbox.workspace import Workspace


class FinalTests(BaseModel):
    """Summary of the grading run."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    exit_code: int | None = None
    timed_out: bool = False
    duration_s: float = 0.0
    outcomes: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_report(cls, report: TestReport) -> FinalTests:
        return cls(
            passed=report.count(Outcome.PASSED),
            failed=report.count(Outcome.FAILED),
            errors=report.count(Outcome.ERROR),
            skipped=report.count(Outcome.SKIPPED),
            exit_code=report.exit_code,
            timed_out=report.timed_out,
            duration_s=report.duration_s,
            outcomes={k: str(v) for k, v in report.outcomes.items()},
        )

    @property
    def all_passed(self) -> bool:
        """Suite ran to completion with no failures or errors."""
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0


class Grade(BaseModel):
    """Result of grading one task attempt."""

    final_tests: FinalTests
    resolved: bool
    # Original test files the agent changed; they were restored before grading.
    modified_test_files: list[str] = Field(default_factory=list)
    missing_graded_tests: list[str] = Field(default_factory=list)


def protected_test_files(task: Task, workspace: Workspace) -> list[str]:
    """Baseline files the agent must not be able to influence grading through."""
    baseline = set(workspace.baseline_files())
    graded = {f for f in task.graded_test_files if f in baseline}
    heuristic = {f for f in baseline if is_test_file(f)}
    return sorted(graded | heuristic)


def grade(task: Task, workspace: Workspace, sandbox: DockerSandbox) -> Grade:
    """Restore original tests, run the full suite, and decide ``resolved``.

    Resolved means every FAIL_TO_PASS and PASS_TO_PASS test ran and passed.
    Raises SandboxError if the sandbox itself fails.
    """
    protected = protected_test_files(task, workspace)
    modified = workspace.changed_files(protected)
    workspace.restore(modified)

    report = sandbox.run_pytest(workspace.root)
    final = FinalTests.from_report(report)
    missing = [nid for nid in task.graded_tests if nid not in final.outcomes]
    resolved = not report.timed_out and all(
        final.outcomes.get(nid) == Outcome.PASSED for nid in task.graded_tests
    )
    return Grade(
        final_tests=final,
        resolved=resolved,
        modified_test_files=modified,
        missing_graded_tests=missing,
    )
