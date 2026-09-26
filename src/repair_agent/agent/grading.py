"""Grading with test integrity.

Before the grading run:

1. Every baseline test file (graded test files, anything that looks like a test file)
   and every baseline test-config file (conftest.py, pytest.ini, tox.ini, setup.cfg,
   pyproject.toml) is reset to its original content.
2. Newly added files that can change test collection or interpreter start-up
   (conftest.py, pytest/tox/setup config, sitecustomize/usercustomize, *.pth) are deleted.
3. Only the graded node ids (FAIL_TO_PASS + PASS_TO_PASS) are run, so new test modules
   the agent added are never imported and cannot monkeypatch the code under test.

The agent therefore cannot "resolve" a task by changing how tests run.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field

from repair_agent.agent.task import Task, is_test_file
from repair_agent.sandbox.docker import DockerSandbox
from repair_agent.sandbox.junit import Outcome, TestReport
from repair_agent.sandbox.workspace import Workspace

# Files that configure pytest; restored if they existed, deleted if the agent added them.
TEST_CONFIG_FILES = frozenset(
    {"conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml"}
)
# Files Python runs at start-up when found on sys.path (PYTHONPATH includes the workspace).
STARTUP_FILES = frozenset({"sitecustomize.py", "usercustomize.py"})


class FinalTests(BaseModel):
    """Summary of the grading run (graded tests only)."""

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
        """The run completed with no failures or errors."""
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0


class Grade(BaseModel):
    """Result of grading one task attempt."""

    final_tests: FinalTests
    resolved: bool
    # Original test/config files the agent changed; restored before grading.
    modified_test_files: list[str] = Field(default_factory=list)
    # Files the agent added that could affect collection; deleted before grading.
    removed_files: list[str] = Field(default_factory=list)
    # Non-test files the agent changed (the attempted fix).
    source_files_changed: list[str] = Field(default_factory=list)
    missing_graded_tests: list[str] = Field(default_factory=list)
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0


def is_test_config(path: str) -> bool:
    return PurePosixPath(path).name in TEST_CONFIG_FILES


def affects_collection(path: str) -> bool:
    """A new file that could change what pytest collects or how Python starts."""
    p = PurePosixPath(path)
    return p.name in TEST_CONFIG_FILES or p.name in STARTUP_FILES or p.suffix == ".pth"


def protected_test_files(task: Task, workspace: Workspace) -> list[str]:
    """Baseline files the agent must not be able to influence grading through."""
    baseline = set(workspace.baseline_files())
    graded = {f for f in task.graded_test_files if f in baseline}
    heuristic = {f for f in baseline if is_test_file(f) or is_test_config(f)}
    return sorted(graded | heuristic)


def sanitize_for_grading(task: Task, workspace: Workspace) -> tuple[list[str], list[str]]:
    """Restore protected files and delete collection-affecting additions.

    Returns (modified_and_restored, removed).
    """
    modified = workspace.changed_files(protected_test_files(task, workspace))
    workspace.restore(modified)
    removed = [f for f in workspace.added_files() if affects_collection(f)]
    for rel in removed:
        (workspace.root / Path(rel)).unlink(missing_ok=True)
    return modified, removed


def _passed(outcomes: dict[str, str], ids: list[str]) -> int:
    return sum(1 for nid in ids if outcomes.get(nid) == Outcome.PASSED)


def grade(task: Task, workspace: Workspace, sandbox: DockerSandbox) -> Grade:
    """Sanitize the workspace, run the graded tests, and decide ``resolved``.

    Resolved means every FAIL_TO_PASS and PASS_TO_PASS test ran and passed.
    Raises SandboxError if the sandbox itself fails.
    """
    source_changed = [
        f
        for f in workspace.all_changed_files()
        if not is_test_file(f) and not affects_collection(f)
    ]
    modified, removed = sanitize_for_grading(task, workspace)

    report = sandbox.run_pytest(workspace.root, task.graded_tests)
    final = FinalTests.from_report(report)
    outcomes = final.outcomes
    resolved = not report.timed_out and all(
        outcomes.get(nid) == Outcome.PASSED for nid in task.graded_tests
    )
    return Grade(
        final_tests=final,
        resolved=resolved,
        modified_test_files=modified,
        removed_files=removed,
        source_files_changed=source_changed,
        missing_graded_tests=[nid for nid in task.graded_tests if nid not in outcomes],
        f2p_passed=_passed(outcomes, task.fail_to_pass),
        f2p_total=len(task.fail_to_pass),
        p2p_passed=_passed(outcomes, task.pass_to_pass),
        p2p_total=len(task.pass_to_pass),
    )
