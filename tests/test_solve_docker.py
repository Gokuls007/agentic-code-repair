"""solve_task end to end through the real Docker sandbox, with a scripted LLM (no API cost)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fakes import ScriptedProvider, reply, tc
from repair_agent.agent import Outcome, StopReason, load_task, solve_task
from repair_agent.config import Settings
from repair_agent.sandbox.docker import DockerSandbox
from repair_agent.sandbox.workspace import Workspace
from repair_agent.tracing import Tracer

pytestmark = pytest.mark.docker

TASK_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "tasks" / "calc-mean-001.yaml"


def _solve(script, sandbox: DockerSandbox, tmp_path: Path):
    task = load_task(TASK_PATH)
    with (
        Workspace.from_directory(task.repo_dir(), patch=task.seed_patch, parent_dir=tmp_path) as ws,
        Tracer(tmp_path / "runs", "run1", task.id) as tracer,
    ):
        return solve_task(
            task,
            ws,
            provider=ScriptedProvider(script),
            sandbox=sandbox,
            settings=Settings(),
            tracer=tracer,
            run_id="run1",
        )


def test_scripted_fix_resolves_in_real_sandbox(
    docker_sandbox: DockerSandbox, tmp_path: Path
) -> None:
    script = [
        reply(tc("run_tests", test_selector="tests/test_stats.py")),
        reply(
            tc(
                "edit_file",
                path="src/calc/stats.py",
                old_str="return sum(xs) / (len(xs) - 1)",
                new_str="return sum(xs) / len(xs)",
            )
        ),
        reply(tc("run_tests")),
        reply(tc("finish", summary="fixed the denominator")),
    ]
    result = _solve(script, docker_sandbox, tmp_path)
    assert result.stop_reason == StopReason.FINISHED
    assert result.outcome == Outcome.FINISHED_TESTS_PASS
    assert result.resolved and result.test_runs == 2
    assert result.final_tests.passed == 6


def test_test_edit_cheat_is_not_resolved_in_real_sandbox(
    docker_sandbox: DockerSandbox, tmp_path: Path
) -> None:
    script = [
        reply(
            tc(
                "edit_file",
                path="tests/test_stats.py",
                old_str="assert mean([1, 2, 3, 4]) == 2.5",
                new_str="assert mean([1, 2, 3, 4]) == 3.3333333333333335",
            )
        ),
        reply(tc("finish", summary="changed the expected value")),
    ]
    result = _solve(script, docker_sandbox, tmp_path)
    assert result.modified_test_files == ["tests/test_stats.py"]
    assert not result.resolved
    assert result.final_tests.outcomes["tests/test_stats.py::test_mean_basic"] == "failed"
