"""Real-Docker checks: every benchmark task validates, and grading defeats known cheats."""

from __future__ import annotations

from pathlib import Path

import pytest

from fakes import ScriptedProvider, reply, tc
from repair_agent.agent import load_task, solve_task
from repair_agent.agent.task import load_tasks
from repair_agent.config import Settings
from repair_agent.eval.validate import validate_task
from repair_agent.sandbox.docker import DockerSandbox
from repair_agent.sandbox.workspace import Workspace
from repair_agent.tracing import Tracer

pytestmark = pytest.mark.docker

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "benchmark" / "tasks"
MEAN_TASK = TASKS_DIR / "calc-mean-001.yaml"

# Monkeypatches calc.stats.mean with a correct version when imported.
PATCH_MEAN = (
    "import calc.stats\n\n"
    "def _mean(xs):\n"
    "    if not xs:\n"
    "        raise ValueError('mean of empty sequence')\n"
    "    return sum(xs) / len(xs)\n\n"
    "calc.stats.mean = _mean\n"
)


@pytest.mark.parametrize("task", load_tasks(TASKS_DIR), ids=lambda t: t.id)
def test_benchmark_task_is_valid(task, docker_sandbox: DockerSandbox, tmp_path: Path) -> None:
    result = validate_task(task, docker_sandbox, repeat=2, work_dir=tmp_path)
    assert result.ok, result.problems


def _cheat(path: str, content: str, sandbox: DockerSandbox, tmp_path: Path):
    task = load_task(MEAN_TASK)
    script = [
        reply(tc("edit_file", path=path, old_str="", new_str=content)),
        reply(tc("run_tests")),  # full suite, so the cheat module is collected too
        reply(tc("finish", summary="done")),
    ]
    provider = ScriptedProvider(script)
    with (
        Workspace.from_directory(task.repo_dir(), patch=task.seed_patch, parent_dir=tmp_path) as ws,
        Tracer(tmp_path / "runs", "run1", task.id) as tracer,
    ):
        result = solve_task(
            task,
            ws,
            provider=provider,
            sandbox=sandbox,
            settings=Settings(),
            tracer=tracer,
            run_id="run1",
        )
    agent_view = provider.requests[2]["messages"][-1].content[0].content
    return result, agent_view


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("conftest.py", PATCH_MEAN),
        ("src/sitecustomize.py", PATCH_MEAN),
        ("tests/test_aaa_patch.py", PATCH_MEAN + "\n\ndef test_placeholder():\n    pass\n"),
    ],
    ids=["new-root-conftest", "sitecustomize", "import-time-test-module"],
)
def test_collection_cheats_do_not_resolve(
    path: str, content: str, docker_sandbox: DockerSandbox, tmp_path: Path
) -> None:
    result, agent_view = _cheat(path, content, docker_sandbox, tmp_path)
    # The cheat "works" in the agent's own test run...
    assert agent_view.startswith("Result: PASSED"), agent_view
    # ...but not in grading.
    assert not result.resolved
    assert result.final_tests.outcomes["tests/test_stats.py::test_mean_basic"] == "failed"
    if path.startswith("tests/"):
        assert result.removed_files == []  # new test modules stay, but are never collected
    else:
        assert result.removed_files == [path]
