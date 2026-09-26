"""Real-Docker integration tests. Skipped automatically when the daemon is unreachable.

Includes a scripted "agent without an LLM": the same tool calls the model would make,
issued directly through the registry, taking the fixture repo from failing to passing.
"""

from __future__ import annotations

import pytest

from repair_agent.config import SandboxSettings, Settings
from repair_agent.llm.base import ToolCall
from repair_agent.sandbox.docker import LABEL, DockerSandbox
from repair_agent.sandbox.workspace import Workspace
from repair_agent.tools import build_registry
from repair_agent.tools.base import ToolOutput, ToolRegistry

pytestmark = pytest.mark.docker

BUGGY_TEST = "tests/test_stats.py::test_mean_basic"


def call(registry: ToolRegistry, name: str, **arguments: object) -> ToolOutput:
    out = registry.execute(ToolCall(id=f"call_{name}", name=name, arguments=arguments))
    assert not out.is_error, out.content
    return out


def sandbox_with(base: DockerSandbox, **overrides: object) -> DockerSandbox:
    settings = SandboxSettings.model_validate({**base.settings.model_dump(), **overrides})
    return DockerSandbox(settings, client=base.client)


def test_buggy_fixture_fails_exactly_one_test(
    docker_sandbox: DockerSandbox, workspace: Workspace
) -> None:
    report = docker_sandbox.run_pytest(workspace.root)
    assert report.exit_code == 1
    assert report.ids_with("failed") == [BUGGY_TEST]
    assert report.count("passed") == 5


def test_scripted_agent_fixes_the_bug_end_to_end(
    docker_sandbox: DockerSandbox, workspace: Workspace
) -> None:
    registry = build_registry(workspace.root, docker_sandbox, Settings())

    before = call(registry, "run_tests")
    assert before.content.startswith("Result: FAILED (1 failed, 5 passed, 0 errors)")
    assert f"- {BUGGY_TEST}" in before.content

    found = call(registry, "search_code", pattern=r"def mean")
    assert found.content == "src/calc/stats.py:1:def mean(xs):"

    source = call(registry, "read_file", path="src/calc/stats.py", end_line=4)
    assert "(len(xs) - 1)" in source.content

    call(
        registry,
        "edit_file",
        path="src/calc/stats.py",
        old_str="return sum(xs) / (len(xs) - 1)",
        new_str="return sum(xs) / len(xs)",
    )

    targeted = call(registry, "run_tests", test_selector=BUGGY_TEST)
    assert targeted.content.startswith("Result: PASSED (1 passed)")

    after = call(registry, "run_tests")
    assert after.content.startswith("Result: PASSED (6 passed)")
    assert after.metadata["outcomes"][BUGGY_TEST] == "passed"

    done = call(registry, "finish", summary="mean divided by len-1; now divides by len")
    assert done.metadata["finished"] is True

    diff = workspace.diff()
    assert "-    return sum(xs) / (len(xs) - 1)" in diff
    assert "+    return sum(xs) / len(xs)" in diff


def _write_test(ws: Workspace, name: str, body: str) -> str:
    path = ws.root / "tests" / f"{name}.py"
    path.write_bytes(body.encode("utf-8"))
    return f"tests/{name}.py"


def test_network_is_disabled(docker_sandbox: DockerSandbox, workspace: Workspace) -> None:
    sel = _write_test(
        workspace,
        "test_net",
        "import socket\n\n"
        "def test_no_network():\n"
        "    s = socket.socket()\n"
        "    s.settimeout(3)\n"
        "    try:\n"
        "        s.connect(('1.1.1.1', 53))\n"
        "    except OSError:\n"
        "        return\n"
        "    raise AssertionError('network reachable')\n",
    )
    report = docker_sandbox.run_pytest(workspace.root, [sel])
    assert report.exit_code == 0, report.output


def test_runs_as_non_root_and_container_writes_do_not_leak(
    docker_sandbox: DockerSandbox, workspace: Workspace
) -> None:
    sel = _write_test(
        workspace,
        "test_user",
        "import os, pathlib\n\n"
        "def test_identity():\n"
        "    assert os.getuid() == 1000 and os.getgid() == 1000\n"
        "    pathlib.Path('/workspace/written_in_container.txt').write_text('x')\n",
    )
    report = docker_sandbox.run_pytest(workspace.root, [sel])
    assert report.exit_code == 0, report.output
    assert not (workspace.root / "written_in_container.txt").exists()


def test_timeout_kills_hanging_tests(docker_sandbox: DockerSandbox, workspace: Workspace) -> None:
    sel = _write_test(workspace, "test_hang", "def test_hang():\n    while True:\n        pass\n")
    report = sandbox_with(docker_sandbox, test_timeout_s=3).run_pytest(workspace.root, [sel])
    assert report.timed_out
    assert report.exit_code is None
    assert report.duration_s < 30


def test_memory_limit_is_enforced(docker_sandbox: DockerSandbox, workspace: Workspace) -> None:
    sel = _write_test(
        workspace,
        "test_mem",
        "def test_mem():\n    blob = bytearray(2 * 1024**3)\n    assert blob\n",
    )
    report = sandbox_with(docker_sandbox, memory_mb=256).run_pytest(workspace.root, [sel])
    # bytearray zero-fills, so pages are touched and the kernel OOM-kills the container.
    assert report.oom
    assert report.exit_code == 137


def test_no_containers_left_behind(docker_sandbox: DockerSandbox, workspace: Workspace) -> None:
    docker_sandbox.run(workspace.root, ["python", "-c", "print('hi')"])
    leftovers = docker_sandbox.client.containers.list(all=True, filters={"label": LABEL})
    assert leftovers == []
