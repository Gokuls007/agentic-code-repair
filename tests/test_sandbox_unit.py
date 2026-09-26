"""DockerSandbox behaviour with a fake Docker client: no daemon needed."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import docker.errors
import pytest
import requests

from repair_agent.config import SandboxSettings
from repair_agent.sandbox.docker import (
    JUNIT_PATH,
    LABEL,
    DockerSandbox,
    SandboxError,
    build_workspace_tar,
)
from repair_agent.sandbox.workspace import Workspace
from repair_agent.tools.base import ToolError
from repair_agent.tools.tests_tool import RunTests, parse_selector

JUNIT_ONE_FAIL = b"""<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="tests.test_stats" file="tests/test_stats.py" name="test_mean_basic">
<failure message="assert 3.33 == 2.5"/></testcase>
<testcase classname="tests.test_stats" file="tests/test_stats.py" name="test_median"/>
</testsuite></testsuites>"""


def tar_of(name: str, data: bytes) -> list[bytes]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return [buf.getvalue()]


class FakeContainer:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        hang: bool = False,
        oom: bool = False,
        logs: bytes = b"",
        junit: bytes | None = None,
        start_error: bool = False,
    ):
        self.exit_code, self.hang, self.oom = exit_code, hang, oom
        self._logs, self.junit, self.start_error = logs, junit, start_error
        self.calls: list[str] = []
        self.archive: bytes | None = None
        self.attrs: dict[str, Any] = {}

    def put_archive(self, path: str, data: bytes) -> bool:
        self.calls.append("put_archive")
        self.archive = data
        return True

    def start(self) -> None:
        self.calls.append("start")
        if self.start_error:
            raise docker.errors.APIError("start failed")

    def wait(self, timeout: float) -> dict[str, int]:
        self.calls.append(f"wait({timeout})")
        if self.hang:
            raise requests.exceptions.ReadTimeout("timed out")
        return {"StatusCode": self.exit_code}

    def kill(self) -> None:
        self.calls.append("kill")

    def reload(self) -> None:
        self.attrs = {"State": {"OOMKilled": self.oom}}

    def logs(self, stdout: bool, stderr: bool) -> bytes:
        return self._logs

    def get_archive(self, path: str):
        if path == JUNIT_PATH and self.junit is not None:
            return tar_of("junit.xml", self.junit), {}
        raise docker.errors.NotFound("missing")

    def remove(self, force: bool) -> None:
        self.calls.append("remove")


class FakeContainers:
    def __init__(self, container: FakeContainer):
        self.container = container
        self.create_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> FakeContainer:
        self.create_kwargs = kwargs
        return self.container


class FakeClient:
    def __init__(self, container: FakeContainer):
        self.containers = FakeContainers(container)


def make(container: FakeContainer, **settings: Any) -> tuple[DockerSandbox, FakeClient]:
    client = FakeClient(container)
    return DockerSandbox(SandboxSettings(**settings), client=client), client


def test_container_is_locked_down(workspace: Workspace) -> None:
    sandbox, client = make(FakeContainer(), cpus=0.5, memory_mb=512, pids_limit=64)
    sandbox.run(workspace.root, ["python", "-V"])
    kw = client.containers.create_kwargs
    assert kw["network_mode"] == "none"
    assert kw["user"] == "1000:1000"
    assert kw["cap_drop"] == ["ALL"]
    assert kw["security_opt"] == ["no-new-privileges"]
    assert kw["mem_limit"] == kw["memswap_limit"] == "512m"
    assert kw["nano_cpus"] == 500_000_000
    assert kw["pids_limit"] == 64
    assert kw["labels"] == {LABEL: "1"}
    assert kw["command"] == ["python", "-V"]
    assert not {"volumes", "mounts", "privileged", "network"} & kw.keys()  # no bind mounts


def test_lifecycle_order_and_result(workspace: Workspace) -> None:
    container = FakeContainer(exit_code=3, logs=b"line1\r\nline2\n")
    sandbox, _ = make(container)
    result = sandbox.run(workspace.root, ["x"], timeout_s=7)
    assert container.calls == ["put_archive", "start", "wait(7)", "remove"]
    assert (result.exit_code, result.timed_out, result.oom) == (3, False, False)
    assert result.output == "line1\nline2\n"


def test_timeout_kills_and_still_removes(workspace: Workspace) -> None:
    container = FakeContainer(hang=True)
    sandbox, _ = make(container)
    result = sandbox.run(workspace.root, ["x"], timeout_s=2)
    assert result.timed_out and result.exit_code is None
    assert container.calls[-2:] == ["kill", "remove"]


def test_remove_runs_even_if_start_fails(workspace: Workspace) -> None:
    container = FakeContainer(start_error=True)
    sandbox, _ = make(container)
    with pytest.raises(SandboxError):
        sandbox.run(workspace.root, ["x"])
    assert container.calls[-1] == "remove"


def test_oom_is_reported(workspace: Workspace) -> None:
    sandbox, _ = make(FakeContainer(exit_code=137, oom=True))
    assert sandbox.run(workspace.root, ["x"]).oom


def test_missing_image_gives_build_hint(workspace: Workspace) -> None:
    class NoImage(FakeContainers):
        def create(self, **kwargs: Any) -> FakeContainer:
            raise docker.errors.ImageNotFound("nope")

    client = FakeClient(FakeContainer())
    client.containers = NoImage(FakeContainer())
    sandbox = DockerSandbox(SandboxSettings(), client=client)
    with pytest.raises(SandboxError, match="sandbox build"):
        sandbox.run(workspace.root, ["x"])


def test_logs_are_capped_and_redacted(workspace: Workspace) -> None:
    logs = b"x" * 50 + b" key=sk-ant-api03-SECRETSECRETSECRET\n"
    sandbox, _ = make(FakeContainer(logs=logs), max_log_bytes=60)
    out = sandbox.run(workspace.root, ["x"]).output
    assert out.startswith("[... first ")
    assert "SECRETSECRET" not in out and "[REDACTED]" in out


def test_workspace_tar_is_posix_owned_by_sandbox_user_and_skips_git(workspace: Workspace) -> None:
    (workspace.root / "src/calc/__pycache__").mkdir()
    (workspace.root / "src/calc/__pycache__/ops.cpython-311.pyc").write_bytes(b"x")
    (workspace.root / "win.py").write_bytes(b"a = 1\r\n")
    with tarfile.open(fileobj=io.BytesIO(build_workspace_tar(workspace.root))) as tar:
        members = {m.name: m for m in tar.getmembers()}
        assert "workspace/src/calc/stats.py" in members
        assert not any("\\" in name for name in members)
        assert not any(".git" in name.split("/") for name in members)
        assert not any("__pycache__" in name for name in members)
        assert all(m.uid == 1000 and m.gid == 1000 for m in members.values())
        assert members["workspace/src/calc/stats.py"].mode == 0o644
        assert members["workspace/src"].mode == 0o755
        win = tar.extractfile(members["workspace/win.py"])
        assert win is not None and win.read() == b"a = 1\r\n"  # bytes untouched


def test_run_pytest_parses_junit(workspace: Workspace) -> None:
    sandbox, client = make(FakeContainer(exit_code=1, junit=JUNIT_ONE_FAIL))
    report = sandbox.run_pytest(workspace.root, ["tests/test_stats.py"])
    assert report.outcomes == {
        "tests/test_stats.py::test_mean_basic": "failed",
        "tests/test_stats.py::test_median": "passed",
    }
    assert client.containers.create_kwargs["command"][-1] == "tests/test_stats.py"


def test_run_tests_tool_summary_and_metadata(workspace: Workspace) -> None:
    sandbox, _ = make(FakeContainer(exit_code=1, junit=JUNIT_ONE_FAIL, logs=b"1 failed, 1 passed"))
    tool = RunTests(workspace.root, sandbox)
    out = tool.run(tool.Args())
    lines = out.content.splitlines()
    assert lines[0].startswith("Result: FAILED (1 failed, 1 passed, 0 errors) in ")
    assert lines[0].endswith("[exit code 1]")
    assert lines[1:3] == ["Failed:", "- tests/test_stats.py::test_mean_basic"]
    assert "--- pytest output ---" in out.content
    assert not out.is_error
    assert out.metadata["failed"] == 1 and out.metadata["passed"] == 1
    assert out.metadata["outcomes"]["tests/test_stats.py::test_median"] == "passed"


def test_run_tests_tool_timeout_summary(workspace: Workspace) -> None:
    sandbox, _ = make(FakeContainer(hang=True), test_timeout_s=3)
    tool = RunTests(workspace.root, sandbox)
    out = tool.run(tool.Args())
    assert out.content.startswith("Result: TIMEOUT: killed after 3s")
    assert out.metadata["timed_out"] is True


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (None, []),
        ("  ", []),
        ("tests/test_ops.py", ["tests/test_ops.py"]),
        (
            "tests/a.py::TestX::test_y[1-2] tests/b.py",
            ["tests/a.py::TestX::test_y[1-2]", "tests/b.py"],
        ),
    ],
)
def test_parse_selector_accepts_paths_and_node_ids(
    selector: str | None, expected: list[str]
) -> None:
    assert parse_selector(selector) == expected


@pytest.mark.parametrize(
    "bad", ["-p evil_plugin", "tests --rootdir=/", "tests/a.py;rm -rf /", "$(x)"]
)
def test_parse_selector_rejects_flags_and_shell_chars(bad: str) -> None:
    with pytest.raises(ToolError):
        parse_selector(bad)


def test_windows_paths_never_reach_docker(workspace: Workspace) -> None:
    sandbox, client = make(FakeContainer())
    sandbox.run(workspace.root, ["x"])
    flat = repr(client.containers.create_kwargs)
    assert str(Path(workspace.root)) not in flat
    assert ":\\" not in flat
