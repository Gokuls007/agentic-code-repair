from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from repair_agent.config import SandboxSettings, Settings, get_settings
from repair_agent.sandbox.docker import DockerSandbox
from repair_agent.sandbox.workspace import Workspace, git

# The fixture repo has its own tests/ directory; it is data, not part of our suite.
collect_ignore = ["fixtures"]

FIXTURES = Path(__file__).parent / "fixtures"
_ENV_VARS = ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "GITHUB_TOKEN")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep tests independent of the developer's shell env and .env file."""
    for name in list(os.environ):
        if name.startswith("REPAIR_") or name in _ENV_VARS:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in the working directory
    get_settings.cache_clear()


def make_git_repo(src: Path, dest: Path) -> Path:
    """Copy ``src`` to ``dest`` and commit it as a fresh repository."""
    shutil.copytree(src, dest)
    git(dest, "init", "--quiet", "--initial-branch=main")
    git(dest, "add", "-A")
    git(
        dest,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@localhost",
        "commit",
        "--quiet",
        "--no-verify",
        "-m",
        "initial",
    )
    return dest


@pytest.fixture
def calc_source(tmp_path: Path) -> Path:
    """The calc fixture as a committed git repo (the 'upstream' the workspace clones)."""
    return make_git_repo(FIXTURES / "calc_repo", tmp_path / "upstream")


@pytest.fixture
def workspace(calc_source: Path, tmp_path: Path) -> Iterator[Workspace]:
    """A fresh Workspace cloned from the calc fixture."""
    parent = tmp_path / "workspaces"
    parent.mkdir()
    ws = Workspace.create(calc_source, parent_dir=parent)
    yield ws
    ws.cleanup()


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def docker_sandbox() -> DockerSandbox:
    """A real sandbox with the image built. Skips the test if Docker is unreachable."""
    sandbox = DockerSandbox(SandboxSettings())
    try:
        sandbox.client  # noqa: B018 - connects and pings
    except Exception as exc:
        pytest.skip(f"Docker not available: {exc}")
    sandbox.ensure_image()
    return sandbox
