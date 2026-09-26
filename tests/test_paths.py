from __future__ import annotations

import os
from pathlib import Path

import pytest

from repair_agent.tools.base import ToolError
from repair_agent.tools.paths import display_path, resolve_in_workspace


def test_resolves_relative_posix_and_backslash_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    expected = (tmp_path / "src" / "a.py").resolve()
    assert resolve_in_workspace(tmp_path, "src/a.py") == expected
    assert resolve_in_workspace(tmp_path, "src\\a.py") == expected
    assert resolve_in_workspace(tmp_path, "./src/../src/a.py") == expected
    assert resolve_in_workspace(tmp_path, "") == tmp_path.resolve()


@pytest.mark.parametrize(
    "bad",
    ["../outside.py", "src/../../x", "/etc/passwd", "C:\\Windows\\win.ini", "C:x", "\\\\srv\\s"],
)
def test_rejects_escapes_and_absolute_paths(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ToolError):
        resolve_in_workspace(tmp_path, bad)


@pytest.mark.parametrize("bad", [".git", ".git/config", "./.git/hooks/pre-commit"])
def test_rejects_git_internals(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ToolError, match=r"\.git"):
        resolve_in_workspace(tmp_path, bad)


def test_rejects_symlink_escaping_workspace(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("x", encoding="utf-8")
    try:
        os.symlink(outside, root / "link.txt")
    except OSError:
        pytest.skip("creating symlinks needs Developer Mode or admin on Windows")
    with pytest.raises(ToolError, match="outside"):
        resolve_in_workspace(root, "link.txt")


def test_display_path_is_posix(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b.py"
    assert display_path(tmp_path, nested) == "a/b.py"
    assert display_path(tmp_path, tmp_path) == "."
