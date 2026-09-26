"""list_files, read_file, edit_file against the calc fixture workspace (no Docker)."""

from __future__ import annotations

from pathlib import Path

import pytest

from repair_agent.sandbox.workspace import Workspace
from repair_agent.tools.base import ToolError
from repair_agent.tools.files import EditFile, ListFiles, ReadFile


def list_files(root: Path, max_entries: int = 500, **kw: object) -> str:
    tool = ListFiles(root, max_entries=max_entries)
    return tool.run(tool.Args(**kw)).content


def read(root: Path, max_lines: int = 400, **kw: object):
    tool = ReadFile(root, max_lines=max_lines)
    return tool.run(tool.Args(**kw))


def edit(root: Path, **kw: str):
    tool = EditFile(root)
    return tool.run(tool.Args(**kw))


# --- list_files -------------------------------------------------------------


def test_list_files_tree(workspace: Workspace) -> None:
    (workspace.root / "src" / "calc" / "__pycache__").mkdir()
    (workspace.root / "src" / "calc" / "junk.pyc").write_bytes(b"x")
    assert list_files(workspace.root, depth=3) == "\n".join(
        [
            "src/",
            "  calc/",
            "    __init__.py",
            "    ops.py",
            "    stats.py",
            "tests/",
            "  test_ops.py",
            "  test_stats.py",
        ]
    )


def test_list_files_depth_and_subdir(workspace: Workspace) -> None:
    assert list_files(workspace.root, depth=1) == "src/\ntests/"
    assert list_files(workspace.root, path="src/calc", depth=1) == "__init__.py\nops.py\nstats.py"


def test_list_files_entry_cap(workspace: Workspace) -> None:
    out = list_files(workspace.root, max_entries=3, depth=3)
    assert out.splitlines()[:3] == ["src/", "  calc/", "    __init__.py"]
    assert "[... 5 more entries not shown" in out


@pytest.mark.parametrize(
    ("path", "message"), [("nope", "path not found"), ("src/calc/ops.py", "not a directory")]
)
def test_list_files_errors(workspace: Workspace, path: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        list_files(workspace.root, path=path)


# --- read_file --------------------------------------------------------------


def test_read_file_numbers_lines(workspace: Workspace) -> None:
    out = read(workspace.root, path="src/calc/stats.py", end_line=4)
    assert out.content.splitlines()[0] == "src/calc/stats.py (lines 1-4 of 12)"
    assert out.content.splitlines()[4] == "     4\t    return sum(xs) / (len(xs) - 1)"
    assert not out.truncated


def test_read_file_pages_long_files(workspace: Workspace) -> None:
    big = workspace.root / "big.py"
    big.write_bytes(b"".join(f"x = {i}\n".encode() for i in range(1, 1001)))
    first = read(workspace.root, max_lines=400, path="big.py")
    assert first.content.splitlines()[0] == (
        "big.py (lines 1-400 of 1000; call again with start_line=401 for more)"
    )
    assert first.truncated
    last = read(workspace.root, max_lines=400, path="big.py", start_line=901)
    assert last.content.splitlines()[0] == "big.py (lines 901-1000 of 1000)"
    assert not last.truncated


def test_read_file_crlf_is_noted_and_numbered_correctly(workspace: Workspace) -> None:
    (workspace.root / "win.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    out = read(workspace.root, path="win.py").content
    assert out == "win.py (lines 1-2 of 2) [CRLF line endings]\n     1\ta = 1\n     2\tb = 2"


@pytest.mark.parametrize(
    ("setup", "kwargs", "message"),
    [
        (None, {"path": "missing.py"}, "file not found"),
        (None, {"path": "src"}, "is a directory"),
        (b"\x89PNG\x00\x00data", {"path": "blob.bin"}, "binary"),
        (None, {"path": "src/calc/ops.py", "start_line": 99}, "start_line 99 > file length"),
    ],
)
def test_read_file_errors(
    workspace: Workspace, setup: bytes | None, kwargs: dict, message: str
) -> None:
    if setup is not None:
        (workspace.root / kwargs["path"]).write_bytes(setup)
    with pytest.raises(ToolError, match=message):
        read(workspace.root, **kwargs)


def test_read_file_non_utf8_is_lossy_not_fatal(workspace: Workspace) -> None:
    (workspace.root / "latin.txt").write_bytes("café\n".encode("latin-1"))
    out = read(workspace.root, path="latin.txt").content
    assert "not valid UTF-8" in out
    assert "caf�" in out


# --- edit_file --------------------------------------------------------------


def test_edit_file_replaces_and_shows_context(workspace: Workspace) -> None:
    out = edit(
        workspace.root,
        path="src/calc/stats.py",
        old_str="return sum(xs) / (len(xs) - 1)",
        new_str="return sum(xs) / len(xs)",
    )
    assert out.content.startswith("Edited src/calc/stats.py (1 replacement). Lines 1-7 now:")
    assert "     4\t    return sum(xs) / len(xs)" in out.content
    assert b"/ len(xs)\n" in (workspace.root / "src/calc/stats.py").read_bytes()
    assert b"\r\n" not in (workspace.root / "src/calc/stats.py").read_bytes()


def test_edit_file_no_match(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match=r"old_str not found in src/calc/ops\.py"):
        edit(workspace.root, path="src/calc/ops.py", old_str="return a * b", new_str="x")


def test_edit_file_multiple_matches_lists_lines(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match=r"matches 2 times .*\(lines 11, 13\)"):
        edit(workspace.root, path="src/calc/ops.py", old_str="        return ", new_str="  ret ")


def test_edit_file_identical_strings(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match="identical"):
        edit(workspace.root, path="src/calc/ops.py", old_str="a + b", new_str="a + b")


def test_edit_file_preserves_crlf_byte_for_byte(workspace: Workspace) -> None:
    target = workspace.root / "win.py"
    target.write_bytes(b"def f():\r\n    return 1\r\n\r\nx = f()\r\n")
    edit(
        workspace.root,
        path="win.py",
        old_str="def f():\n    return 1",
        new_str="def f():\n    return 2",
    )
    assert target.read_bytes() == b"def f():\r\n    return 2\r\n\r\nx = f()\r\n"


def test_edit_file_creates_new_file(workspace: Workspace) -> None:
    out = edit(workspace.root, path="src/calc/new_mod.py", old_str="", new_str="X = 1\nY = 2\n")
    assert out.content == "Created src/calc/new_mod.py (2 lines)."
    assert (workspace.root / "src/calc/new_mod.py").read_bytes() == b"X = 1\nY = 2\n"


def test_edit_file_empty_old_str_on_existing_file_is_rejected(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match="already exists"):
        edit(workspace.root, path="src/calc/ops.py", old_str="", new_str="oops")


def test_edit_file_rejects_escape(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match="outside"):
        edit(workspace.root, path="../evil.py", old_str="", new_str="x")
