from __future__ import annotations

import shutil

import pytest

from repair_agent.sandbox.workspace import Workspace
from repair_agent.tools.base import ToolError
from repair_agent.tools.search import SearchCode

pytestmark = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")


def search(ws: Workspace, max_results: int = 100, **kw: str | None):
    tool = SearchCode(ws.root, max_results=max_results, timeout_s=20)
    return tool.run(tool.Args(**kw))


def test_finds_matches_with_posix_paths(workspace: Workspace) -> None:
    out = search(workspace, pattern=r"def mean")
    assert out.content == "src/calc/stats.py:1:def mean(xs):"
    assert out.metadata["matches"] == 1


def test_glob_restricts_files(workspace: Workspace) -> None:
    out = search(workspace, pattern="mean", glob="tests/**")
    lines = out.content.splitlines()
    assert lines and all(line.startswith("tests/test_stats.py:") for line in lines)


def test_no_matches_is_not_an_error(workspace: Workspace) -> None:
    out = search(workspace, pattern="definitely_not_here")
    assert not out.is_error
    assert out.content == "No matches for pattern 'definitely_not_here'."


def test_bad_regex_is_a_tool_error(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match="search failed"):
        search(workspace, pattern="(unclosed")


def test_result_cap(workspace: Workspace) -> None:
    out = search(workspace, max_results=2, pattern="def ")
    lines = out.content.splitlines()
    assert len(lines) == 3
    assert lines[-1].startswith("[... ")
    assert "more matches not shown" in lines[-1]
    assert out.truncated


def test_pattern_starting_with_dash_is_a_pattern_not_a_flag(workspace: Workspace) -> None:
    out = search(workspace, pattern="--version")
    assert out.content.startswith("No matches")


def test_does_not_search_git_dir(workspace: Workspace) -> None:
    out = search(workspace, pattern="initial")  # commit message lives in .git
    assert out.content.startswith("No matches")


def test_missing_ripgrep_gives_install_hint(workspace: Workspace) -> None:
    tool = SearchCode(workspace.root, max_results=10, timeout_s=5, rg_path="")
    tool.rg_path = None
    with pytest.raises(ToolError, match=r"ripgrep .* not installed"):
        tool.run(tool.Args(pattern="x"))
