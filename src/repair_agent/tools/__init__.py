"""Agent tools and the registry that exposes them to the LLM."""

from __future__ import annotations

from pathlib import Path

from repair_agent.config import Settings
from repair_agent.sandbox.docker import DockerSandbox
from repair_agent.tools.base import Tool, ToolError, ToolOutput, ToolRegistry
from repair_agent.tools.files import EditFile, ListFiles, ReadFile
from repair_agent.tools.finish import Finish
from repair_agent.tools.search import SearchCode
from repair_agent.tools.tests_tool import RunTests


def build_registry(
    workspace_root: Path, sandbox: DockerSandbox, settings: Settings
) -> ToolRegistry:
    """All six agent tools, bound to one workspace and sandbox."""
    t = settings.tools
    root = Path(workspace_root).resolve()
    tools: list[Tool] = [
        ListFiles(root, max_entries=t.max_list_entries),
        ReadFile(root, max_lines=t.max_read_lines),
        SearchCode(root, max_results=t.max_search_results, timeout_s=t.search_timeout_s),
        EditFile(root),
        RunTests(root, sandbox),
        Finish(),
    ]
    return ToolRegistry(tools, max_output_chars=t.max_output_chars)


__all__ = ["Tool", "ToolError", "ToolOutput", "ToolRegistry", "build_registry"]
