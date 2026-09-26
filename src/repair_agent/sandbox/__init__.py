"""Sandboxed execution: host workspace + throwaway Docker containers."""

from repair_agent.sandbox.docker import DockerSandbox, ExecResult, SandboxError
from repair_agent.sandbox.junit import Outcome, TestReport
from repair_agent.sandbox.workspace import Workspace, WorkspaceError

__all__ = [
    "DockerSandbox",
    "ExecResult",
    "Outcome",
    "SandboxError",
    "TestReport",
    "Workspace",
    "WorkspaceError",
]
