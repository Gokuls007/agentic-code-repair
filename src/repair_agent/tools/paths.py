"""Confine tool paths to the workspace.

The model always speaks POSIX-style relative paths (``src/calc/ops.py``). These helpers
map them onto the host filesystem and back, rejecting anything that would escape the
workspace or touch git internals.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from repair_agent.tools.base import ToolError


def resolve_in_workspace(root: Path, path: str) -> Path:
    """Map a model-supplied relative path to an absolute host path inside ``root``.

    Raises:
        ToolError: for absolute paths, drive letters, escapes via ``..`` or symlinks,
            and anything inside ``.git``.
    """
    raw = (path or ".").strip().replace("\\", "/")
    if raw.startswith("/") or PureWindowsPath(raw).drive:
        raise ToolError(f"path must be relative to the repository root, got {path!r}")
    root = root.resolve()
    candidate = (root / PurePosixPath(raw)).resolve()
    if not candidate.is_relative_to(root):
        raise ToolError(f"path {path!r} is outside the repository")
    rel = candidate.relative_to(root)
    if rel.parts and rel.parts[0] == ".git":
        raise ToolError("the .git directory is off-limits")
    return candidate


def display_path(root: Path, path: Path) -> str:
    """POSIX-style path relative to ``root``, as shown to the model."""
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    return rel or "."
