"""File tools: list_files, read_file, edit_file.

These run on the host workspace and never execute repo code. All I/O is done in bytes
so line endings survive exactly (Python text mode on Windows would rewrite ``\\n`` as
``\\r\\n``).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from repair_agent.tools.base import Tool, ToolArgs, ToolError, ToolOutput
from repair_agent.tools.paths import display_path, resolve_in_workspace

SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache"}
)
SKIP_SUFFIXES = (".pyc", ".pyo")
EDIT_CONTEXT_LINES = 3


def is_binary(data: bytes) -> bool:
    """Heuristic used by git: a NUL byte in the first 8 KB means binary."""
    return b"\x00" in data[:8192]


def split_lines(text: str) -> list[str]:
    """Split on LF/CRLF only (unlike str.splitlines, which also splits on \\f, \\u2028...)."""
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def number_lines(lines: list[str], first: int) -> str:
    """Format lines ``cat -n`` style: right-aligned number, tab, text."""
    return "\n".join(f"{first + i:>6}\t{line}" for i, line in enumerate(lines))


def _decode(data: bytes) -> tuple[str, bool]:
    """Decode UTF-8, falling back to replacement characters. Returns (text, was_lossy)."""
    try:
        return data.decode("utf-8"), False
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), True


class ListFiles(Tool):
    """Directory tree of the repository."""

    name = "list_files"
    description = (
        "List files and directories as an indented tree. Directories end with '/'. "
        "Skips .git, caches, and virtualenvs. Use a subdirectory path and small depth "
        "to explore large repos."
    )

    class Args(ToolArgs):
        path: str = Field(default=".", description="Directory relative to the repo root.")
        depth: int = Field(default=2, ge=1, le=5, description="How many levels to descend.")

    def __init__(self, root: Path, max_entries: int):
        self.root = root
        self.max_entries = max_entries

    def run(self, args: Args) -> ToolOutput:
        target = resolve_in_workspace(self.root, args.path)
        if not target.exists():
            raise ToolError(f"path not found: {args.path}")
        if not target.is_dir():
            raise ToolError(f"not a directory: {args.path} (use read_file)")

        lines: list[str] = []
        total = self._walk(target, args.depth, 0, lines)
        shown = lines[: self.max_entries]
        if not shown:
            return ToolOutput(content=f"{display_path(self.root, target)}/ is empty.")
        content = "\n".join(shown)
        if total > len(shown):
            content += (
                f"\n[... {total - len(shown)} more entries not shown; "
                "list a subdirectory or reduce depth ...]"
            )
        return ToolOutput(content=content, truncated=total > len(shown))

    def _walk(self, directory: Path, depth: int, level: int, out: list[str]) -> int:
        """Append entries under ``directory`` to ``out``; return the total entry count."""
        try:
            children = list(directory.iterdir())
        except OSError:
            return 0
        dirs = sorted(
            # Symlinked dirs are not followed: they could lead outside the workspace.
            (c for c in children if c.is_dir() and not c.is_symlink() and c.name not in SKIP_DIRS),
            key=lambda p: p.name,
        )
        files = sorted(
            (c for c in children if c.is_file() and not c.name.endswith(SKIP_SUFFIXES)),
            key=lambda p: p.name,
        )
        count = 0
        indent = "  " * level
        for d in dirs:
            count += 1
            if len(out) < self.max_entries:
                out.append(f"{indent}{d.name}/")
            if level + 1 < depth:
                count += self._walk(d, depth, level + 1, out)
        for f in files:
            count += 1
            if len(out) < self.max_entries:
                out.append(f"{indent}{f.name}")
        return count


class ReadFile(Tool):
    """Read a file (or a line range) with line numbers."""

    name = "read_file"
    description = (
        "Read a text file with line numbers (number, tab, text). Long files are paged: "
        "the header says which lines are shown and how to get more. Line numbers are "
        "for reference only; do not include them in edit_file's old_str."
    )

    class Args(ToolArgs):
        path: str = Field(description="File path relative to the repo root.")
        start_line: int = Field(default=1, ge=1, description="First line to show (1-based).")
        end_line: int | None = Field(
            default=None, ge=1, description="Last line to show, inclusive. Omit for a full page."
        )

    def __init__(self, root: Path, max_lines: int):
        self.root = root
        self.max_lines = max_lines

    def run(self, args: Args) -> ToolOutput:
        target = resolve_in_workspace(self.root, args.path)
        shown_path = display_path(self.root, target)
        if not target.exists():
            raise ToolError(f"file not found: {args.path}")
        if target.is_dir():
            raise ToolError(f"{args.path} is a directory (use list_files)")
        data = target.read_bytes()
        if is_binary(data):
            raise ToolError(f"{shown_path} looks like a binary file ({len(data):,} bytes)")

        text, lossy = _decode(data)
        lines = split_lines(text)
        total = len(lines)
        notes = []
        if b"\r\n" in data:
            notes.append("[CRLF line endings]")
        if lossy:
            notes.append("[not valid UTF-8; undecodable bytes shown as �]")
        suffix = (" " + " ".join(notes)) if notes else ""

        if total == 0:
            return ToolOutput(content=f"{shown_path} (empty file){suffix}")
        if args.start_line > total:
            raise ToolError(f"start_line {args.start_line} > file length {total} ({shown_path})")
        if args.end_line is not None and args.end_line < args.start_line:
            raise ToolError("end_line must be >= start_line")

        requested_end = min(args.end_line or total, total)
        end = min(requested_end, args.start_line + self.max_lines - 1)
        paged = end < requested_end or (args.end_line is None and end < total)
        header = f"{shown_path} (lines {args.start_line}-{end} of {total}"
        if paged:
            header += f"; call again with start_line={end + 1} for more"
        header += f"){suffix}"
        body = number_lines(lines[args.start_line - 1 : end], args.start_line)
        return ToolOutput(
            content=f"{header}\n{body}",
            truncated=paged,
            metadata={"path": shown_path, "lines": [args.start_line, end], "total_lines": total},
        )


class EditFile(Tool):
    """Exact-match string replacement."""

    name = "edit_file"
    description = (
        "Replace an exact snippet in a file. old_str must match the file exactly "
        "(including indentation) and occur exactly once; include enough surrounding "
        'lines to make it unique. To create a new file, pass old_str="" and a path '
        "that does not exist yet."
    )

    class Args(ToolArgs):
        path: str = Field(description="File path relative to the repo root.")
        old_str: str = Field(description="Exact text to replace (empty only to create a file).")
        new_str: str = Field(description="Replacement text.")

    def __init__(self, root: Path):
        self.root = root

    def run(self, args: Args) -> ToolOutput:
        target = resolve_in_workspace(self.root, args.path)
        shown_path = display_path(self.root, target)
        if args.old_str == "":
            return self._create(target, shown_path, args.new_str)
        if args.old_str == args.new_str:
            raise ToolError("old_str and new_str are identical; nothing to change")
        if not target.is_file():
            raise ToolError(f"file not found: {args.path}")

        data = target.read_bytes()
        if is_binary(data):
            raise ToolError(f"{shown_path} is a binary file; refusing to edit")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"{shown_path} is not valid UTF-8; refusing to edit") from exc

        old, new = args.old_str, args.new_str
        if _uses_crlf(data):  # match the model's LF text against the file's CRLF lines
            old, new = _to_crlf(old), _to_crlf(new)

        positions = _find_all(text, old)
        if not positions:
            raise ToolError(
                f"old_str not found in {shown_path}. Re-read the file: whitespace and "
                "indentation must match exactly."
            )
        if len(positions) > 1:
            lines = ", ".join(str(text.count("\n", 0, p) + 1) for p in positions[:10])
            raise ToolError(
                f"old_str matches {len(positions)} times in {shown_path} (lines {lines}); "
                "include more surrounding lines so it is unique."
            )

        start = positions[0]
        updated = text[:start] + new + text[start + len(old) :]
        target.write_bytes(updated.encode("utf-8"))

        first_line = text.count("\n", 0, start) + 1
        new_line_count = max(1, new.count("\n") + (0 if new.endswith("\n") else 1))
        return ToolOutput(
            content=_edited_snippet(updated, shown_path, first_line, new_line_count),
            metadata={
                "path": shown_path,
                "lines_changed": max(old.count("\n"), new.count("\n")) + 1,
            },
        )

    def _create(self, target: Path, shown_path: str, content: str) -> ToolOutput:
        if target.exists():
            raise ToolError(
                f"{shown_path} already exists; old_str may only be empty when creating a new file"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
        line_count = len(split_lines(content))
        return ToolOutput(
            content=f"Created {shown_path} ({line_count} lines).",
            metadata={"path": shown_path, "created": True, "lines_changed": line_count},
        )


def _uses_crlf(data: bytes) -> bool:
    """True if every newline in the file is CRLF."""
    return b"\r\n" in data and data.count(b"\r\n") == data.count(b"\n")


def _to_crlf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def _find_all(text: str, needle: str) -> list[int]:
    positions, start = [], 0
    while (index := text.find(needle, start)) != -1:
        positions.append(index)
        start = index + 1
    return positions


def _edited_snippet(text: str, shown_path: str, first_line: int, line_count: int) -> str:
    lines = split_lines(text)
    lo = max(1, first_line - EDIT_CONTEXT_LINES)
    hi = min(len(lines), first_line + line_count - 1 + EDIT_CONTEXT_LINES)
    body = number_lines(lines[lo - 1 : hi], lo)
    return f"Edited {shown_path} (1 replacement). Lines {lo}-{hi} now:\n{body}"
