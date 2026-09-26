"""search_code: ripgrep over the workspace."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pydantic import Field

from repair_agent.tools.base import Tool, ToolArgs, ToolError, ToolOutput

RG_INSTALL_HINT = (
    "install it with `winget install BurntSushi.ripgrep.MSVC` (or your package manager)"
)


class SearchCode(Tool):
    """Regex search across the repository with ripgrep."""

    name = "search_code"
    description = (
        "Search file contents with a ripgrep regular expression. Returns 'path:line:text' "
        "lines. Respects .gitignore. Use glob to restrict files, e.g. '*.py' or 'tests/**'."
    )

    class Args(ToolArgs):
        pattern: str = Field(min_length=1, description="Rust-regex pattern (ripgrep syntax).")
        glob: str | None = Field(
            default=None, description="Optional file glob, e.g. '*.py'. Prefix '!' to exclude."
        )

    def __init__(self, root: Path, max_results: int, timeout_s: float, rg_path: str | None = None):
        self.root = root
        self.max_results = max_results
        self.timeout_s = timeout_s
        self.rg_path = rg_path or shutil.which("rg")

    def command(self, args: Args) -> list[str]:
        """The exact ripgrep argv (passed as a list, never through a shell)."""
        if not self.rg_path:
            raise ToolError(f"ripgrep (rg) is not installed; {RG_INSTALL_HINT}")
        cmd = [
            self.rg_path,
            "--no-config",
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--path-separator",
            "/",
            "--max-columns",
            "300",
            "--max-columns-preview",
            "--hidden",
            "--glob",
            "!.git",
        ]
        if args.glob:
            cmd += ["--glob", args.glob]
        return [*cmd, "-e", args.pattern, "."]

    def run(self, args: Args) -> ToolOutput:
        try:
            proc = subprocess.run(
                self.command(args),
                cwd=self.root,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"search timed out after {self.timeout_s:.0f}s; use a more specific pattern or glob"
            ) from exc

        if proc.returncode == 1:
            where = f" in files matching {args.glob!r}" if args.glob else ""
            return ToolOutput(
                content=f"No matches for pattern {args.pattern!r}{where}.",
                metadata={"matches": 0},
            )
        if proc.returncode != 0:
            message = proc.stderr.decode("utf-8", errors="replace").strip()
            raise ToolError(f"search failed: {message or f'rg exit code {proc.returncode}'}")

        stdout = proc.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
        matches = [line.removeprefix("./") for line in stdout.splitlines() if line]
        shown = matches[: self.max_results]
        content = "\n".join(shown)
        if len(matches) > len(shown):
            content += (
                f"\n[... {len(matches) - len(shown)} more matches not shown; "
                "refine the pattern or add a glob ...]"
            )
        return ToolOutput(
            content=content,
            truncated=len(matches) > len(shown),
            metadata={"matches": len(matches)},
        )
