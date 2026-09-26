"""Host-side working copy of the target repo.

The agent's file tools read and edit this copy directly; ``run_tests`` ships a snapshot
of it into a fresh container. Git is invoked with line-ending conversion disabled so
the bytes on disk are exactly the bytes in the repository.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

# Force byte-exact checkouts regardless of the user's global git config (e.g. autocrlf=true
# on Windows), and never prompt for credentials.
_GIT_CONFIG = ["-c", "core.autocrlf=false", "-c", "core.eol=lf", "-c", "core.safecrlf=false"]
_GIT_ENV = {"GIT_TERMINAL_PROMPT": "0"}


class WorkspaceError(RuntimeError):
    """A git operation needed to prepare the workspace failed."""


def git(cwd: Path, *args: str, input_text: str | None = None) -> str:
    """Run a git command in ``cwd`` and return stdout. Raises WorkspaceError on failure."""
    result = subprocess.run(
        ["git", *_GIT_CONFIG, *args],
        cwd=cwd,
        input=input_text.encode("utf-8") if input_text is not None else None,
        capture_output=True,
        env={**os.environ, **_GIT_ENV},
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorkspaceError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout.decode("utf-8", errors="replace")


class Workspace:
    """A disposable git checkout the agent works in. Use as a context manager."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    @classmethod
    def create(
        cls,
        source_repo: Path,
        *,
        base_commit: str | None = None,
        patch: str | None = None,
        parent_dir: Path | None = None,
    ) -> Workspace:
        """Clone ``source_repo`` into a fresh temp dir and prepare it for a task.

        Checks out ``base_commit`` if given, then applies ``patch`` (e.g. a seeded bug)
        and commits it, so that :meth:`diff` shows only the agent's edits.
        """
        root = Path(tempfile.mkdtemp(prefix="repair-ws-", dir=parent_dir))
        try:
            git(root.parent, *_clone_args(Path(source_repo), root))
            if base_commit:
                git(root, "checkout", "--quiet", "--detach", base_commit)
            if patch:
                git(root, "apply", "--whitespace=nowarn", "-", input_text=patch)
                git(root, "add", "-A")
                git(
                    root,
                    "-c",
                    "user.name=repair-agent",
                    "-c",
                    "user.email=repair-agent@localhost",
                    "commit",
                    "--quiet",
                    "--no-verify",
                    "-m",
                    "task setup",
                )
        except Exception:
            _rmtree(root)
            raise
        return cls(root)

    def diff(self) -> str:
        """Unified diff of everything the agent changed, including new files."""
        git(self.root, "add", "--intent-to-add", "-A")
        return git(self.root, "diff", "--no-color", "--no-ext-diff")

    def cleanup(self) -> None:
        """Delete the workspace directory."""
        _rmtree(self.root)

    def __enter__(self) -> Workspace:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.cleanup()


def _clone_args(source: Path, dest: Path) -> list[str]:
    return ["clone", "--quiet", "--no-hardlinks", str(source.resolve()), str(dest)]


def _rmtree(path: Path) -> None:
    """Remove a tree, clearing the read-only bit git sets on objects (Windows)."""

    def _on_error(func: Callable[[str], object], target: str, _exc: object) -> None:
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_error)
    else:  # pragma: no cover - 3.11 only
        shutil.rmtree(path, onerror=_on_error)
