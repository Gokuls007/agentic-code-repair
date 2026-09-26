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
        self._baseline: str | None = None

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
                _commit_all(root, "task setup")
        except Exception:
            _rmtree(root)
            raise
        workspace = cls(root)
        _ = workspace.baseline  # pin the starting commit now
        return workspace

    @classmethod
    def from_directory(
        cls, source_dir: Path, *, patch: str | None = None, parent_dir: Path | None = None
    ) -> Workspace:
        """Create a workspace from a plain directory (e.g. ``benchmark/repos/calc``).

        The directory is committed as a fresh repository first, then ``patch`` is applied
        and committed, exactly as :meth:`create` does for git sources.
        """
        staging = Path(tempfile.mkdtemp(prefix="repair-src-", dir=parent_dir))
        try:
            source = staging / "repo"
            shutil.copytree(
                source_dir, source, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc")
            )
            git(source, "init", "--quiet", "--initial-branch=main")
            _commit_all(source, "import")
            return cls.create(source, patch=patch, parent_dir=parent_dir)
        finally:
            _rmtree(staging)

    @property
    def baseline(self) -> str:
        """Commit the agent started from (task setup included)."""
        if self._baseline is None:
            self._baseline = git(self.root, "rev-parse", "HEAD").strip()
        return self._baseline

    def diff(self) -> str:
        """Unified diff of everything the agent changed, including new files."""
        git(self.root, "add", "--intent-to-add", "-A")
        return git(self.root, "diff", "--no-color", "--no-ext-diff", self.baseline)

    def baseline_files(self) -> list[str]:
        """POSIX paths of every file tracked at the baseline commit."""
        return git(self.root, "ls-tree", "-r", "--name-only", self.baseline).splitlines()

    def changed_files(self, paths: list[str]) -> list[str]:
        """Which of ``paths`` differ from the baseline (modified or deleted)."""
        if not paths:
            return []
        git(self.root, "add", "--intent-to-add", "-A")
        out = git(self.root, "diff", "--name-only", self.baseline, "--", *paths)
        return sorted(set(out.splitlines()))

    def all_changed_files(self) -> list[str]:
        """Every path that differs from the baseline: modified, deleted, or added."""
        git(self.root, "add", "--intent-to-add", "-A")
        return sorted(set(git(self.root, "diff", "--name-only", self.baseline).splitlines()))

    def added_files(self) -> list[str]:
        """Paths that exist now but not at the baseline."""
        git(self.root, "add", "--intent-to-add", "-A")
        out = git(self.root, "diff", "--name-only", "--diff-filter=A", self.baseline)
        return sorted(set(out.splitlines()))

    def restore(self, paths: list[str]) -> None:
        """Reset ``paths`` (which must exist at the baseline) to their baseline content."""
        if paths:
            git(self.root, "checkout", self.baseline, "--", *paths)

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


def _commit_all(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(
        repo,
        "-c",
        "user.name=repair-agent",
        "-c",
        "user.email=repair-agent@localhost",
        "commit",
        "--quiet",
        "--no-verify",
        "--allow-empty",
        "-m",
        message,
    )


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
