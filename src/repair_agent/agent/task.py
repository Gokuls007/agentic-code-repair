"""Benchmark task definition, loaded from YAML."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")

BugType = Literal[
    "off-by-one", "wrong-conditional", "missing-edge-case", "api-misuse", "multi-file"
]
Difficulty = Literal["easy", "medium", "hard"]


class Issue(BaseModel):
    """What the agent is told: a bug report, written the way a user would."""

    title: str = Field(min_length=1)
    body: str = Field(min_length=1)


class Task(BaseModel):
    """One repair task.

    ``fail_to_pass`` / ``pass_to_pass`` are used only for grading and are never shown
    to the agent.
    """

    id: str
    repo: str = Field(description="Directory under benchmark/repos/, or a path to a directory.")
    bug_type: BugType
    difficulty: Difficulty
    issue: Issue
    seed_patch: str | None = Field(default=None, description="Unified diff that plants the bug.")
    fail_to_pass: list[str] = Field(min_length=1)
    pass_to_pass: list[str] = Field(default_factory=list)
    image: str | None = Field(
        default=None, description="Sandbox image override; default comes from the repo."
    )

    source_path: Path | None = Field(default=None, exclude=True)

    @property
    def file_hash(self) -> str | None:
        """sha256 of the task YAML (identifies the exact task version in eval manifests)."""
        if self.source_path is None:
            return None
        return hashlib.sha256(self.source_path.read_bytes()).hexdigest()

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if not _TASK_ID.match(value):
            raise ValueError(f"task id must be filesystem-safe, got {value!r}")
        return value

    @field_validator("seed_patch")
    @classmethod
    def _normalize_patch(cls, value: str | None) -> str | None:
        return normalize_patch(value) if value else value

    @property
    def graded_tests(self) -> list[str]:
        """All node ids that decide ``resolved``."""
        return [*self.fail_to_pass, *self.pass_to_pass]

    @property
    def graded_test_files(self) -> list[str]:
        """Files containing the graded tests (the part of a node id before ``::``)."""
        return sorted({nid.split("::", 1)[0] for nid in self.graded_tests})

    def repo_dir(self, benchmark_root: Path | None = None) -> Path:
        """Resolve ``repo`` to a directory on disk."""
        direct = Path(self.repo)
        if direct.is_absolute() and direct.is_dir():
            return direct
        root = benchmark_root or _default_benchmark_root(self.source_path)
        candidate = root / "repos" / self.repo
        if not candidate.is_dir():
            raise FileNotFoundError(f"repo {self.repo!r} not found at {candidate}")
        return candidate


def _default_benchmark_root(task_path: Path | None) -> Path:
    """``benchmark/`` is the parent of the ``tasks/`` directory the YAML lives in."""
    if task_path is not None and task_path.parent.name == "tasks":
        return task_path.parent.parent
    return Path("benchmark")


def load_task(path: Path) -> Task:
    """Parse and validate a task YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    task = Task.model_validate(data)
    task.source_path = Path(path).resolve()
    return task


def load_tasks(tasks_dir: Path, pattern: str = "*") -> list[Task]:
    """All tasks in ``tasks_dir`` whose id matches the glob ``pattern``, sorted by id."""
    from fnmatch import fnmatch

    tasks = [load_task(p) for p in sorted(Path(tasks_dir).glob("*.yaml"))]
    return [t for t in tasks if fnmatch(t.id, pattern)]


_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


def normalize_patch(patch: str) -> str:
    """Repair blank context lines that YAML or an editor stripped.

    A blank context line is a single space. Editors turn it into an empty line, and a
    YAML ``|`` block drops trailing blank lines entirely; ``git apply`` rejects both.
    Using each hunk header's line counts, empty lines inside a hunk become ``" "`` and
    missing trailing context lines are re-added.
    """
    out: list[str] = []
    old_left = new_left = 0

    def close_hunk() -> None:
        nonlocal old_left, new_left
        # Only blank context lines can go missing at the end of a hunk.
        while old_left > 0 and old_left == new_left:
            out.append(" ")
            old_left -= 1
            new_left -= 1
        old_left = new_left = 0

    for line in patch.replace("\r\n", "\n").split("\n"):
        header = _HUNK.match(line)
        if header:
            close_hunk()
            old_left = int(header.group(1) or 1)
            new_left = int(header.group(2) or 1)
            out.append(line)
            continue
        if old_left or new_left:
            if line == "":
                line = " "
            if line.startswith("-"):
                old_left -= 1
            elif line.startswith("+"):
                new_left -= 1
            elif line.startswith(" "):
                old_left -= 1
                new_left -= 1
            out.append(line)
            continue
        if line.startswith(("--- ", "+++ ", "diff ", "index ")) or line.strip():
            out.append(line)
    close_hunk()
    return "\n".join(out) + "\n"


def is_test_file(path: str) -> bool:
    """Heuristic for pytest test files, including conftest.py (which can skip tests)."""
    p = PurePosixPath(path)
    if p.suffix != ".py":
        return False
    return (
        p.name == "conftest.py"
        or p.name.startswith("test_")
        or p.stem.endswith("_test")
        or "tests" in p.parts[:-1]
        or "test" in p.parts[:-1]
    )
