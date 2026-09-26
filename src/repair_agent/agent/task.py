"""Benchmark task definition, loaded from YAML."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import yaml
from pydantic import BaseModel, Field, field_validator

_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")


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
    issue: Issue
    seed_patch: str | None = Field(default=None, description="Unified diff that plants the bug.")
    fail_to_pass: list[str] = Field(min_length=1)
    pass_to_pass: list[str] = Field(default_factory=list)
    image: str | None = Field(default=None, description="Override the sandbox image (Phase 4).")

    source_path: Path | None = Field(default=None, exclude=True)

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


def normalize_patch(patch: str) -> str:
    """Restore the leading space on blank context lines inside hunks.

    YAML files and editors commonly strip trailing whitespace, turning a blank context
    line (``" "``) into ``""``, which ``git apply`` rejects. Trailing blank lines after
    the last hunk are dropped.
    """
    out: list[str] = []
    in_hunk = False
    for line in patch.replace("\r\n", "\n").split("\n"):
        if line.startswith("@@"):
            in_hunk = True
        elif line.startswith(("--- ", "+++ ", "diff ")):
            in_hunk = False
        out.append(" " if in_hunk and line == "" else line)
    while out and out[-1].strip() == "":
        out.pop()
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
