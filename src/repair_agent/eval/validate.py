"""Validate benchmark tasks in the real sandbox, with no LLM involved.

For every task we prove, automatically:

1. the correct (checked-in) code passes its whole suite, deterministically;
2. the seed patch applies, touches only source files, and makes exactly the
   FAIL_TO_PASS tests fail while every PASS_TO_PASS test keeps passing;
3. the grading pipeline agrees: the buggy workspace grades ``resolved=False`` and a
   perfect fix (the seed patch reverted) grades ``resolved=True``;
4. the issue text describes symptoms only (see :mod:`repair_agent.eval.leaks`).

Results are cached by a content hash of the task file, the repo tree, and the image,
so unchanged tasks are not re-validated before every eval.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from repair_agent.agent.grading import grade
from repair_agent.agent.task import Task, is_test_file
from repair_agent.eval.leaks import find_leaks, patched_files
from repair_agent.sandbox.docker import DockerSandbox, SandboxError
from repair_agent.sandbox.images import repo_image_tag, task_sandbox
from repair_agent.sandbox.junit import Outcome
from repair_agent.sandbox.workspace import Workspace, WorkspaceError, git

VALIDATOR_VERSION = "1"


class TaskValidation(BaseModel):
    """Outcome of validating one task."""

    task_id: str
    ok: bool
    problems: list[str] = Field(default_factory=list)
    clean_tests: int = 0
    measured_fail_to_pass: list[str] = Field(default_factory=list)
    measured_pass_to_pass: list[str] = Field(default_factory=list)
    cache_key: str = ""
    validated_at: datetime | None = None
    duration_s: float = 0.0


def repo_tree_hash(repo_dir: Path) -> str:
    """sha256 over every file's relative path and bytes (caches and bytecode excluded)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(repo_dir).rglob("*") if p.is_file()):
        rel = path.relative_to(repo_dir).as_posix()
        if "__pycache__" in rel or rel.endswith(".pyc"):
            continue
        digest.update(rel.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def cache_key(task: Task, base_image: str, python_version: str | None = None) -> str:
    repo_dir = task.repo_dir()
    parts = [
        VALIDATOR_VERSION,
        task.file_hash or "",
        repo_tree_hash(repo_dir),
        task.image or repo_image_tag(repo_dir, base_image, python_version) or base_image,
        python_version or "",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _minus(a: list[str], b: list[str]) -> list[str]:
    return sorted(set(a) - set(b))


def _passed_ids(outcomes: dict[str, Outcome]) -> set[str]:
    return {k for k, v in outcomes.items() if v == Outcome.PASSED}


def validate_task(
    task: Task, base_sandbox: DockerSandbox, *, repeat: int = 2, work_dir: Path | None = None
) -> TaskValidation:
    """Run every check for one task. Never raises for task problems; they are reported."""
    started = time.perf_counter()
    problems: list[str] = list(f"leak: {p}" for p in find_leaks(task))
    result = TaskValidation(
        task_id=task.id,
        ok=False,
        cache_key=cache_key(task, base_sandbox.settings.image, base_sandbox.python_version()),
    )

    patch = task.seed_patch or ""
    if not patch:
        problems.append("task has no seed_patch")
    touched = patched_files(patch)
    if any(is_test_file(p) for p in touched):
        problems.append(f"seed patch touches test files: {[p for p in touched if is_test_file(p)]}")

    try:
        sandbox = task_sandbox(base_sandbox, task.image, task.repo_dir())
        clean_passed, clean_n = _check_clean(task, sandbox, repeat, work_dir, problems)
        if patch:
            _check_buggy(task, sandbox, clean_passed, work_dir, problems, result)
        result.clean_tests = clean_n
    except (SandboxError, WorkspaceError, FileNotFoundError) as exc:
        problems.append(f"could not run: {exc}")

    result.problems = problems
    result.ok = not problems
    result.validated_at = datetime.now(UTC)
    result.duration_s = round(time.perf_counter() - started, 2)
    return result


def _check_clean(
    task: Task, sandbox: DockerSandbox, repeat: int, work_dir: Path | None, problems: list[str]
) -> tuple[set[str], int]:
    with Workspace.from_directory(task.repo_dir(), parent_dir=work_dir) as ws:
        runs = [sandbox.run_pytest(ws.root) for _ in range(max(1, repeat))]
    first = runs[0]
    if first.exit_code != 0 or first.count(Outcome.PASSED) != len(first.outcomes):
        failing = sorted(k for k, v in first.outcomes.items() if v != Outcome.PASSED)
        problems.append(f"correct code does not pass its suite (exit {first.exit_code}): {failing}")
    if not first.outcomes:
        problems.append("no tests collected on the correct code")
    for other in runs[1:]:
        if other.outcomes != first.outcomes:
            problems.append("suite is not deterministic on the correct code")
            break
    clean_passed = _passed_ids(first.outcomes)
    missing = [nid for nid in task.graded_tests if nid not in first.outcomes]
    if missing:
        problems.append(f"graded tests not collected on the correct code: {missing}")
    return clean_passed, len(first.outcomes)


def _check_buggy(
    task: Task,
    sandbox: DockerSandbox,
    clean_passed: set[str],
    work_dir: Path | None,
    problems: list[str],
    result: TaskValidation,
) -> None:
    try:
        ws = Workspace.from_directory(task.repo_dir(), patch=task.seed_patch, parent_dir=work_dir)
    except WorkspaceError as exc:
        problems.append(f"seed patch does not apply: {exc}")
        return
    with ws:
        buggy = sandbox.run_pytest(ws.root)
        buggy_passed = _passed_ids(buggy.outcomes)
        f2p = sorted(clean_passed - buggy_passed)
        p2p = sorted(clean_passed & buggy_passed)
        result.measured_fail_to_pass, result.measured_pass_to_pass = f2p, p2p
        if not f2p:
            problems.append("seed patch does not make any test fail")
        if set(task.fail_to_pass) != set(f2p):
            problems.append(
                f"FAIL_TO_PASS mismatch: declared-only {_minus(task.fail_to_pass, f2p)}, "
                f"measured-only {_minus(f2p, task.fail_to_pass)}"
            )
        if set(task.pass_to_pass) != set(p2p):
            problems.append(
                f"PASS_TO_PASS mismatch: declared-only {_minus(task.pass_to_pass, p2p)}, "
                f"measured-only {_minus(p2p, task.pass_to_pass)}"
            )
        # Grade with the measured lists so the pipeline check is meaningful even before
        # --fill-tests has written them.
        measured = task.model_copy(
            update={"fail_to_pass": f2p or task.fail_to_pass, "pass_to_pass": p2p}
        )
        if grade(measured, ws, sandbox).resolved:
            problems.append("grading says the buggy code is resolved")
        git(ws.root, "apply", "-R", "--whitespace=nowarn", "-", input_text=task.seed_patch)
        if not grade(measured, ws, sandbox).resolved:
            problems.append("grading says a perfect fix (seed patch reverted) is not resolved")


# --- YAML write-back ----------------------------------------------------------------

_LISTS_START = re.compile(r"^fail_to_pass:", re.MULTILINE)


def write_test_lists(task: Task, f2p: list[str], p2p: list[str]) -> None:
    """Rewrite the trailing ``fail_to_pass``/``pass_to_pass`` block of the task YAML."""
    if task.source_path is None:
        raise ValueError("task has no source file")
    text = task.source_path.read_text(encoding="utf-8")
    match = _LISTS_START.search(text)
    if match is None:
        raise ValueError(f"{task.source_path}: no fail_to_pass block found")

    def fmt(name: str, ids: list[str]) -> str:
        if not ids:
            return f"{name}: []\n"
        return f"{name}:\n" + "".join(f"  - {i}\n" for i in ids)

    new = text[: match.start()] + fmt("fail_to_pass", f2p) + fmt("pass_to_pass", p2p)
    task.source_path.write_text(new, encoding="utf-8", newline="\n")


# --- cache ----------------------------------------------------------------------------


class ValidationCache:
    """JSON file of the latest validation per task, keyed by content hash."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        if self.path.is_file():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, task_id: str, key: str) -> TaskValidation | None:
        entry = self._data.get(task_id)
        if entry and entry.get("cache_key") == key:
            return TaskValidation.model_validate(entry)
        return None

    def put(self, result: TaskValidation) -> None:
        self._data[result.task_id] = result.model_dump(mode="json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")


def ensure_validated(
    tasks: list[Task],
    base_sandbox: DockerSandbox,
    cache: ValidationCache,
    *,
    on_result: Callable[[TaskValidation, bool], None] | None = None,
) -> list[TaskValidation]:
    """Validate tasks whose cached result is missing or stale; return every result."""
    results = []
    for task in tasks:
        key = cache_key(task, base_sandbox.settings.image, base_sandbox.python_version())
        cached = cache.get(task.id, key)
        if cached is None:
            cached = validate_task(task, base_sandbox)
            cache.put(cached)
            fresh = True
        else:
            fresh = False
        if on_result:
            on_result(cached, fresh)
        results.append(cached)
    return results
