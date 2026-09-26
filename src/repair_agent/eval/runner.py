"""The eval runner: N attempts per task, resumable, sequential by default.

Layout::

    runs/eval-<id>/manifest.json               config snapshot, git sha, task hashes
    runs/eval-<id>/<task>/run-<k>/trace.jsonl  step-by-step trace
    runs/eval-<id>/<task>/run-<k>/result.json  AgentResult (a completed attempt)
    runs/eval-<id>/<task>/run-<k>/result.infra.json  attempt lost to infrastructure

An attempt with ``result.json`` is never re-run. An attempt that ended because of the
infrastructure (provider quota or outage, Docker failure) is saved as
``result.infra.json``, excluded from metrics, and re-run on resume; the eval stops at
that point instead of recording the remaining tasks as failures.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from repair_agent import __version__
from repair_agent.agent.loop import solve_task
from repair_agent.agent.state import AgentResult, Outcome, StopReason
from repair_agent.agent.task import Task
from repair_agent.config import Settings
from repair_agent.llm.base import LLMProvider
from repair_agent.sandbox.docker import DockerSandbox, SandboxError
from repair_agent.sandbox.workspace import Workspace, WorkspaceError
from repair_agent.tracing import Tracer, new_run_id

MANIFEST = "manifest.json"
RESULT = "result.json"
INFRA_RESULT = "result.infra.json"
TRACE = "trace.jsonl"


class TaskMeta(BaseModel):
    """Per-task facts recorded in the manifest (for breakdowns and resume checks)."""

    file_hash: str | None
    repo: str
    bug_type: str
    difficulty: str


class EvalManifest(BaseModel):
    eval_id: str
    created_at: datetime
    runs: int
    package_version: str
    git_sha: str | None = None
    git_dirty: bool | None = None
    config: dict[str, Any]
    tasks: dict[str, TaskMeta]
    # Runtime facts (images, interpreter); recorded for reproducibility, not compared on resume.
    environment: dict[str, Any] = Field(default_factory=dict)

    def compatible_with(self, other: EvalManifest) -> list[str]:
        """Reasons ``other`` cannot resume this eval (empty if compatible)."""
        problems = []
        if self.config != other.config:
            changed = sorted(
                k
                for k in self.config.keys() | other.config.keys()
                if self.config.get(k) != other.config.get(k)
            )
            problems.append(f"config differs: {changed}")
        if self.tasks != other.tasks:
            problems.append("task set or task files differ")
        if self.runs != other.runs:
            problems.append(f"runs per task differ ({self.runs} vs {other.runs})")
        return problems


def config_snapshot(settings: Settings, provider_name: str) -> dict[str, Any]:
    """Everything that can change results; two evals are comparable only if this matches."""
    llm = settings.llm
    return {
        "provider": provider_name,
        "model": llm.model,
        "tool_choice": llm.tool_choice_for(provider_name),
        "effort": llm.effort,
        "temperature": llm.temperature,
        "max_output_tokens": llm.max_output_tokens,
        "prompt_caching": llm.prompt_caching if provider_name == "anthropic" else None,
        "groq_tpm_limit": llm.groq_tpm_limit if provider_name == "groq" else None,
        "groq_free_tier": llm.groq_free_tier if provider_name == "groq" else None,
        "max_retries": llm.max_retries,
        "max_retry_wait_s": llm.max_retry_wait_s,
        "request_timeout_s": llm.request_timeout_s,
        "price": price.model_dump() if (price := settings.pricing.get(llm.model)) else None,
        "budget": settings.budget.model_dump(),
        "agent": settings.agent.model_dump(),
        "tools": settings.tools.model_dump(),
        "sandbox": settings.sandbox.model_dump(),
    }


def git_state(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    """(HEAD sha, working tree dirty?) or (None, None) outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)runs"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha, bool(status.strip())


def build_manifest(
    eval_id: str,
    settings: Settings,
    provider_name: str,
    tasks: list[Task],
    runs: int,
    environment: dict[str, Any] | None = None,
) -> EvalManifest:
    sha, dirty = git_state()
    return EvalManifest(
        eval_id=eval_id,
        created_at=datetime.now(UTC),
        runs=runs,
        package_version=__version__,
        git_sha=sha,
        git_dirty=dirty,
        config=config_snapshot(settings, provider_name),
        tasks={
            t.id: TaskMeta(
                file_hash=t.file_hash, repo=t.repo, bug_type=t.bug_type, difficulty=t.difficulty
            )
            for t in tasks
        },
        environment=environment or {},
    )


# The loop checks its wall-clock budget before every model call and tool call, and each
# request is capped at the time left, so an attempt can only overrun by about one request
# timeout plus grading. Anything well beyond that means the host was suspended (e.g. the
# machine went to sleep mid-request), which says nothing about the agent.
OVERRUN_GRACE_S = 600.0


def wall_limit_s(wall_clock_timeout_s: float) -> float:
    """Longest wall time an attempt can legitimately take under this budget."""
    return wall_clock_timeout_s + OVERRUN_GRACE_S


def is_infra_failure(result: AgentResult, wall_limit: float | None = None) -> bool:
    """The attempt says nothing about the agent: the provider, sandbox, or host failed.

    LLM errors other than 400 (quota/rate limits, outages, auth, connectivity) are
    infrastructure; a 400 (e.g. unparseable model output after retries) is a model failure.
    An attempt that ran far past its wall-clock budget (``wall_limit``) was suspended.
    """
    if result.stop_reason == StopReason.SANDBOX_ERROR or result.outcome == Outcome.NO_FINAL_TESTS:
        return True
    if wall_limit is not None and result.wall_s > wall_limit:
        return True
    return result.stop_reason == StopReason.LLM_ERROR and result.error_status != 400


def attempt_dir(eval_dir: Path, task_id: str, run: int) -> Path:
    return Path(eval_dir) / task_id / f"run-{run}"


@dataclass
class RunSummary:
    completed: int = 0
    skipped: int = 0
    stopped_reason: str | None = None
    results: list[AgentResult] = field(default_factory=list)


class EvalRunner:
    """Runs every (task, run) attempt that does not already have a result."""

    def __init__(
        self,
        *,
        settings: Settings,
        tasks: list[Task],
        eval_dir: Path,
        runs: int,
        provider: LLMProvider,
        sandbox_for: Callable[[Task], DockerSandbox],
        parallel: int = 1,
        echo: Callable[[str], None] = print,
        solve: Callable[..., AgentResult] = solve_task,
    ):
        self.settings = settings
        self.tasks = tasks
        self.eval_dir = Path(eval_dir)
        self.runs = runs
        self.provider = provider
        self.sandbox_for = sandbox_for
        self.parallel = max(1, parallel)
        self.echo = echo
        self.solve = solve
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.wall_limit = wall_limit_s(settings.budget.wall_clock_timeout_s)

    def reclassify_existing(self) -> list[str]:
        """Move saved results that are infra failures under current rules aside, so they re-run.

        Returns ``task run k`` labels of the attempts that were reclassified.
        """
        moved = []
        for task in self.tasks:
            for k in range(1, self.runs + 1):
                path = attempt_dir(self.eval_dir, task.id, k) / RESULT
                if not path.is_file():
                    continue
                result = AgentResult.model_validate_json(path.read_text(encoding="utf-8"))
                if is_infra_failure(result, self.wall_limit):
                    path.replace(path.with_name(INFRA_RESULT))
                    moved.append(f"{task.id} run {k}")
        return moved

    def pending(self) -> list[tuple[Task, int]]:
        """Attempts without a result, in round order (every task's run 1, then run 2, ...)."""
        return [
            (task, k)
            for k in range(1, self.runs + 1)
            for task in self.tasks
            if not (attempt_dir(self.eval_dir, task.id, k) / RESULT).is_file()
        ]

    def run(self) -> RunSummary:
        for label in self.reclassify_existing():
            self.echo(f"[{label}] earlier result reclassified as infrastructure; re-running")
        todo = self.pending()
        summary = RunSummary(skipped=len(self.tasks) * self.runs - len(todo))
        if self.parallel == 1:
            for task, k in todo:
                if self._stop.is_set():
                    break
                self._attempt(task, k, summary)
        else:
            with ThreadPoolExecutor(max_workers=self.parallel) as pool:
                for future in [pool.submit(self._attempt, t, k, summary) for t, k in todo]:
                    future.result()
        return summary

    def _attempt(self, task: Task, k: int, summary: RunSummary) -> None:
        if self._stop.is_set():
            return
        directory = attempt_dir(self.eval_dir, task.id, k)
        directory.mkdir(parents=True, exist_ok=True)
        _set_aside_partial(directory)
        started = time.perf_counter()
        try:
            result = self._solve(task, k, directory)
        except (SandboxError, WorkspaceError) as exc:
            self._halt(summary, f"{task.id} run {k}: sandbox failure: {exc}")
            return
        infra = is_infra_failure(result, self.wall_limit)
        if infra:
            (directory / RESULT).replace(directory / INFRA_RESULT)
        with self._lock:
            if infra:
                self._halt(summary, f"{task.id} run {k}: {result.stop_reason}: {result.error}")
            else:
                summary.completed += 1
                summary.results.append(result)
            mark = "infra" if infra else ("PASS" if result.resolved else "fail")
            self.echo(
                f"[{task.id} run {k}] {mark:<5} {result.outcome} (stop: {result.stop_reason}) "
                f"iters={result.iterations} tests={result.test_runs} "
                f"tokens={result.tokens_in + result.cache_read_tokens + result.tokens_out:,} "
                f"{time.perf_counter() - started:.0f}s"
            )

    def _solve(self, task: Task, k: int, directory: Path) -> AgentResult:
        sandbox = self.sandbox_for(task)
        workspace = Workspace.from_directory(task.repo_dir(), patch=task.seed_patch)
        try:
            with Tracer(
                directory.parent,
                directory.name,
                TRACE.removesuffix(".jsonl"),
                secret_values=self.settings.secret_values(),
            ) as tracer:
                return self.solve(
                    task,
                    workspace,
                    provider=self.provider,
                    sandbox=sandbox,
                    settings=self.settings,
                    tracer=tracer,
                    run_id=f"{self.eval_dir.name}/{task.id}/run-{k}",
                    result_path=directory / RESULT,
                )
        finally:
            workspace.cleanup()

    def _halt(self, summary: RunSummary, reason: str) -> None:
        self._stop.set()
        summary.stopped_reason = summary.stopped_reason or reason


def _set_aside_partial(directory: Path) -> None:
    """Keep the trace of an interrupted attempt for inspection, but start fresh."""
    trace = directory / TRACE
    if trace.exists() and not (directory / RESULT).exists():
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        trace.replace(directory / f"trace.partial-{stamp}.jsonl")
    infra = directory / INFRA_RESULT
    if infra.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        infra.replace(directory / f"result.infra-{stamp}.json")


def new_eval_id() -> str:
    return f"eval-{new_run_id()}"


def load_manifest(eval_dir: Path) -> EvalManifest:
    return EvalManifest.model_validate_json((Path(eval_dir) / MANIFEST).read_text(encoding="utf-8"))


def write_manifest(eval_dir: Path, manifest: EvalManifest) -> None:
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    (Path(eval_dir) / MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


class AttemptRecord(BaseModel):
    """One attempt as read back from disk for metrics."""

    task_id: str
    run: int
    result: AgentResult
    infra: bool = Field(default=False)


def load_attempts(eval_dir: Path, wall_limit: float | None = None) -> list[AttemptRecord]:
    """Every attempt directory's result (completed and infra), sorted by task and run.

    With ``wall_limit``, saved results that overran it are treated as infra as well.
    """
    records = []
    for directory in sorted(Path(eval_dir).glob("*/run-*")):
        run = int(directory.name.removeprefix("run-"))
        for name, saved_as_infra in ((RESULT, False), (INFRA_RESULT, True)):
            path = directory / name
            if path.is_file():
                result = AgentResult.model_validate_json(path.read_text(encoding="utf-8"))
                infra = saved_as_infra or is_infra_failure(result, wall_limit)
                records.append(
                    AttemptRecord(
                        task_id=directory.parent.name, run=run, result=result, infra=infra
                    )
                )
    return records


def manifest_wall_limit(manifest: EvalManifest) -> float:
    """The overrun threshold implied by the budget recorded in an eval's manifest."""
    return wall_limit_s(float(manifest.config["budget"]["wall_clock_timeout_s"]))
