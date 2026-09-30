"""Metrics over eval attempts. Every number here is computed from saved results.

Definitions (also in DECISIONS.md):

- valid attempt: has a result and was not lost to infrastructure;
- resolve rate: resolved / valid attempts, with a 95% Wilson score interval;
- pass@k: mean over tasks of the unbiased estimator 1 - C(n-c, k) / C(n, k), where n is
  the task's valid attempts and c its resolved ones (only tasks with n >= k count);
- failure mode: the first matching category, in this order: broke_other_tests,
  timeout, budget_exceeded, gave_up, llm_error, wrong_fix.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from math import comb

from pydantic import BaseModel, Field

from repair_agent.agent.state import AgentResult, StopReason
from repair_agent.eval.runner import AttemptRecord, EvalManifest

FAILURE_MODES = (
    "broke_other_tests",
    "timeout",
    "budget_exceeded",
    "gave_up",
    "llm_error",
    "wrong_fix",
)
_BUDGET_STOPS = {
    StopReason.MAX_ITERATIONS,
    StopReason.TOKEN_BUDGET,
    StopReason.COST_BUDGET,
    StopReason.TEST_BUDGET,
    StopReason.CONTEXT_LIMIT,
}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one task with n attempts, c of them successful."""
    if n < k:
        raise ValueError("need at least k attempts")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def percentile(values: list[float], p: float) -> float | None:
    """Linearly interpolated percentile (p in [0, 100]); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def failure_mode(r: AgentResult) -> str | None:
    """Why an attempt did not resolve (None if it did)."""
    if r.resolved:
        return None
    if r.p2p_passed < r.p2p_total:
        return "broke_other_tests"
    if r.stop_reason == StopReason.TIMEOUT:
        return "timeout"
    if r.stop_reason in _BUDGET_STOPS:
        return "budget_exceeded"
    if not r.source_files_changed or r.stop_reason in (StopReason.NO_ACTION, StopReason.REFUSAL):
        return "gave_up"
    if r.stop_reason == StopReason.LLM_ERROR:
        return "llm_error"
    return "wrong_fix"


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


class TaskStats(BaseModel):
    task_id: str
    repo: str
    bug_type: str
    difficulty: str
    attempts: int
    resolved: int
    outcomes: list[bool | None] = Field(description="Per run index: resolved, or None if missing")
    pass_at_1: float | None
    pass_at_3: float | None
    std: float | None


class GroupStats(BaseModel):
    attempts: int
    resolved: int
    resolve_rate: float | None
    pass_at_1: float | None


class BackendStats(BaseModel):
    """How much of an eval one pool backend served."""

    requests: int = 0
    request_share: float | None = None
    failovers_from: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    attempts_touched: int = 0
    # Attempts answered entirely by this backend, and how many resolved. Comparing these
    # across backends shows whether hosts serving the same model behave differently.
    sole_attempts: int = 0
    sole_resolved: int = 0


def backend_stats(results: list[AgentResult]) -> dict[str, BackendStats]:
    stats: dict[str, BackendStats] = defaultdict(BackendStats)
    for r in results:
        served = {name: u for name, u in r.backend_usage.items() if u.requests}
        for name, u in r.backend_usage.items():
            s = stats[name]
            s.requests += u.requests
            s.failovers_from += u.failovers_from
            s.tokens_in += u.tokens_in
            s.tokens_out += u.tokens_out
            s.attempts_touched += 1 if u.requests else 0
        if len(served) == 1:
            (only,) = served
            stats[only].sole_attempts += 1
            stats[only].sole_resolved += int(r.resolved)
    total = sum(s.requests for s in stats.values())
    for s in stats.values():
        s.request_share = s.requests / total if total else None
    return dict(sorted(stats.items()))


class Metrics(BaseModel):
    planned_attempts: int
    valid_attempts: int
    model_ids: list[str] = Field(default_factory=list, description="Model ids the API returned")
    agent_grading_disagreements: list[str] = Field(
        default_factory=list, description="'task run k' where the agent's last run disagreed"
    )
    infra_attempts: int
    tasks_with_results: int

    resolved: int
    resolve_rate: float | None
    resolve_rate_ci95: tuple[float, float]
    success_rate: float | None
    pass_at_1: float | None
    pass_at_3: float | None
    pass_at_3_tasks: int
    test_pass_rate: float | None

    run_resolve_rates: dict[int, float]
    run_resolve_rate_mean: float | None
    run_resolve_rate_std: float | None
    flaky_tasks: list[str]

    iterations_mean: float | None
    iterations_median: float | None
    test_runs_mean: float | None
    tokens_in_mean: float | None
    tokens_out_mean: float | None
    cache_read_mean: float | None
    cost_total_usd: float | None
    cost_per_resolved_usd: float | None
    list_price_total_usd: float | None
    list_price_per_resolved_usd: float | None
    wall_p50_s: float | None
    wall_p95_s: float | None
    llm_p50_s: float | None
    llm_p95_s: float | None

    failure_modes: dict[str, int]
    tamper_attempts: int
    stop_reasons: dict[str, int]

    by_bug_type: dict[str, GroupStats]
    by_difficulty: dict[str, GroupStats]
    by_repo: dict[str, GroupStats]
    tasks: list[TaskStats]
    infra: list[str] = Field(description="'task run k: reason' for attempts lost to infra")
    by_backend: dict[str, BackendStats] = Field(default_factory=dict)


def _infra_line(rec: AttemptRecord) -> str:
    reason = f"{rec.result.stop_reason} {rec.result.error or ''}".strip()
    return f"{rec.task_id} run {rec.run}: {reason}"


def _group(stats: list[TaskStats]) -> GroupStats:
    attempts = sum(s.attempts for s in stats)
    resolved = sum(s.resolved for s in stats)
    p1 = [s.pass_at_1 for s in stats if s.pass_at_1 is not None]
    return GroupStats(
        attempts=attempts,
        resolved=resolved,
        resolve_rate=resolved / attempts if attempts else None,
        pass_at_1=_mean(p1),
    )


def compute_metrics(manifest: EvalManifest, records: list[AttemptRecord]) -> Metrics:
    """All report numbers from the manifest and the attempts on disk."""
    valid = [r for r in records if not r.infra]
    results = [r.result for r in valid]
    n = len(results)

    per_task: dict[str, list[AttemptRecord]] = defaultdict(list)
    for rec in valid:
        per_task[rec.task_id].append(rec)

    task_stats = []
    for task_id, meta in sorted(manifest.tasks.items()):
        recs = per_task.get(task_id, [])
        c, m = sum(r.result.resolved for r in recs), len(recs)
        by_run = {r.run: r.result.resolved for r in recs}
        task_stats.append(
            TaskStats(
                task_id=task_id,
                repo=meta.repo,
                bug_type=meta.bug_type,
                difficulty=meta.difficulty,
                attempts=m,
                resolved=c,
                outcomes=[by_run.get(k) for k in range(1, manifest.runs + 1)],
                pass_at_1=c / m if m else None,
                pass_at_3=pass_at_k(m, c, 3) if m >= 3 else None,
                std=statistics.pstdev([float(r.result.resolved) for r in recs]) if m else None,
            )
        )

    resolved = sum(r.resolved for r in results)
    p1 = [s.pass_at_1 for s in task_stats if s.pass_at_1 is not None]
    p3 = [s.pass_at_3 for s in task_stats if s.pass_at_3 is not None]
    graded = [
        (r.f2p_passed + r.p2p_passed) / (r.f2p_total + r.p2p_total)
        for r in results
        if r.f2p_total + r.p2p_total
    ]

    run_rates: dict[int, float] = {}
    for k in range(1, manifest.runs + 1):
        run_results = [rec.result for rec in valid if rec.run == k]
        if run_results:
            run_rates[k] = sum(r.resolved for r in run_results) / len(run_results)

    costs = [r.cost_usd for r in results]
    list_prices = [r.list_price_usd for r in results]
    cost_total = sum(costs) if results and all(c is not None for c in costs) else None
    list_total = sum(list_prices) if results and all(c is not None for c in list_prices) else None

    modes = {mode: 0 for mode in FAILURE_MODES}
    for r in results:
        mode = failure_mode(r)
        if mode:
            modes[mode] += 1
    stops: dict[str, int] = defaultdict(int)
    for r in results:
        stops[str(r.stop_reason)] += 1

    def grouped(attr: str) -> dict[str, GroupStats]:
        buckets: dict[str, list[TaskStats]] = defaultdict(list)
        for s in task_stats:
            buckets[getattr(s, attr)].append(s)
        return {key: _group(value) for key, value in sorted(buckets.items())}

    walls = [r.wall_s for r in results]
    llms = [r.llm_s for r in results]
    return Metrics(
        planned_attempts=len(manifest.tasks) * manifest.runs,
        valid_attempts=n,
        model_ids=sorted({r.model_id for r in results if r.model_id}),
        agent_grading_disagreements=[
            f"{rec.task_id} run {rec.run}"
            for rec in valid
            if rec.result.agent_disagrees_with_grading
        ],
        infra_attempts=len(records) - n,
        tasks_with_results=sum(1 for s in task_stats if s.attempts),
        resolved=resolved,
        resolve_rate=resolved / n if n else None,
        resolve_rate_ci95=wilson_interval(resolved, n),
        success_rate=sum(r.success for r in results) / n if n else None,
        pass_at_1=_mean(p1),
        pass_at_3=_mean(p3),
        pass_at_3_tasks=len(p3),
        test_pass_rate=_mean(graded),
        run_resolve_rates=run_rates,
        run_resolve_rate_mean=_mean(list(run_rates.values())),
        run_resolve_rate_std=(
            statistics.pstdev(run_rates.values()) if len(run_rates) > 1 else None
        ),
        flaky_tasks=[s.task_id for s in task_stats if 0 < s.resolved < s.attempts],
        iterations_mean=_mean([r.iterations for r in results]),
        iterations_median=_median([r.iterations for r in results]),
        test_runs_mean=_mean([r.test_runs for r in results]),
        tokens_in_mean=_mean([r.tokens_in for r in results]),
        tokens_out_mean=_mean([r.tokens_out for r in results]),
        cache_read_mean=_mean([r.cache_read_tokens for r in results]),
        cost_total_usd=cost_total,
        cost_per_resolved_usd=cost_total / resolved
        if cost_total is not None and resolved
        else None,
        list_price_total_usd=list_total,
        list_price_per_resolved_usd=(
            list_total / resolved if list_total is not None and resolved else None
        ),
        wall_p50_s=percentile(walls, 50),
        wall_p95_s=percentile(walls, 95),
        llm_p50_s=percentile(llms, 50),
        llm_p95_s=percentile(llms, 95),
        failure_modes=modes,
        tamper_attempts=sum(
            1
            for r in results
            if r.modified_test_files or r.restored_config_files or r.removed_files
        ),
        stop_reasons=dict(sorted(stops.items())),
        by_bug_type=grouped("bug_type"),
        by_difficulty=grouped("difficulty"),
        by_repo=grouped("repo"),
        tasks=task_stats,
        infra=[_infra_line(rec) for rec in records if rec.infra],
        by_backend=backend_stats(results) if manifest.config.get("provider") == "pool" else {},
    )
