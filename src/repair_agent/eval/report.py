"""Markdown + JSON reports from computed metrics."""

from __future__ import annotations

import json
from pathlib import Path

from repair_agent.eval.metrics import FAILURE_MODES, GroupStats, Metrics
from repair_agent.eval.runner import EvalManifest

RESULTS_HEADER = (
    "| Date | Eval id | Provider / model | Tasks x runs | Valid attempts | Resolve rate (95% CI) "
    "| pass@1 | pass@3 | Avg iterations | Tokens in / out per attempt | Cost (list-price equiv.) "
    "| Wall p50 / p95 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.4f}"


def _group_table(title: str, groups: dict[str, GroupStats]) -> list[str]:
    lines = [
        f"### By {title}",
        "",
        f"| {title} | Attempts | Resolved | Resolve rate | pass@1 |",
        "|---|---|---|---|---|",
    ]
    for key, g in groups.items():
        lines.append(
            f"| {key} | {g.attempts} | {g.resolved} | {pct(g.resolve_rate)} | {pct(g.pass_at_1)} |"
        )
    return [*lines, ""]


def render_markdown(manifest: EvalManifest, m: Metrics) -> str:
    cfg = manifest.config
    complete = m.valid_attempts == m.planned_attempts
    lo, hi = m.resolve_rate_ci95
    lines = [
        f"# Eval report: {manifest.eval_id}",
        "",
        f"- **Status:** {'complete' if complete else 'PARTIAL'}: {m.valid_attempts} of "
        f"{m.planned_attempts} attempts have results"
        + (f" ({m.infra_attempts} lost to infrastructure, excluded)" if m.infra_attempts else ""),
        f"- **Provider / model:** {cfg['provider']} / `{cfg['model']}` "
        f"(tool_choice: {cfg['tool_choice']}, effort: {cfg['effort'] or 'default'}, "
        f"temperature: {'default' if cfg['temperature'] is None else cfg['temperature']})",
        f"- **Tasks x runs:** {len(manifest.tasks)} x {manifest.runs}",
        f"- **Budgets:** {cfg['budget']['max_iterations']} iterations, "
        f"{cfg['budget']['max_test_runs']} test runs, {cfg['budget']['max_tokens_per_task']:,} "
        f"tokens, {cfg['budget']['wall_clock_timeout_s']:.0f}s per attempt",
        f"- **Code:** `{(manifest.git_sha or 'unknown')[:12]}`"
        + (" (working tree had uncommitted changes)" if manifest.git_dirty else ""),
        f"- **Started:** {manifest.created_at:%Y-%m-%d %H:%M} UTC",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Resolve rate | **{pct(m.resolve_rate)}** ({m.resolved}/{m.valid_attempts}; "
        f"95% CI {pct(lo)}-{pct(hi)}) |",
        f"| pass@1 (mean over tasks) | {pct(m.pass_at_1)} |",
        f"| pass@3 | {pct(m.pass_at_3)}"
        + (
            f" (over {m.pass_at_3_tasks} tasks with ≥3 runs)"
            if m.pass_at_3 is not None
            else " (needs ≥3 runs per task)"
        )
        + " |",
        f"| Success (finish called and resolved) | {pct(m.success_rate)} |",
        f"| Graded-test pass rate | {pct(m.test_pass_rate)} |",
        "| Resolve rate by run | "
        + (", ".join(f"run {k}: {pct(v)}" for k, v in m.run_resolve_rates.items()) or "n/a")
        + (
            f" (mean {pct(m.run_resolve_rate_mean)} ± {pct(m.run_resolve_rate_std)})"
            if m.run_resolve_rate_std is not None
            else ""
        )
        + " |",
        f"| Flaky tasks (resolved in some runs only) | "
        f"{len(m.flaky_tasks)}{': ' + ', '.join(m.flaky_tasks) if m.flaky_tasks else ''} |",
        "",
        "## Effort and cost",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Iterations (mean / median) | {num(m.iterations_mean)} / {num(m.iterations_median)} |",
        f"| Test runs (mean) | {num(m.test_runs_mean)} |",
        f"| Tokens per attempt (in / out / cache read) | {num(m.tokens_in_mean, 0)} / "
        f"{num(m.tokens_out_mean, 0)} / {num(m.cache_read_mean, 0)} |",
        f"| Cost charged (total / per resolved) | {money(m.cost_total_usd)} / "
        f"{money(m.cost_per_resolved_usd)} |",
        f"| List-price equivalent (total / per resolved) | {money(m.list_price_total_usd)} / "
        f"{money(m.list_price_per_resolved_usd)} |",
        f"| Wall time p50 / p95 | {num(m.wall_p50_s)}s / {num(m.wall_p95_s)}s |",
        f"| LLM time p50 / p95 | {num(m.llm_p50_s)}s / {num(m.llm_p95_s)}s |",
        "",
        "## Failure modes (unresolved attempts)",
        "",
        "| Mode | Attempts |",
        "|---|---|",
        *[f"| {mode} | {m.failure_modes.get(mode, 0)} |" for mode in FAILURE_MODES],
        "",
        f"Stop reasons: {', '.join(f'{k}={v}' for k, v in m.stop_reasons.items()) or 'n/a'}. "
        f"Attempts that edited original tests or added collection-affecting files "
        f"(restored or removed before grading): {m.tamper_attempts}.",
        "",
        "## Breakdowns",
        "",
        *_group_table("bug type", m.by_bug_type),
        *_group_table("difficulty", m.by_difficulty),
        *_group_table("repo", m.by_repo),
        "## Per task",
        "",
        "| Task | Bug type | Difficulty | Runs | pass@1 | pass@3 |",
        "|---|---|---|---|---|---|",
    ]
    for t in m.tasks:
        runs = " ".join("✓" if o else ("✗" if o is False else "·") for o in t.outcomes)
        lines.append(
            f"| {t.task_id} | {t.bug_type} | {t.difficulty} | {runs} "
            f"| {pct(t.pass_at_1)} | {pct(t.pass_at_3)} |"
        )
    lines += ["", "✓ resolved · ✗ not resolved · · no result yet", ""]
    if m.infra:
        lines += [
            "## Attempts lost to infrastructure (excluded, re-run on resume)",
            "",
            *[f"- {line}" for line in m.infra],
            "",
        ]
    return "\n".join(lines)


def results_row(manifest: EvalManifest, m: Metrics) -> str:
    """One RESULTS.md table row with the headline numbers of this eval."""
    cfg = manifest.config
    lo, hi = m.resolve_rate_ci95
    status = "" if m.valid_attempts == m.planned_attempts else " (partial)"
    return (
        f"| {manifest.created_at:%Y-%m-%d} | `{manifest.eval_id}`{status} | {cfg['provider']} / "
        f"`{cfg['model']}` | {len(manifest.tasks)} x {manifest.runs} | {m.valid_attempts} "
        f"| {pct(m.resolve_rate)} ({pct(lo)}-{pct(hi)}) | {pct(m.pass_at_1)} "
        f"| {pct(m.pass_at_3)} | {num(m.iterations_mean)} | {num(m.tokens_in_mean, 0)} / "
        f"{num(m.tokens_out_mean, 0)} | {money(m.cost_total_usd)} "
        f"({money(m.list_price_total_usd)}) | {num(m.wall_p50_s, 0)}s / {num(m.wall_p95_s, 0)}s |"
    )


def write_report(eval_dir: Path, manifest: EvalManifest, m: Metrics) -> tuple[Path, Path]:
    md = Path(eval_dir) / "report.md"
    js = Path(eval_dir) / "report.json"
    md.write_text(render_markdown(manifest, m), encoding="utf-8")
    js.write_text(
        json.dumps(
            {"manifest": manifest.model_dump(mode="json"), "metrics": m.model_dump(mode="json")},
            indent=2,
        ),
        encoding="utf-8",
    )
    return md, js


def append_results_row(results_md: Path, manifest: EvalManifest, m: Metrics) -> None:
    """Add this eval to the 'Benchmark runs' table in RESULTS.md (created if missing)."""
    text = results_md.read_text(encoding="utf-8") if results_md.exists() else "# Results\n"
    section = "## Benchmark runs"
    row = results_row(manifest, m)
    if section not in text:
        text = text.rstrip("\n") + f"\n\n{section}\n\n{RESULTS_HEADER}{row}\n"
    else:
        head, tail = text.split(section, 1)
        table_end = tail.find("\n\n", tail.find("|---"))
        if table_end == -1:
            tail = tail.rstrip("\n") + f"\n{row}\n"
        else:
            tail = tail[:table_end] + f"\n{row}" + tail[table_end:]
        text = head + section + tail
    results_md.write_text(text, encoding="utf-8")
