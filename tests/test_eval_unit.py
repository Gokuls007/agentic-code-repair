"""Eval pieces without Docker or an LLM: leaks, validation cache, metrics, report, runner."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repair_agent.agent.grading import FinalTests, affects_collection, sanitize_for_grading
from repair_agent.agent.state import AgentResult, Outcome, StopReason
from repair_agent.agent.task import Task, load_task, load_tasks
from repair_agent.config import Settings
from repair_agent.eval.leaks import changed_lines, find_leaks, patched_files
from repair_agent.eval.metrics import (
    compute_metrics,
    failure_mode,
    pass_at_k,
    percentile,
    wilson_interval,
)
from repair_agent.eval.report import append_results_row, render_markdown, results_row
from repair_agent.eval.runner import (
    INFRA_RESULT,
    RESULT,
    AttemptRecord,
    EvalManifest,
    EvalRunner,
    TaskMeta,
    attempt_dir,
    build_manifest,
    is_infra_failure,
    load_attempts,
)
from repair_agent.eval.validate import TaskValidation, ValidationCache, cache_key, write_test_lists
from repair_agent.sandbox.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "benchmark" / "tasks"


# --- benchmark shape ------------------------------------------------------------------


def test_benchmark_has_27_tasks_with_the_planned_mix() -> None:
    tasks = load_tasks(TASKS_DIR)
    assert len(tasks) == 27
    assert {t.repo for t in tasks} == {"calc", "textkit", "inventory", "schedule"}
    by_type: dict[str, int] = {}
    for t in tasks:
        by_type[t.bug_type] = by_type.get(t.bug_type, 0) + 1
    assert by_type == {
        "off-by-one": 6,
        "wrong-conditional": 5,
        "missing-edge-case": 6,
        "api-misuse": 7,
        "multi-file": 3,
    }
    assert sorted(t.difficulty for t in tasks).count("hard") == 4


@pytest.mark.parametrize("task", load_tasks(TASKS_DIR), ids=lambda t: t.id)
def test_every_issue_is_symptom_only(task: Task) -> None:
    assert find_leaks(task) == []


def test_multi_file_tasks_touch_more_than_one_file() -> None:
    for task in load_tasks(TASKS_DIR):
        touched = patched_files(task.seed_patch or "")
        assert (len(touched) > 1) == (task.bug_type == "multi-file"), task.id


# --- leak check -----------------------------------------------------------------------


def _task(body: str, patch: str) -> Task:
    return Task.model_validate(
        {
            "id": "t-1",
            "repo": "calc",
            "bug_type": "off-by-one",
            "difficulty": "easy",
            "issue": {"title": "t", "body": body},
            "seed_patch": patch,
            "fail_to_pass": ["a::b"],
        }
    )


PATCH = (
    "--- a/src/pkg/core.py\n+++ b/src/pkg/core.py\n@@ -1 +1 @@\n"
    "-    return total / n\n+    return total / (n - 1)\n"
)


def test_leaks_detect_paths_code_and_hints() -> None:
    assert patched_files(PATCH) == ["src/pkg/core.py"]
    assert changed_lines(PATCH) == ["return total / n", "return total / (n - 1)"]
    assert find_leaks(_task("average is wrong: avg([1,2]) gives 3", PATCH)) == []
    assert "core.py" in find_leaks(_task("look in core.py", PATCH))[0]
    assert "changed line" in find_leaks(_task("it does `return total / (n - 1)`", PATCH))[0]
    assert "hint" in find_leaks(_task("classic off-by-one", PATCH))[0]


def test_import_lines_are_not_leaks() -> None:
    patch = (
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n"
        "-from datetime import datetime\n+from datetime import date\n"
    )
    assert changed_lines(patch) == []


# --- validation helpers -----------------------------------------------------------------


def test_write_test_lists_rewrites_only_the_trailing_block(tmp_path: Path) -> None:
    src = TASKS_DIR / "calc-median-002.yaml"
    copy = tmp_path / "tasks" / src.name
    copy.parent.mkdir()
    shutil.copy(src, copy)
    task = load_task(copy)
    write_test_lists(task, ["tests/a.py::x"], ["tests/a.py::y", "tests/b.py::z"])
    reloaded = load_task(copy)
    assert reloaded.fail_to_pass == ["tests/a.py::x"]
    assert reloaded.pass_to_pass == ["tests/a.py::y", "tests/b.py::z"]
    assert reloaded.issue == task.issue and reloaded.seed_patch == task.seed_patch


def test_validation_cache_is_keyed_by_content(tmp_path: Path) -> None:
    task = load_task(TASKS_DIR / "calc-mean-001.yaml")
    key = cache_key(task, "img:1")
    assert key == cache_key(task, "img:1") and key != cache_key(task, "img:2")
    cache = ValidationCache(tmp_path / "v.json")
    cache.put(TaskValidation(task_id=task.id, ok=True, cache_key=key))
    assert ValidationCache(tmp_path / "v.json").get(task.id, key).ok
    assert ValidationCache(tmp_path / "v.json").get(task.id, "other") is None


# --- grading hardening (no Docker) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("conftest.py", True),
        ("src/calc/conftest.py", True),
        ("pytest.ini", True),
        ("setup.cfg", True),
        ("pyproject.toml", True),
        ("src/sitecustomize.py", True),
        ("usercustomize.py", True),
        ("src/evil.pth", True),
        ("src/calc/helper.py", False),
        ("tests/test_new.py", False),
    ],
)
def test_affects_collection(path: str, expected: bool) -> None:
    assert affects_collection(path) is expected


def test_sanitize_restores_config_and_removes_new_collection_files(tmp_path: Path) -> None:
    task = load_task(TASKS_DIR / "calc-mean-001.yaml")
    with Workspace.from_directory(
        task.repo_dir(), patch=task.seed_patch, parent_dir=tmp_path
    ) as ws:
        (ws.root / "conftest.py").write_text("import calc.stats\n", encoding="utf-8")
        (ws.root / "src" / "sitecustomize.py").write_text("x = 1\n", encoding="utf-8")
        (ws.root / "pytest.ini").write_text("[pytest]\naddopts = -p no:x\n", encoding="utf-8")
        (ws.root / "tests" / "test_new.py").write_text("def test_x(): pass\n", encoding="utf-8")
        (ws.root / "tests" / "test_stats.py").write_text("# gutted\n", encoding="utf-8")
        modified, removed = sanitize_for_grading(task, ws)
        assert modified == ["tests/test_stats.py"]
        assert removed == ["conftest.py", "pytest.ini", "src/sitecustomize.py"]
        assert not (ws.root / "conftest.py").exists()
        assert (ws.root / "tests" / "test_new.py").exists()  # new tests stay (not graded)
        assert b"test_mean_basic" in (ws.root / "tests" / "test_stats.py").read_bytes()


# --- metrics ------------------------------------------------------------------------------


def test_wilson_interval_known_values() -> None:
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("n", "c", "k", "expected"),
    [(3, 0, 1, 0.0), (3, 1, 1, 1 / 3), (3, 1, 3, 1.0), (3, 0, 3, 0.0), (5, 2, 3, 1 - 1 / 10)],
)
def test_pass_at_k(n: int, c: int, k: int, expected: float) -> None:
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_percentile_interpolates() -> None:
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 95) == pytest.approx(10.5)
    assert percentile([], 50) is None


def result(
    task_id: str,
    *,
    resolved: bool,
    stop: StopReason = StopReason.FINISHED,
    p2p_ok: bool = True,
    changed: bool = True,
    status: int | None = None,
    tokens: int = 1000,
    wall: float = 10.0,
    **extra: object,
) -> AgentResult:
    now = datetime(2026, 9, 26, tzinfo=UTC)
    outcome = Outcome.FINISHED_TESTS_PASS if resolved else Outcome.FINISHED_TESTS_FAIL
    return AgentResult(
        run_id="r",
        task_id=task_id,
        provider="groq",
        model="m",
        model_id="m",
        temperature=None,
        started_at=now,
        ended_at=now,
        stop_reason=stop,
        outcome=outcome,
        success=resolved,
        resolved=resolved,
        error_status=status,
        iterations=5,
        test_runs=2,
        tool_calls={},
        tokens_in=tokens,
        tokens_out=100,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=0.0,
        list_price_usd=0.001,
        wall_s=wall,
        llm_s=wall / 2,
        final_tests=FinalTests(),
        diff="",
        source_files_changed=["src/x.py"] if changed else [],
        f2p_passed=1 if resolved else 0,
        f2p_total=1,
        p2p_passed=3 if p2p_ok else 2,
        p2p_total=3,
        **extra,
    )


@pytest.mark.parametrize(
    ("kwargs", "mode"),
    [
        ({"resolved": True}, None),
        ({"resolved": False, "p2p_ok": False, "stop": StopReason.TIMEOUT}, "broke_other_tests"),
        ({"resolved": False, "stop": StopReason.TIMEOUT}, "timeout"),
        ({"resolved": False, "stop": StopReason.MAX_ITERATIONS}, "budget_exceeded"),
        ({"resolved": False, "stop": StopReason.TEST_BUDGET}, "budget_exceeded"),
        ({"resolved": False, "changed": False}, "gave_up"),
        ({"resolved": False, "stop": StopReason.NO_ACTION}, "gave_up"),
        ({"resolved": False, "stop": StopReason.LLM_ERROR, "status": 400}, "llm_error"),
        ({"resolved": False}, "wrong_fix"),
    ],
)
def test_failure_mode_order(kwargs: dict, mode: str | None) -> None:
    assert failure_mode(result("t", **kwargs)) == mode


def manifest(runs: int = 3) -> EvalManifest:
    return EvalManifest(
        eval_id="eval-test",
        created_at=datetime(2026, 9, 26, tzinfo=UTC),
        runs=runs,
        package_version="0.1.0",
        git_sha="abc123",
        git_dirty=False,
        config={
            "provider": "groq",
            "model": "m",
            "tool_choice": "required",
            "effort": None,
            "temperature": None,
            "budget": {
                "max_iterations": 30,
                "max_test_runs": 10,
                "max_tokens_per_task": 150000,
                "wall_clock_timeout_s": 1800,
            },
        },
        tasks={
            "a": TaskMeta(file_hash="1", repo="r1", bug_type="off-by-one", difficulty="easy"),
            "b": TaskMeta(file_hash="2", repo="r1", bug_type="api-misuse", difficulty="hard"),
        },
    )


def records() -> list[AttemptRecord]:
    rows = [
        ("a", 1, result("a", resolved=True, wall=10)),
        ("a", 2, result("a", resolved=True, wall=20)),
        ("a", 3, result("a", resolved=False, wall=30)),
        ("b", 1, result("b", resolved=False, stop=StopReason.MAX_ITERATIONS, wall=40)),
        ("b", 2, result("b", resolved=False, wall=50)),
        ("b", 3, result("b", resolved=False, stop=StopReason.LLM_ERROR, status=429)),
    ]
    return [
        AttemptRecord(task_id=t, run=k, result=r, infra=is_infra_failure(r)) for t, k, r in rows
    ]


def test_compute_metrics_hand_checked() -> None:
    m = compute_metrics(manifest(), records())
    assert (m.planned_attempts, m.valid_attempts, m.infra_attempts) == (6, 5, 1)
    assert m.resolved == 2 and m.resolve_rate == pytest.approx(0.4)
    assert m.pass_at_1 == pytest.approx((2 / 3 + 0 / 2) / 2)
    assert m.pass_at_3 == pytest.approx(1.0) and m.pass_at_3_tasks == 1  # only "a" has 3 runs
    assert m.run_resolve_rates == {1: 0.5, 2: 0.5, 3: 0.0}
    assert m.flaky_tasks == ["a"]
    assert m.failure_modes["budget_exceeded"] == 1 and m.failure_modes["wrong_fix"] == 2
    assert m.wall_p50_s == 30 and m.cost_total_usd == 0.0
    assert m.list_price_per_resolved_usd == pytest.approx(0.005 / 2)
    assert m.by_difficulty["hard"].resolved == 0 and m.by_repo["r1"].attempts == 5
    assert m.tasks[0].outcomes == [True, True, False]
    assert m.tasks[1].outcomes == [False, False, None]  # infra attempt shows as missing
    assert len(m.infra) == 1 and "b run 3" in m.infra[0]


def test_report_and_results_row() -> None:
    mf, m = manifest(), compute_metrics(manifest(), records())
    md = render_markdown(mf, m)
    assert "PARTIAL: 5 of 6 attempts" in md
    assert "| Resolve rate | **40.0%** (2/5; 95% CI" in md
    assert "| a | off-by-one | easy | ✓ ✓ ✗ |" in md
    assert "Attempts lost to infrastructure" in md
    row = results_row(mf, m)
    assert row.startswith("| 2026-09-26 | `eval-test` (partial) | groq / `m` | 2 x 3 | 5 | 40.0%")


def test_append_results_row_creates_and_extends_table(tmp_path: Path) -> None:
    path = tmp_path / "RESULTS.md"
    path.write_text("# Results\n\n## Smoke tests\n\ntext\n", encoding="utf-8")
    append_results_row(path, manifest(), compute_metrics(manifest(), records()))
    append_results_row(path, manifest(), compute_metrics(manifest(), records()))
    text = path.read_text(encoding="utf-8")
    assert text.count("## Benchmark runs") == 1
    assert text.count("`eval-test`") == 2
    assert text.index("## Smoke tests") < text.index("## Benchmark runs")


# --- runner -----------------------------------------------------------------------------------


def test_infra_classification() -> None:
    assert is_infra_failure(result("t", resolved=False, stop=StopReason.LLM_ERROR, status=429))
    assert is_infra_failure(result("t", resolved=False, stop=StopReason.LLM_ERROR, status=None))
    assert not is_infra_failure(result("t", resolved=False, stop=StopReason.LLM_ERROR, status=400))
    assert is_infra_failure(result("t", resolved=False, stop=StopReason.SANDBOX_ERROR))
    assert not is_infra_failure(result("t", resolved=False, stop=StopReason.TIMEOUT))


class FakeSolve:
    """Stands in for solve_task: writes a scripted AgentResult to result_path."""

    def __init__(self, plan: dict[tuple[str, int], AgentResult]):
        self.plan = plan
        self.calls: list[tuple[str, str]] = []

    def __call__(self, task, workspace, *, run_id, result_path, **_: object) -> AgentResult:
        run = int(run_id.rsplit("run-", 1)[1])
        self.calls.append((task.id, run))
        r = self.plan[(task.id, run)]
        result_path.write_text(r.model_dump_json(), encoding="utf-8")
        return r


def make_runner(tmp_path: Path, plan: dict, runs: int = 2) -> tuple[EvalRunner, FakeSolve, list]:
    tasks = [
        load_task(TASKS_DIR / "calc-mean-001.yaml"),
        load_task(TASKS_DIR / "calc-median-002.yaml"),
    ]
    fake = FakeSolve(plan)
    runner = EvalRunner(
        settings=Settings(),
        tasks=tasks,
        eval_dir=tmp_path / "eval-x",
        runs=runs,
        provider=object(),
        sandbox_for=lambda task: object(),
        echo=lambda _: None,
        solve=fake,
    )
    return runner, fake, tasks


def test_runner_round_order_and_resume(tmp_path: Path) -> None:
    ok = result("x", resolved=True)
    plan = {(t, k): ok for t in ("calc-mean-001", "calc-median-002") for k in (1, 2)}
    runner, fake, _ = make_runner(tmp_path, plan)
    # Simulate a crash: run 1 of the second task left a trace but no result.
    crashed = attempt_dir(runner.eval_dir, "calc-median-002", 1)
    crashed.mkdir(parents=True)
    (crashed / "trace.jsonl").write_text("{}\n", encoding="utf-8")
    (attempt_dir(runner.eval_dir, "calc-mean-001", 1)).mkdir(parents=True)
    (attempt_dir(runner.eval_dir, "calc-mean-001", 1) / RESULT).write_text(
        ok.model_dump_json(), encoding="utf-8"
    )

    summary = runner.run()
    assert fake.calls == [("calc-median-002", 1), ("calc-mean-001", 2), ("calc-median-002", 2)]
    assert (summary.completed, summary.skipped, summary.stopped_reason) == (3, 1, None)
    assert list(crashed.glob("trace.partial-*.jsonl"))  # interrupted trace kept aside
    assert runner.pending() == []
    assert len(load_attempts(runner.eval_dir)) == 4


def test_runner_stops_on_quota_and_resumes_the_lost_attempt(tmp_path: Path) -> None:
    ok = result("x", resolved=True)
    quota = result(
        "x",
        resolved=False,
        stop=StopReason.LLM_ERROR,
        status=429,
        error="Rate limit reached: tokens per day",
    )
    plan = {
        ("calc-mean-001", 1): ok,
        ("calc-median-002", 1): quota,
        ("calc-mean-001", 2): ok,
        ("calc-median-002", 2): ok,
    }
    runner, fake, _ = make_runner(tmp_path, plan)
    summary = runner.run()
    assert "calc-median-002 run 1" in summary.stopped_reason
    assert fake.calls == [("calc-mean-001", 1), ("calc-median-002", 1)]  # stopped there
    lost = attempt_dir(runner.eval_dir, "calc-median-002", 1)
    assert (lost / INFRA_RESULT).exists() and not (lost / RESULT).exists()

    plan[("calc-median-002", 1)] = ok  # quota is back
    runner2, fake2, _ = make_runner(tmp_path, plan)
    runner2.solve = fake2
    summary2 = runner2.run()
    assert fake2.calls == [("calc-median-002", 1), ("calc-mean-001", 2), ("calc-median-002", 2)]
    assert summary2.stopped_reason is None
    recs = load_attempts(runner2.eval_dir)
    assert sum(r.infra for r in recs) == 0 and len(recs) == 4


def test_manifest_resume_compatibility() -> None:
    tasks = [load_task(TASKS_DIR / "calc-mean-001.yaml")]
    a = build_manifest("e1", Settings(), "groq", tasks, 3)
    same = build_manifest("e1", Settings(), "groq", tasks, 3)
    assert a.compatible_with(same) == []
    changed = Settings()
    changed.budget.max_iterations = 5
    problems = a.compatible_with(build_manifest("e1", changed, "groq", tasks, 3))
    assert problems and "budget" in problems[0]
    assert a.compatible_with(build_manifest("e1", Settings(), "groq", tasks, 1))
    assert a.compatible_with(build_manifest("e1", Settings(), "groq", [], 3))
