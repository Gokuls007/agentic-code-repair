from __future__ import annotations

from pathlib import Path

import pytest

from repair_agent.agent.context import ELIDED_PREFIX, elide_old_tool_results, stub_for
from repair_agent.agent.retry import backoff_delay, call_with_retry
from repair_agent.agent.task import Task, is_test_file, load_task, normalize_patch
from repair_agent.llm.base import LLMError, Message, TextBlock, ToolCall, ToolResult
from repair_agent.sandbox.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "benchmark" / "tasks" / "calc-mean-001.yaml"

# --- retry ------------------------------------------------------------------------


@pytest.mark.parametrize(("attempt", "expected"), [(0, 2.0), (1, 4.0), (2, 8.0), (5, 30.0)])
def test_backoff_is_exponential_and_capped(attempt: int, expected: float) -> None:
    assert backoff_delay(attempt, base_s=2, cap_s=30, rng=lambda: 1.0) == expected


def test_backoff_full_jitter_and_retry_after_floor() -> None:
    assert backoff_delay(3, base_s=2, cap_s=30, rng=lambda: 0.25) == 4.0
    assert backoff_delay(0, base_s=2, cap_s=30, retry_after_s=12, rng=lambda: 1.0) == 12


def _flaky(errors: list[Exception]):
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return "ok"

    return fn, calls


def test_call_with_retry_succeeds_after_transient_errors() -> None:
    fn, calls = _flaky([LLMError("x", retryable=True), LLMError("y", retryable=True)])
    waits: list[float] = []
    assert (
        call_with_retry(
            fn,
            max_retries=3,
            base_s=1,
            cap_s=10,
            time_left=lambda: 999,
            sleep=waits.append,
            rng=lambda: 1.0,
        )
        == "ok"
    )
    assert calls["n"] == 3 and waits == [1.0, 2.0]


def test_call_with_retry_does_not_retry_non_retryable() -> None:
    fn, calls = _flaky([LLMError("bad", retryable=False, status_code=400)])
    with pytest.raises(LLMError, match="bad"):
        call_with_retry(
            fn, max_retries=3, base_s=1, cap_s=10, time_left=lambda: 999, sleep=lambda _: None
        )
    assert calls["n"] == 1


def test_call_with_retry_never_sleeps_past_the_deadline() -> None:
    fn, calls = _flaky([LLMError("slow down", retryable=True, retry_after_s=60)])
    waits: list[float] = []
    with pytest.raises(LLMError):
        call_with_retry(
            fn, max_retries=3, base_s=1, cap_s=10, time_left=lambda: 30, sleep=waits.append
        )
    assert waits == [] and calls["n"] == 1


# --- context elision -------------------------------------------------------------


def _conversation(n_rounds: int) -> list[Message]:
    messages = [Message(role="user", content=[TextBlock(text="<issue>bug</issue>")])]
    for i in range(n_rounds):
        call = ToolCall(id=f"c{i}", name="read_file", arguments={"path": f"f{i}.py"})
        messages.append(Message(role="assistant", content=[call]))
        messages.append(
            Message(
                role="user",
                content=[
                    ToolResult(tool_call_id=f"c{i}", content="x" * 500),
                    TextBlock(text="[budget]"),
                ],
            )
        )
    return messages


def test_elide_keeps_recent_turns_and_is_idempotent() -> None:
    messages = _conversation(5)
    assert elide_old_tool_results(messages, keep_recent_turns=2) == 3
    contents = [
        m.content[0].content
        for m in messages
        if m.role == "user" and isinstance(m.content[0], ToolResult)
    ]
    assert all(c.startswith(ELIDED_PREFIX) for c in contents[:3])
    assert contents[3:] == ["x" * 500, "x" * 500]
    assert elide_old_tool_results(messages, keep_recent_turns=2) == 0
    assert messages[0].content[0].text == "<issue>bug</issue>"
    assert messages[2].content[1].text == "[budget]"


def test_stub_formats() -> None:
    edit = ToolCall(id="1", name="edit_file", arguments={"path": "a.py", "old_str": "x" * 999})
    edited = ToolResult(tool_call_id="1", content="Edited a.py (1 replacement). Lines 3-9 now:\n")
    assert (
        stub_for(edit, edited) == f"{ELIDED_PREFIX} edit_file a.py lines 3-9; call again if needed]"
    )
    created = ToolResult(tool_call_id="1", content="Created a.py (12 lines).")
    assert stub_for(edit, created) == (
        f"{ELIDED_PREFIX} edit_file a.py (created, 12 lines); call again if needed]"
    )
    failed = ToolResult(tool_call_id="1", content="Error: old_str not found in a.py.")
    assert stub_for(edit, failed) == f"{ELIDED_PREFIX} edit_file a.py; call again if needed]"
    tests = ToolCall(id="2", name="run_tests", arguments={})
    out = ToolResult(tool_call_id="2", content="Result: FAILED (1 failed)\nlots of output")
    assert stub_for(tests, out) == f"{ELIDED_PREFIX} run_tests. Result: FAILED (1 failed)]"
    search = ToolCall(id="3", name="search_code", arguments={"pattern": "p" * 100})
    assert "..." in stub_for(search, ToolResult(tool_call_id="3", content="m"))


# --- task ---------------------------------------------------------------------------


def test_load_benchmark_task() -> None:
    task = load_task(TASK_PATH)
    assert task.id == "calc-mean-001"
    assert task.repo_dir() == ROOT / "benchmark" / "repos" / "calc"
    assert task.graded_test_files == [
        "tests/test_ops.py",
        "tests/test_rounding.py",
        "tests/test_stats.py",
    ]
    assert (task.bug_type, task.difficulty) == ("off-by-one", "easy")
    assert "tests/test_stats.py::test_mean_basic" in task.fail_to_pass


def test_issue_describes_symptom_not_cause() -> None:
    body = load_task(TASK_PATH).issue.body
    assert "mean([1, 2, 3, 4])" in body and "2.5" in body
    for leak in ("len(xs) - 1", "stats.py", "off-by-one", "denominator"):
        assert leak not in body


def test_seed_patch_plants_the_bug(tmp_path: Path) -> None:
    task = load_task(TASK_PATH)
    with Workspace.from_directory(
        task.repo_dir(), patch=task.seed_patch, parent_dir=tmp_path
    ) as ws:
        stats = (ws.root / "src/calc/stats.py").read_text(encoding="utf-8")
        assert "sum(xs) / (len(xs) - 1)" in stats
        assert ws.diff() == ""
        assert "tests/test_stats.py" in ws.baseline_files()


def test_normalize_patch_restores_blank_context_lines() -> None:
    raw = "--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n a\n\n-b\n+c\n\n\n"
    assert normalize_patch(raw) == "--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n a\n \n-b\n+c\n"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_ops.py", True),
        ("tests/helpers.py", True),
        ("conftest.py", True),
        ("src/pkg/conftest.py", True),
        ("pkg/foo_test.py", True),
        ("src/calc/stats.py", False),
        ("tests/data.json", False),
    ],
)
def test_is_test_file(path: str, expected: bool) -> None:
    assert is_test_file(path) is expected


def test_task_id_must_be_safe() -> None:
    with pytest.raises(ValueError, match="filesystem-safe"):
        Task.model_validate(
            {
                "id": "../x",
                "repo": "r",
                "issue": {"title": "t", "body": "b"},
                "fail_to_pass": ["t::x"],
            }
        )


def test_workspace_restore_and_changed_files(tmp_path: Path) -> None:
    task = load_task(TASK_PATH)
    with Workspace.from_directory(
        task.repo_dir(), patch=task.seed_patch, parent_dir=tmp_path
    ) as ws:
        test_file = ws.root / "tests/test_ops.py"
        original = test_file.read_bytes()
        test_file.write_bytes(b"# hacked\n")
        (ws.root / "tests/test_stats.py").unlink()
        changed = ws.changed_files(["tests/test_ops.py", "tests/test_stats.py", "src/calc/ops.py"])
        assert changed == ["tests/test_ops.py", "tests/test_stats.py"]
        ws.restore(changed)
        assert test_file.read_bytes() == original
        assert (ws.root / "tests/test_stats.py").exists()
        assert ws.changed_files(["tests/test_ops.py"]) == []
