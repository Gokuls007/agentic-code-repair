"""Agent loop + solve_task with a scripted LLM and a Docker-free sandbox (no API cost)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from fakes import FakeClock, FakeSandbox, RecordingSleep, ScriptedProvider, reply, tc
from repair_agent.agent import AgentResult, Outcome, StopReason, load_task, solve_task
from repair_agent.agent.context import ELIDED_PREFIX
from repair_agent.agent.prompts import (
    NUDGE_CUT_OFF,
    NUDGE_NO_TOOL,
    NUDGE_TOOL_REQUIRED,
    SYSTEM_PROMPT,
)
from repair_agent.config import AgentSettings, BudgetSettings, LLMSettings, Settings
from repair_agent.llm.base import (
    INVALID_JSON_KEY,
    NO_TOOL_CALL,
    LLMError,
    TextBlock,
    ToolCall,
    ToolResult,
)
from repair_agent.llm.base import StopReason as LLMStop
from repair_agent.sandbox.workspace import Workspace
from repair_agent.tracing import EventKind, Tracer, read_trace

ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "benchmark" / "tasks" / "calc-mean-001.yaml"
BUG = "return sum(xs) / (len(xs) - 1)"
FIX = "return sum(xs) / len(xs)"


@pytest.fixture
def task():
    return load_task(TASK_PATH)


@pytest.fixture
def ws(task, tmp_path: Path) -> Iterator[Workspace]:
    parent = tmp_path / "ws"
    parent.mkdir()
    workspace = Workspace.from_directory(task.repo_dir(), patch=task.seed_patch, parent_dir=parent)
    yield workspace
    workspace.cleanup()


def make_settings(**budget: object) -> Settings:
    return Settings(
        llm=LLMSettings(model="claude-sonnet-5", retry_base_delay_s=1, retry_max_delay_s=8),
        budget=BudgetSettings(**budget),
        agent=AgentSettings(),
    )


def run(task, ws, script, *, settings=None, sandbox=None, clock=None, sleep=None, tmp=None):
    provider = ScriptedProvider(script)
    settings = settings or make_settings()
    sandbox = sandbox or FakeSandbox()
    runs = (tmp or ws.root.parent) / "runs"
    with Tracer(runs, "run1", task.id) as tracer:
        result = solve_task(
            task,
            ws,
            provider=provider,
            sandbox=sandbox,
            settings=settings,
            tracer=tracer,
            run_id="run1",
            clock=clock or FakeClock(),
            sleep=sleep or RecordingSleep(),
            rng=lambda: 1.0,
        )
    return result, provider, tracer.path


FIX_SCRIPT = [
    reply(tc("search_code", pattern="def mean"), text="Looking for mean."),
    reply(tc("read_file", path="src/calc/stats.py")),
    reply(tc("edit_file", path="src/calc/stats.py", old_str=BUG, new_str=FIX)),
    reply(tc("run_tests", test_selector="tests/test_stats.py")),
    reply(tc("finish", summary="mean divided by len-1; now divides by len.")),
]


# --- happy path ------------------------------------------------------------------


def test_fixes_bug_and_records_everything(task, ws) -> None:
    result, _, trace_path = run(task, ws, list(FIX_SCRIPT))

    assert result.stop_reason == StopReason.FINISHED
    assert result.outcome == Outcome.FINISHED_TESTS_PASS
    assert result.success and result.resolved
    assert result.iterations == 5
    assert result.test_runs == 1
    assert result.tool_calls == {
        "search_code": 1,
        "read_file": 1,
        "edit_file": 1,
        "run_tests": 1,
        "finish": 1,
    }
    assert result.finish_summary == "mean divided by len-1; now divides by len."
    assert result.model == "claude-sonnet-5"
    assert result.model_id == "claude-sonnet-5-scripted"
    assert result.temperature is None
    assert (result.tokens_in, result.tokens_out) == (5_000, 500)
    assert result.cost_usd == pytest.approx(5_000 * 2e-6 + 500 * 10e-6)
    assert result.modified_test_files == []
    assert f"-    {BUG}" in result.diff and f"+    {FIX}" in result.diff
    assert result.final_tests is not None and result.final_tests.failed == 0

    saved = AgentResult.model_validate_json(
        trace_path.with_name(f"{task.id}.result.json").read_text(encoding="utf-8")
    )
    assert saved == result

    kinds = [e.kind for e in read_trace(trace_path)]
    assert kinds[0] == EventKind.TASK_START and kinds[-1] == EventKind.TASK_END
    assert EventKind.GRADE in kinds and kinds.count(EventKind.LLM_CALL) == 5


def test_initial_prompt_and_feedback_shape(task, ws) -> None:
    _, provider, _ = run(task, ws, list(FIX_SCRIPT))
    first = provider.requests[0]
    assert first["system"] == SYSTEM_PROMPT
    assert first["tools"] == [
        "list_files",
        "read_file",
        "search_code",
        "edit_file",
        "run_tests",
        "finish",
    ]
    prompt = first["messages"][0].content[0].text
    assert "<issue>\nmean() returns the wrong average" in prompt
    assert "src/\n  calc/\n    __init__.py" in prompt
    assert "test_mean_basic" not in prompt  # grading targets are never revealed

    second = provider.requests[1]["messages"]
    results_msg = second[-1]
    assert results_msg.role == "user"
    assert isinstance(results_msg.content[0], ToolResult)
    assert results_msg.content[0].content == "src/calc/stats.py:4:def mean(xs):"
    budget = results_msg.content[-1]
    assert isinstance(budget, TextBlock)
    assert budget.text.startswith("[budget] iteration 1/30 · test runs 0/10 · tokens 1,100/")


# --- grading integrity -------------------------------------------------------------


def test_finish_with_failing_tests_is_recorded(task, ws) -> None:
    result, _, _ = run(task, ws, [reply(tc("finish", summary="looks fine to me"))])
    assert result.stop_reason == StopReason.FINISHED
    assert result.outcome == Outcome.FINISHED_TESTS_FAIL
    assert not result.success and not result.resolved
    assert result.final_tests.outcomes["tests/test_stats.py::test_mean_basic"] == "failed"


def test_editing_the_test_instead_of_the_code_does_not_resolve(task, ws) -> None:
    cheat = [
        reply(
            tc(
                "edit_file",
                path="tests/test_stats.py",
                old_str="assert mean([1, 2, 3, 4]) == 2.5",
                new_str="assert mean([1, 2, 3, 4]) == 3.3333333333333335",
            )
        ),
        reply(tc("run_tests")),
        reply(tc("finish", summary="updated the expected value")),
    ]
    sandbox = FakeSandbox()
    result, provider, _ = run(task, ws, cheat, sandbox=sandbox)

    # The agent's own test run "passed" because the test was edited...
    tool_msg = provider.requests[2]["messages"][-1]
    assert tool_msg.content[0].content.startswith("Result: PASSED")
    # ...but grading restores the original test first.
    assert result.modified_test_files == ["tests/test_stats.py"]
    assert result.resolved is False
    assert result.outcome == Outcome.FINISHED_TESTS_FAIL
    assert "3.3333333333333335" in result.diff  # the attempt stays visible in the diff
    assert b"== 2.5" in (ws.root / "tests/test_stats.py").read_bytes()


def test_new_test_files_added_by_agent_are_kept(task, ws) -> None:
    script = [
        reply(
            tc(
                "edit_file",
                path="tests/test_extra.py",
                old_str="",
                new_str="def test_extra():\n    assert True\n",
            )
        ),
        *FIX_SCRIPT[2:],
    ]
    result, _, _ = run(task, ws, script)
    assert result.resolved
    assert result.modified_test_files == []
    assert (ws.root / "tests/test_extra.py").exists()


# --- stop conditions ---------------------------------------------------------------


def test_max_iterations(task, ws) -> None:
    script = [reply(tc("list_files")) for _ in range(5)]
    result, provider, _ = run(task, ws, script, settings=make_settings(max_iterations=3))
    assert result.stop_reason == StopReason.MAX_ITERATIONS
    assert result.iterations == 3 and len(provider.requests) == 3
    assert result.outcome == Outcome.STOPPED_TESTS_FAIL  # final grading still runs


def test_token_budget(task, ws) -> None:
    script = [reply(tc("list_files"), input_tokens=9_000, output_tokens=2_000)] * 3
    result, provider, _ = run(task, ws, script, settings=make_settings(max_tokens_per_task=10_000))
    assert result.stop_reason == StopReason.TOKEN_BUDGET
    assert len(provider.requests) == 1


def test_test_run_budget_gives_one_finish_only_turn_then_stops(task, ws) -> None:
    # 2 allowed runs, a 3rd (refused, grace turn granted), then a non-finish call.
    script = [reply(tc("run_tests")) for _ in range(3)] + [reply(tc("list_files"))]
    sandbox = FakeSandbox()
    result, provider, _ = run(
        task, ws, script, settings=make_settings(max_test_runs=2), sandbox=sandbox
    )
    assert result.stop_reason == StopReason.TEST_BUDGET
    assert result.test_runs == 2
    assert len(sandbox.calls) == 3  # 2 agent runs + 1 grading run; the 3rd was refused
    refused = provider.requests[3]["messages"][-1].content[0]
    assert refused.is_error and "one final turn" in refused.content
    assert len(provider.requests) == 4  # the grace turn happened, then the loop stopped


def test_finish_on_the_grace_turn_is_accepted(task, ws) -> None:
    script = [
        reply(tc("edit_file", path="src/calc/stats.py", old_str=BUG, new_str=FIX)),
        *[reply(tc("run_tests")) for _ in range(3)],
        reply(tc("finish", summary="fixed; out of test runs")),
    ]
    result, _, _ = run(task, ws, script, settings=make_settings(max_test_runs=2))
    assert result.stop_reason == StopReason.FINISHED
    assert result.outcome == Outcome.FINISHED_TESTS_PASS and result.resolved


def test_text_reply_on_the_grace_turn_stops_with_test_budget(task, ws) -> None:
    script = [*[reply(tc("run_tests")) for _ in range(3)], reply(text="I think it's fixed.")]
    result, _, _ = run(task, ws, script, settings=make_settings(max_test_runs=2))
    assert result.stop_reason == StopReason.TEST_BUDGET


def test_wall_clock_timeout(task, ws) -> None:
    clock = FakeClock(step=100.0)  # every clock read advances 100s
    script = [reply(tc("list_files")) for _ in range(10)]
    result, _, _ = run(
        task, ws, script, settings=make_settings(wall_clock_timeout_s=450), clock=clock
    )
    assert result.stop_reason == StopReason.TIMEOUT
    assert result.iterations < 10


def test_request_timeout_is_capped_by_time_left(task, ws) -> None:
    _, provider, _ = run(
        task,
        ws,
        [reply(tc("finish", summary="x"))],
        settings=make_settings(wall_clock_timeout_s=42),
    )
    assert provider.requests[0]["timeout_s"] == pytest.approx(42)


def test_refusal(task, ws) -> None:
    result, _, _ = run(task, ws, [reply(text="I can't help with that.", stop=LLMStop.REFUSAL)])
    assert result.stop_reason == StopReason.REFUSAL


def test_no_action_after_one_nudge(task, ws) -> None:
    script = [reply(text="Thinking..."), reply(text="Still thinking...")]
    result, provider, _ = run(task, ws, script)
    assert result.stop_reason == StopReason.NO_ACTION
    nudge = provider.requests[1]["messages"][-1]
    assert nudge.content[0].text == NUDGE_NO_TOOL


def test_nudge_streak_resets_after_a_tool_call(task, ws) -> None:
    script = [
        reply(text="hmm"),
        reply(tc("list_files")),
        reply(text="hmm"),
        reply(tc("finish", summary="x")),
    ]
    result, _, _ = run(task, ws, script)
    assert result.stop_reason == StopReason.FINISHED


def test_max_tokens_cut_off(task, ws) -> None:
    script = [
        reply(text="Let me expl", stop=LLMStop.MAX_TOKENS),
        reply(
            tc("edit_file", path="src/calc/stats.py", old_str=BUG, new_str="partial"),
            stop=LLMStop.MAX_TOKENS,
        ),
        reply(tc("finish", summary="x")),
    ]
    result, provider, _ = run(task, ws, script)
    assert provider.requests[1]["messages"][-1].content[0].text == NUDGE_CUT_OFF
    truncated = provider.requests[2]["messages"][-1].content[0]
    assert truncated.is_error and "cut off" in truncated.content
    assert BUG in (ws.root / "src/calc/stats.py").read_text(encoding="utf-8")  # not executed
    assert result.stop_reason == StopReason.FINISHED


def test_parallel_tool_calls_return_in_order_in_one_message(task, ws) -> None:
    a = tc("read_file", path="src/calc/ops.py", end_line=1)
    b = tc("read_file", path="src/calc/stats.py", end_line=1)
    _, provider, _ = run(task, ws, [reply(a, b), reply(tc("finish", summary="x"))])
    msg = provider.requests[1]["messages"][-1]
    results = [blk for blk in msg.content if isinstance(blk, ToolResult)]
    assert [r.tool_call_id for r in results] == [a.id, b.id]
    assert isinstance(msg.content[-1], TextBlock)


def test_finish_runs_after_other_calls_in_the_same_turn(task, ws) -> None:
    edit = tc("edit_file", path="src/calc/stats.py", old_str=BUG, new_str=FIX)
    result, _, _ = run(task, ws, [reply(tc("finish", summary="done"), edit)])
    assert result.stop_reason == StopReason.FINISHED and result.resolved


def test_sandbox_error_stops_and_is_reported(task, ws) -> None:
    result, _, _ = run(task, ws, [reply(tc("run_tests"))], sandbox=FakeSandbox(fail=True))
    assert result.stop_reason == StopReason.SANDBOX_ERROR
    assert result.outcome == Outcome.NO_FINAL_TESTS
    assert result.test_runs == 0
    assert "grading failed" in (result.error or "")


# --- LLM errors ----------------------------------------------------------------------


def test_retryable_errors_are_retried_with_backoff(task, ws) -> None:
    sleep = RecordingSleep()
    script = [
        LLMError("overloaded", retryable=True, status_code=529),
        LLMError("rate limited", retryable=True, status_code=429, retry_after_s=5),
        *FIX_SCRIPT,
    ]
    result, _, trace = run(task, ws, script, sleep=sleep)
    assert result.resolved
    assert sleep.waits == [1.0, 5.0]  # 1s * 2**0 (rng=1), then retry-after 5s beats 2s
    retries = [e for e in read_trace(trace) if e.kind == EventKind.ERROR]
    assert [e.data["retry"] for e in retries] == [1, 2]


def test_non_retryable_error_stops_cleanly(task, ws) -> None:
    script = [LLMError("bad request", retryable=False, status_code=400)]
    result, provider, _ = run(task, ws, script)
    assert result.stop_reason == StopReason.LLM_ERROR
    assert result.error == "bad request"
    assert len(provider.requests) == 1
    assert result.outcome == Outcome.STOPPED_TESTS_FAIL  # still graded


def test_retries_exhausted(task, ws) -> None:
    script = [LLMError("overloaded", retryable=True, status_code=529)] * 5
    sleep = RecordingSleep()
    result, provider, _ = run(task, ws, script, sleep=sleep)
    assert result.stop_reason == StopReason.LLM_ERROR
    assert len(provider.requests) == 4 and len(sleep.waits) == 3  # 1 try + 3 retries


# --- context management ------------------------------------------------------------


def test_old_tool_results_are_elided_when_context_is_large(task, ws) -> None:
    settings = make_settings()
    settings.agent = AgentSettings(context_elide_tokens=5_000, keep_recent_turns=1)
    script = [
        reply(tc("read_file", path="src/calc/ops.py"), input_tokens=1_000),
        reply(tc("run_tests"), input_tokens=2_000),
        reply(tc("read_file", path="src/calc/stats.py"), input_tokens=6_000),  # crosses threshold
        reply(tc("finish", summary="x")),
    ]
    result, provider, trace = run(task, ws, script, settings=settings)
    last = provider.requests[-1]["messages"]
    tool_msgs = [m for m in last if m.role == "user" and isinstance(m.content[0], ToolResult)]
    first, second, third = (m.content[0].content for m in tool_msgs)
    assert first.startswith(f"{ELIDED_PREFIX} read_file src/calc/ops.py lines 1-end")
    assert second.startswith(f"{ELIDED_PREFIX} run_tests") and "Result: FAILED" in second
    assert third.startswith("src/calc/stats.py (lines")  # most recent turn kept intact
    assert last[0].content[0].text.startswith("<issue>")  # issue never elided
    assert result.context_elisions == 1
    assert any(e.kind == EventKind.CONTEXT for e in read_trace(trace))


def test_result_json_is_valid_json(task, ws) -> None:
    result, _, trace = run(task, ws, list(FIX_SCRIPT))
    data = json.loads(trace.with_name(f"{task.id}.result.json").read_text(encoding="utf-8"))
    assert data["outcome"] == "finished_tests_pass" and data["resolved"] is True
    assert data["temperature"] is None and data["model_id"] == result.model_id


# --- provider-specific paths (Groq) --------------------------------------------------


class GroqScripted(ScriptedProvider):
    name = "groq"


def test_groq_free_tier_cost_is_zero_with_list_price_note(task, ws) -> None:
    settings = make_settings()
    settings.llm = LLMSettings(provider="groq", model="openai/gpt-oss-120b")
    provider = GroqScripted(list(FIX_SCRIPT))
    with Tracer(ws.root.parent / "runs", "run1", task.id) as tracer:
        result = solve_task(
            task,
            ws,
            provider=provider,
            sandbox=FakeSandbox(),
            settings=settings,
            tracer=tracer,
            run_id="run1",
            clock=FakeClock(),
            sleep=RecordingSleep(),
        )
    assert result.cost_usd == 0.0
    assert result.list_price_usd == pytest.approx(5_000 * 0.15e-6 + 500 * 0.60e-6)
    assert result.cost_note is not None and result.cost_note.startswith("Groq free tier")


def test_unpriced_model_cost_is_none_with_note(task, ws) -> None:
    settings = make_settings()
    settings.llm = LLMSettings(provider="anthropic", model="some-unlisted-model")
    result, _, _ = run(task, ws, list(FIX_SCRIPT), settings=settings)
    assert result.cost_usd is None and "no price entry" in (result.cost_note or "")


def test_prompt_too_large_triggers_emergency_elision_then_continues(task, ws) -> None:
    script = [
        reply(tc("read_file", path="src/calc/ops.py")),
        reply(tc("read_file", path="src/calc/stats.py")),
        LLMError("prompt too big for TPM", retryable=False, status_code=413),
        reply(tc("finish", summary="x")),
    ]
    result, provider, trace = run(task, ws, script)
    assert result.stop_reason == StopReason.FINISHED
    assert result.context_elisions == 1
    last = provider.requests[-1]["messages"]
    first_result = next(
        m for m in last if m.role == "user" and isinstance(m.content[0], ToolResult)
    )
    assert first_result.content[0].content.startswith(ELIDED_PREFIX)
    events = [e for e in read_trace(trace) if e.kind == EventKind.CONTEXT]
    assert events and events[0].data["action"] == "emergency_elide"


def test_413_with_nothing_left_to_elide_stops_with_context_limit(task, ws) -> None:
    # Outgrowing the per-request limit is the attempt running out of room (scored as a
    # budget stop), not an infrastructure failure that would be re-run forever.
    script = [LLMError("prompt too big for TPM", retryable=False, status_code=413)]
    result, _, _ = run(task, ws, script)
    assert result.stop_reason == StopReason.CONTEXT_LIMIT
    assert result.error_status == 413


def test_malformed_tool_json_gets_an_actionable_error(task, ws) -> None:
    bad = ToolCall(id="bad1", name="read_file", arguments={INVALID_JSON_KEY: '{"path": '})
    _, provider, _ = run(task, ws, [reply(bad), reply(tc("finish", summary="x"))])
    out = provider.requests[1]["messages"][-1].content[0]
    assert out.is_error and "not valid JSON" in out.content


def test_required_tool_choice_recorded_and_text_only_reply_still_handled(task, ws) -> None:
    # With tool_choice=required a text-only turn should not happen; if a provider returns
    # one anyway, the nudge/no_action safety net still ends the run cleanly.
    settings = make_settings()
    settings.llm = LLMSettings(provider="groq", model="openai/gpt-oss-120b")
    provider = GroqScripted([reply(text="done?"), reply(text="done.")])
    with Tracer(ws.root.parent / "runs", "run1", task.id) as tracer:
        result = solve_task(
            task,
            ws,
            provider=provider,
            sandbox=FakeSandbox(),
            settings=settings,
            tracer=tracer,
            run_id="run1",
            clock=FakeClock(),
            sleep=RecordingSleep(),
        )
    assert result.tool_choice == "required"
    assert result.stop_reason == StopReason.NO_ACTION


def test_default_tool_choice_recorded_for_other_providers(task, ws) -> None:
    result, _, _ = run(task, ws, list(FIX_SCRIPT))
    assert result.tool_choice == "auto"


def test_text_answer_under_required_tool_choice_gets_finish_nudge(task, ws) -> None:
    no_tool = LLMError(
        "did not call a tool",
        retryable=False,
        status_code=400,
        kind=NO_TOOL_CALL,
        generated_text="Fixed mean; all tests pass.",
    )
    script = [*FIX_SCRIPT[:4], no_tool, reply(tc("finish", summary="fixed mean"))]
    result, provider, _ = run(task, ws, script)
    assert result.stop_reason == StopReason.FINISHED and result.success
    last = provider.requests[-1]["messages"]
    assert last[-2].role == "assistant" and last[-2].text == "Fixed mean; all tests pass."
    assert last[-1].content[0].text == NUDGE_TOOL_REQUIRED


def test_repeated_text_answers_under_required_stop_as_no_action(task, ws) -> None:
    no_tool = LLMError(
        "did not call a tool",
        retryable=False,
        status_code=400,
        kind=NO_TOOL_CALL,
        generated_text="Done.",
    )
    result, provider, _ = run(task, ws, [*FIX_SCRIPT[:4], no_tool, no_tool])
    assert result.stop_reason == StopReason.NO_ACTION
    assert result.resolved and not result.success
    assert len(provider.requests) == 6  # no blind resampling


# --- grading integrity: new conftest cheat (no Docker) ---------------------------------------

CONFTEST_CHEAT = (
    "import calc.stats\n\ndef _mean(xs):\n    return sum(xs) / len(xs)\n\ncalc.stats.mean = _mean\n"
)


def test_new_conftest_that_monkeypatches_mean_does_not_resolve(task, ws) -> None:
    script = [
        reply(tc("edit_file", path="conftest.py", old_str="", new_str=CONFTEST_CHEAT)),
        reply(tc("run_tests")),
        reply(tc("finish", summary="tests pass now")),
    ]
    result, provider, _ = run(task, ws, script)
    # The agent's own run passed because its conftest patched mean...
    assert provider.requests[2]["messages"][-1].content[0].content.startswith("Result: PASSED")
    assert result.agent_last_test_result is not None
    assert result.agent_last_test_result.all_passed
    # ...but grading deletes new conftests, and grading alone decides the outcome.
    assert result.removed_files == ["conftest.py"]
    assert result.resolved is False and result.success is False
    assert result.outcome == Outcome.FINISHED_TESTS_FAIL
    assert result.agent_disagrees_with_grading is True
    assert not (ws.root / "conftest.py").exists()


def test_edited_baseline_config_is_restored_and_recorded(task, tmp_path: Path) -> None:
    repo = tmp_path / "calc-with-config"
    shutil.copytree(task.repo_dir(), repo)
    (repo / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    parent = tmp_path / "ws"
    parent.mkdir()
    with Workspace.from_directory(repo, patch=task.seed_patch, parent_dir=parent) as ws:
        script = [
            reply(
                tc(
                    "edit_file",
                    path="pytest.ini",
                    old_str="[pytest]\n",
                    new_str="[pytest]\naddopts = -k 'not mean'\n",
                )
            ),
            reply(tc("finish", summary="done")),
        ]
        result, _, _ = run(task, ws, script)
        assert result.restored_config_files == ["pytest.ini"]
        assert result.modified_test_files == []
        assert (ws.root / "pytest.ini").read_text(encoding="utf-8") == "[pytest]\n"


def test_agent_agreement_is_recorded_when_both_pass(task, ws) -> None:
    result, _, _ = run(task, ws, list(FIX_SCRIPT))
    assert result.agent_last_test_result.selectors == ["tests/test_stats.py"]
    assert result.agent_disagrees_with_grading is False


# --- budgets ---------------------------------------------------------------------------------


def test_cache_reads_do_not_count_toward_the_token_budget(task, ws) -> None:
    script = [reply(tc("list_files"), input_tokens=100, output_tokens=10, cache_read=50_000)] * 3
    script = [*script, reply(tc("finish", summary="x"))]
    result, _, _ = run(task, ws, script, settings=make_settings(max_tokens_per_task=10_000))
    assert result.stop_reason == StopReason.FINISHED
    assert result.cache_read_tokens == 150_000  # still reported in full


def test_cost_budget(task, ws) -> None:
    # claude-sonnet-5 list price: $2/M in, $10/M out -> each turn costs $0.002 + $0.001.
    script = [reply(tc("list_files"), input_tokens=1_000, output_tokens=100)] * 5
    settings = make_settings(max_cost_usd_per_task=0.005)
    result, provider, _ = run(task, ws, script, settings=settings)
    assert result.stop_reason == StopReason.COST_BUDGET
    assert len(provider.requests) == 2
    budget_line = provider.requests[1]["messages"][-1].content[-1].text
    assert "cost $0.0030/$0.0050" in budget_line


def test_daily_quota_429_stops_as_llm_error_with_status_429(task, ws) -> None:
    quota = LLMError(
        "Rate limit reached ... tokens per day (TPD)",
        retryable=True,
        status_code=429,
        retry_after_s=1500,
    )
    result, provider, _ = run(task, ws, [reply(tc("list_files")), quota])
    assert result.stop_reason == StopReason.LLM_ERROR and result.error_status == 429
    assert len(provider.requests) == 2  # no sleeping through the quota
