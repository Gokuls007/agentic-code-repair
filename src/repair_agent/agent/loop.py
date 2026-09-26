"""ReAct-style agent loop with hard budgets, plus the end-to-end ``solve_task`` entry point."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from repair_agent.agent import prompts
from repair_agent.agent.context import elide_old_tool_results
from repair_agent.agent.grading import grade
from repair_agent.agent.retry import call_with_retry
from repair_agent.agent.state import AgentResult, AgentState, Outcome, StopReason
from repair_agent.agent.task import Task
from repair_agent.config import Settings
from repair_agent.llm.base import (
    NO_TOOL_CALL,
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from repair_agent.llm.base import StopReason as LLMStop
from repair_agent.llm.pricing import estimate_cost
from repair_agent.sandbox.docker import DockerSandbox, SandboxError
from repair_agent.sandbox.workspace import Workspace
from repair_agent.tools import build_registry
from repair_agent.tools.base import ToolOutput, ToolRegistry
from repair_agent.tools.files import ListFiles
from repair_agent.tracing import EventKind, Tracer

RUN_TESTS = "run_tests"
FINISH = "finish"


@dataclass
class LoopResult:
    """What the loop hands back to :func:`solve_task`."""

    stop_reason: StopReason
    state: AgentState
    messages: list[Message] = field(default_factory=list)
    error: str | None = None
    error_status: int | None = None


class AgentLoop:
    """Drives one conversation: model turn -> tool calls -> results -> ... until a stop."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        registry: ToolRegistry,
        settings: Settings,
        tracer: Tracer,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ):
        self.provider = provider
        self.registry = registry
        self.settings = settings
        self.tracer = tracer
        self.clock = clock
        self.sleep = sleep
        self.rng = rng
        self._started = 0.0

    # --- budgets --------------------------------------------------------------

    def elapsed(self) -> float:
        return self.clock() - self._started

    def time_left(self) -> float:
        return self.settings.budget.wall_clock_timeout_s - self.elapsed()

    def _budget_stop(self, state: AgentState) -> StopReason | None:
        b = self.settings.budget
        if state.iteration >= b.max_iterations:
            return StopReason.MAX_ITERATIONS
        if state.usage.total_tokens >= b.max_tokens_per_task:
            return StopReason.TOKEN_BUDGET
        if self.time_left() <= 0:
            return StopReason.TIMEOUT
        return None

    def _budget_text(self, state: AgentState) -> str:
        b = self.settings.budget
        return prompts.budget_status(
            iteration=state.iteration,
            max_iterations=b.max_iterations,
            test_runs=state.test_runs,
            max_test_runs=b.max_test_runs,
            tokens=state.usage.total_tokens,
            max_tokens=b.max_tokens_per_task,
            elapsed_s=self.elapsed(),
            timeout_s=b.wall_clock_timeout_s,
        )

    # --- main loop ------------------------------------------------------------

    def run(self, initial_prompt: str) -> LoopResult:
        """Run until a stop condition. Never raises for LLM or tool failures."""
        self._started = self.clock()
        state = AgentState()
        messages = [Message(role="user", content=[TextBlock(text=initial_prompt)])]
        specs = self.registry.specs()

        while True:
            stop = self._budget_stop(state)
            if stop:
                return LoopResult(stop, state, messages)

            state.iteration += 1
            self.tracer.next_step()
            try:
                response = self._call_llm(messages, specs)
            except LLMError as exc:
                if exc.kind == NO_TOOL_CALL:
                    # Tool use was required but the model replied in text: treat it as a
                    # text-only turn (keep what it said, then ask it to call finish).
                    self.tracer.log(
                        EventKind.ERROR,
                        {
                            "source": "llm",
                            "status": exc.status_code,
                            "kind": exc.kind,
                            "message": str(exc),
                        },
                    )
                    state.no_action_streak += 1
                    if state.no_action_streak >= 2:
                        return LoopResult(StopReason.NO_ACTION, state, messages)
                    if exc.generated_text:
                        self.tracer.log(EventKind.THOUGHT, {"text": exc.generated_text})
                        messages.append(
                            Message(role="assistant", content=[TextBlock(text=exc.generated_text)])
                        )
                    messages.append(_user_text(prompts.NUDGE_TOOL_REQUIRED))
                    continue
                if exc.status_code == 413 and not exc.retryable:
                    # Prompt too large for the provider's per-request limit: trim old tool
                    # output as far as possible and try again (at most once per cut).
                    count = elide_old_tool_results(messages, keep_recent_turns=1)
                    if count:
                        state.elisions += 1
                        self.tracer.log(
                            EventKind.CONTEXT,
                            {
                                "action": "emergency_elide",
                                "tool_results_elided": count,
                                "reason": str(exc),
                            },
                        )
                        continue
                reason = StopReason.TIMEOUT if self.time_left() <= 0 else StopReason.LLM_ERROR
                self.tracer.log(
                    EventKind.ERROR,
                    {"source": "llm", "status": exc.status_code, "message": str(exc)},
                )
                return LoopResult(
                    reason, state, messages, error=str(exc), error_status=exc.status_code
                )

            self._record_response(state, response)
            messages.append(response.message)

            if response.stop_reason == LLMStop.REFUSAL:
                return LoopResult(StopReason.REFUSAL, state, messages)

            calls = response.message.tool_calls
            if response.stop_reason == LLMStop.MAX_TOKENS and calls:
                # Arguments may be incomplete; answer every call without running it.
                results = [
                    ToolResult(tool_call_id=c.id, content=prompts.TRUNCATED_CALL, is_error=True)
                    for c in calls
                ]
                messages.append(self._results_message(results, state))
                continue
            if not calls:
                if response.stop_reason == LLMStop.MAX_TOKENS:
                    messages.append(_user_text(prompts.NUDGE_CUT_OFF))
                    continue
                state.no_action_streak += 1
                if state.no_action_streak >= 2:
                    return LoopResult(StopReason.NO_ACTION, state, messages)
                messages.append(_user_text(prompts.NUDGE_NO_TOOL))
                continue
            state.no_action_streak = 0

            results, stop = self._run_tools(calls, state)
            if stop == StopReason.TIMEOUT:
                return LoopResult(stop, state, messages)
            messages.append(self._results_message(results, state))
            if stop:
                return LoopResult(stop, state, messages)
            self._maybe_elide(messages, response, state)

    def _call_llm(self, messages: list[Message], specs: list[ToolSpec]) -> LLMResponse:
        llm = self.settings.llm

        def attempt() -> LLMResponse:
            timeout = max(1.0, min(llm.request_timeout_s, self.time_left()))
            return self.provider.complete(
                system=prompts.SYSTEM_PROMPT, messages=messages, tools=specs, timeout_s=timeout
            )

        def on_retry(n: int, exc: LLMError, wait: float) -> None:
            self.tracer.log(
                EventKind.ERROR,
                {
                    "source": "llm",
                    "retry": n,
                    "status": exc.status_code,
                    "wait_s": round(wait, 2),
                    "message": str(exc),
                },
            )

        return call_with_retry(
            attempt,
            max_retries=llm.max_retries,
            base_s=llm.retry_base_delay_s,
            cap_s=llm.retry_max_delay_s,
            time_left=self.time_left,
            sleep=self.sleep,
            on_retry=on_retry,
            rng=self.rng,
        )

    def _record_response(self, state: AgentState, response: LLMResponse) -> None:
        state.usage = state.usage + response.usage
        state.llm_s += response.latency_s
        state.model_id = response.model
        u = response.usage
        self.tracer.log(
            EventKind.LLM_CALL,
            {
                "model": response.model,
                "stop_reason": str(response.stop_reason),
                "latency_s": round(response.latency_s, 3),
                "cache_read_tokens": u.cache_read_input_tokens,
                "cache_write_tokens": u.cache_creation_input_tokens,
            },
            tokens_in=u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens,
            tokens_out=u.output_tokens,
        )
        if response.message.text:
            self.tracer.log(EventKind.THOUGHT, {"text": response.message.text})

    def _run_tools(
        self, calls: list[ToolCall], state: AgentState
    ) -> tuple[list[ToolResult], StopReason | None]:
        """Execute one round of tool calls (finish last). Returns results and a stop, if any."""
        ordered = [c for c in calls if c.name != FINISH] + [c for c in calls if c.name == FINISH]
        by_id: dict[str, ToolResult] = {}
        stop: StopReason | None = None
        for call in ordered:
            if self.time_left() <= 0:
                return [], StopReason.TIMEOUT
            state.tool_calls[call.name] = state.tool_calls.get(call.name, 0) + 1
            self.tracer.log(
                EventKind.TOOL_CALL, {"id": call.id, "name": call.name, "arguments": call.arguments}
            )
            if call.name == RUN_TESTS and state.test_runs >= self.settings.budget.max_test_runs:
                output = ToolOutput(
                    content=(
                        f"Error: test-run budget exhausted "
                        f"({state.test_runs}/{self.settings.budget.max_test_runs} used)."
                    ),
                    is_error=True,
                )
                stop = stop or StopReason.TEST_BUDGET
            else:
                output = self.registry.execute(call)
                if call.name == RUN_TESTS and not output.metadata.get("sandbox_error"):
                    state.test_runs += 1
            self.tracer.log(
                EventKind.TOOL_RESULT,
                {
                    "id": call.id,
                    "name": call.name,
                    "is_error": output.is_error,
                    "truncated": output.truncated,
                    "content": output.content,
                    "metadata": output.metadata,
                },
            )
            if output.metadata.get("sandbox_error"):
                stop = StopReason.SANDBOX_ERROR
            if output.metadata.get("finished"):
                state.finish_summary = output.metadata.get("summary")
                stop = StopReason.FINISHED
            by_id[call.id] = output.to_result(call.id)
        return [by_id[c.id] for c in calls if c.id in by_id], stop  # original order

    def _results_message(self, results: list[ToolResult], state: AgentState) -> Message:
        return Message(role="user", content=[*results, TextBlock(text=self._budget_text(state))])

    def _maybe_elide(
        self, messages: list[Message], response: LLMResponse, state: AgentState
    ) -> None:
        u = response.usage
        prompt_tokens = u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens
        if prompt_tokens <= self.settings.agent.context_elide_tokens:
            return
        count = elide_old_tool_results(messages, self.settings.agent.keep_recent_turns)
        if count:
            state.elisions += 1
            self.tracer.log(
                EventKind.CONTEXT,
                {"action": "elide", "tool_results_elided": count, "prompt_tokens": prompt_tokens},
            )


def _user_text(text: str) -> Message:
    return Message(role="user", content=[TextBlock(text=text)])


# --- end-to-end -----------------------------------------------------------------


def repo_tree(root: Path, settings: Settings) -> str:
    """File-tree summary for the initial prompt (same format as the list_files tool)."""
    tool = ListFiles(root, max_entries=settings.agent.tree_max_entries)
    return tool.run(tool.Args(depth=settings.agent.tree_depth)).content


def solve_task(
    task: Task,
    workspace: Workspace,
    *,
    provider: LLMProvider,
    sandbox: DockerSandbox,
    settings: Settings,
    tracer: Tracer,
    run_id: str,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
    result_path: Path | None = None,
) -> AgentResult:
    """Run the agent on ``task`` in ``workspace``, grade it, and write the result record."""
    started_at = datetime.now(UTC)
    wall_start = time.perf_counter()
    registry = build_registry(workspace.root, sandbox, settings)
    prompt = prompts.initial_prompt(
        task.issue.title, task.issue.body, repo_tree(workspace.root, settings)
    )
    tracer.log(
        EventKind.TASK_START,
        {
            "task_id": task.id,
            "provider": provider.name,
            "model": settings.llm.model,
            "budget": settings.budget.model_dump(),
            "prompt": prompt,
        },
    )

    loop = AgentLoop(
        provider=provider,
        registry=registry,
        settings=settings,
        tracer=tracer,
        clock=clock,
        sleep=sleep,
        rng=rng,
    )
    result = loop.run(prompt)
    state = result.state
    diff = workspace.diff()  # the agent's changes, captured before test files are restored

    error = result.error
    try:
        graded = grade(task, workspace, sandbox)
    except SandboxError as exc:
        graded = None
        error = error or f"grading failed: {exc}"
    if graded is not None:
        tracer.log(EventKind.GRADE, graded.model_dump(mode="json"))

    outcome = _outcome(result.stop_reason, graded.final_tests.all_passed if graded else None)
    cost, list_price, cost_note = _cost(provider.name, state, settings)
    record = AgentResult(
        run_id=run_id,
        task_id=task.id,
        provider=provider.name,
        model=settings.llm.model,
        model_id=state.model_id,
        temperature=settings.llm.temperature,
        tool_choice=settings.llm.tool_choice_for(provider.name),
        started_at=started_at,
        ended_at=datetime.now(UTC),
        stop_reason=result.stop_reason,
        outcome=outcome,
        success=outcome == Outcome.FINISHED_TESTS_PASS,
        resolved=bool(graded and graded.resolved),
        finish_summary=state.finish_summary,
        error=error,
        error_status=result.error_status,
        iterations=state.iteration,
        test_runs=state.test_runs,
        tool_calls=dict(state.tool_calls),
        context_elisions=state.elisions,
        tokens_in=state.usage.input_tokens,
        tokens_out=state.usage.output_tokens,
        cache_read_tokens=state.usage.cache_read_input_tokens,
        cache_write_tokens=state.usage.cache_creation_input_tokens,
        cost_usd=cost,
        list_price_usd=list_price,
        cost_note=cost_note,
        wall_s=round(time.perf_counter() - wall_start, 2),
        llm_s=round(state.llm_s, 2),
        final_tests=graded.final_tests if graded else None,
        modified_test_files=graded.modified_test_files if graded else [],
        removed_files=graded.removed_files if graded else [],
        source_files_changed=graded.source_files_changed if graded else [],
        missing_graded_tests=graded.missing_graded_tests if graded else [],
        f2p_passed=graded.f2p_passed if graded else 0,
        f2p_total=graded.f2p_total if graded else len(task.fail_to_pass),
        p2p_passed=graded.p2p_passed if graded else 0,
        p2p_total=graded.p2p_total if graded else len(task.pass_to_pass),
        diff=diff,
    )
    tracer.log(EventKind.TASK_END, record.model_dump(mode="json", exclude={"diff"}))
    out = result_path or tracer.path.with_name(f"{task.id}.result.json")
    out.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return record


def _cost(
    provider: str, state: AgentState, settings: Settings
) -> tuple[float | None, float | None, str | None]:
    """(charged, list-price equivalent, note). Free-tier usage is $0.00, never None."""
    list_price = estimate_cost(state.usage, settings.llm.model, settings.pricing)
    if provider == "groq" and settings.llm.groq_free_tier:
        note = "Groq free tier: no charge"
        if list_price is not None:
            note += f" (list-price equivalent ${list_price:.4f})"
        return 0.0, list_price, note
    if list_price is None:
        return None, None, f"no price entry for {settings.llm.model!r}"
    return list_price, list_price, None


def _outcome(stop: StopReason, all_passed: bool | None) -> Outcome:
    if all_passed is None:
        return Outcome.NO_FINAL_TESTS
    if stop == StopReason.FINISHED:
        return Outcome.FINISHED_TESTS_PASS if all_passed else Outcome.FINISHED_TESTS_FAIL
    return Outcome.STOPPED_TESTS_PASS if all_passed else Outcome.STOPPED_TESTS_FAIL
