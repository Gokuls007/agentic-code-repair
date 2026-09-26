"""Mutable loop state and the final result record."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from repair_agent.agent.grading import FinalTests
from repair_agent.llm.base import Usage


class StopReason(StrEnum):
    """Why the agent loop ended."""

    FINISHED = "finished"
    MAX_ITERATIONS = "max_iterations"
    TOKEN_BUDGET = "token_budget"
    TEST_BUDGET = "test_budget"
    TIMEOUT = "timeout"
    REFUSAL = "refusal"
    NO_ACTION = "no_action"
    LLM_ERROR = "llm_error"
    SANDBOX_ERROR = "sandbox_error"


class Outcome(StrEnum):
    """How the attempt ended, combined with the grading run's result."""

    FINISHED_TESTS_PASS = "finished_tests_pass"
    FINISHED_TESTS_FAIL = "finished_tests_fail"
    STOPPED_TESTS_PASS = "stopped_tests_pass"
    STOPPED_TESTS_FAIL = "stopped_tests_fail"
    NO_FINAL_TESTS = "no_final_tests"


class AgentState(BaseModel):
    """Counters the loop updates as it runs."""

    iteration: int = 0
    test_runs: int = 0
    usage: Usage = Field(default_factory=Usage)
    llm_s: float = 0.0
    tool_calls: dict[str, int] = Field(default_factory=dict)
    no_action_streak: int = 0
    finish_summary: str | None = None
    model_id: str | None = None
    elisions: int = 0


class AgentResult(BaseModel):
    """Everything recorded about one task attempt. Saved as ``<task_id>.result.json``."""

    run_id: str
    task_id: str
    provider: str
    model: str = Field(description="Model requested in config.")
    model_id: str | None = Field(description="Exact model id returned by the API.")
    temperature: float | None = Field(description="Temperature sent; None = model default.")
    tool_choice: str = Field(default="auto", description="tool_choice sent: auto or required.")
    started_at: datetime
    ended_at: datetime

    stop_reason: StopReason
    outcome: Outcome
    success: bool = Field(description="Agent called finish and the full suite passes.")
    resolved: bool = Field(description="All FAIL_TO_PASS and PASS_TO_PASS tests pass.")
    finish_summary: str | None = None
    error: str | None = None
    error_status: int | None = Field(
        default=None, description="HTTP status of the LLM error that stopped the run, if any."
    )

    iterations: int
    test_runs: int
    tool_calls: dict[str, int]
    context_elisions: int = 0

    tokens_in: int
    tokens_out: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None = Field(description="Amount charged; 0.0 on a free tier.")
    list_price_usd: float | None = Field(
        default=None, description="Cost at the provider's list price (None if unknown)."
    )
    cost_note: str | None = None
    wall_s: float
    llm_s: float

    final_tests: FinalTests | None
    modified_test_files: list[str] = Field(
        default_factory=list, description="Original test/config files the agent edited (restored)."
    )
    removed_files: list[str] = Field(
        default_factory=list, description="Added files that could affect collection (deleted)."
    )
    source_files_changed: list[str] = Field(default_factory=list)
    missing_graded_tests: list[str] = Field(default_factory=list)
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0
    diff: str
