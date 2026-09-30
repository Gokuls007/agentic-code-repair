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
    COST_BUDGET = "cost_budget"
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


class AgentTestSummary(BaseModel):
    """The agent's own most recent run_tests call (may be a targeted selection)."""

    selectors: list[str] = Field(default_factory=list, description="Empty = full suite.")
    passed: int = 0
    failed: int = 0
    errors: int = 0
    exit_code: int | None = None
    timed_out: bool = False

    @property
    def all_passed(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0

    @classmethod
    def from_metadata(cls, meta: dict) -> AgentTestSummary:
        """Build from run_tests ToolOutput metadata."""
        return cls(
            selectors=list(meta.get("selectors", [])),
            passed=meta.get("passed", 0),
            failed=meta.get("failed", 0),
            errors=meta.get("errors", 0),
            exit_code=meta.get("exit_code"),
            timed_out=bool(meta.get("timed_out")),
        )


class BackendUsage(BaseModel):
    """Requests and tokens served by one backend during an attempt."""

    requests: int = 0
    failovers_from: int = Field(
        default=0, description="Requests this backend failed that another backend then served."
    )
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0

    def usage(self) -> Usage:
        return Usage(
            input_tokens=self.tokens_in,
            output_tokens=self.tokens_out,
            cache_read_input_tokens=self.cache_read_tokens,
        )


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
    # Set once run_tests is requested past the budget: the next turn may only call finish.
    finish_only: bool = False
    last_agent_tests: AgentTestSummary | None = None
    # Per backend that answered (one entry unless the provider is a pool).
    backend_usage: dict[str, BackendUsage] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Everything recorded about one task attempt. Saved as ``<task_id>.result.json``.

    Where each field comes from:

    - **Grading run** (after the loop, original tests and config restored, graded ids
      only; the single source of truth for the task's result): ``outcome``, ``success``,
      ``resolved``, ``final_tests``, ``f2p_*``, ``p2p_*``, ``missing_graded_tests``,
      ``modified_test_files``, ``restored_config_files``, ``removed_files``.
    - **The agent's own test runs** (inside the loop; informational only, never used for
      the outcome): ``test_runs``, ``agent_last_test_result``, ``agent_disagrees_with_grading``.
    - **The loop**: ``stop_reason``, ``finish_summary``, ``iterations``, ``tool_calls``,
      token counts, cost, timings, ``error``.
    - **The workspace** (captured before grading restores anything): ``diff``,
      ``source_files_changed``.
    """

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
    outcome: Outcome = Field(description="Stop reason combined with the grading run's result.")
    success: bool = Field(description="Agent called finish and the grading run passes.")
    resolved: bool = Field(description="Grading run: all FAIL_TO_PASS and PASS_TO_PASS pass.")
    finish_summary: str | None = None
    error: str | None = None
    error_status: int | None = Field(
        default=None, description="HTTP status of the LLM error that stopped the run, if any."
    )

    iterations: int
    test_runs: int = Field(description="run_tests calls the agent made (grading not counted).")
    agent_last_test_result: AgentTestSummary | None = Field(
        default=None, description="The agent's last own test run; informational only."
    )
    agent_disagrees_with_grading: bool | None = Field(
        default=None,
        description="Agent's last run passed/failed while the grading run said the opposite.",
    )
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
    backend_usage: dict[str, BackendUsage] = Field(
        default_factory=dict,
        description="Requests and tokens per backend that answered (provider pools).",
    )
    wall_s: float
    llm_s: float

    final_tests: FinalTests | None
    modified_test_files: list[str] = Field(
        default_factory=list, description="Original test files the agent edited (restored)."
    )
    restored_config_files: list[str] = Field(
        default_factory=list,
        description="Original test config the agent edited (conftest.py, pytest.ini, "
        "pyproject.toml, setup.cfg, tox.ini), restored before grading.",
    )
    removed_files: list[str] = Field(
        default_factory=list,
        description="Files the agent added that can affect test collection or start-up "
        "(new conftest.py, test config, sitecustomize/usercustomize, *.pth), deleted.",
    )
    source_files_changed: list[str] = Field(default_factory=list)
    missing_graded_tests: list[str] = Field(default_factory=list)
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0
    diff: str
