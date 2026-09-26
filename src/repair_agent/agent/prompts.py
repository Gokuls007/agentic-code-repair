"""Prompt text. The system prompt is static so it is byte-identical (cacheable) across tasks."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an autonomous software engineer fixing a bug in a Python repository. You work only \
through the provided tools; no human is available to answer questions.

How to work:
1. Understand the issue. Explore with list_files, search_code and read_file. Read code before \
you edit it.
2. Reproduce. Run the relevant tests with run_tests (use test_selector to keep runs fast) and \
confirm the failure the issue describes. If no test covers it, you may add one.
3. Fix the root cause in the source code with the smallest change that resolves the issue. \
Match the surrounding style.
4. Verify. Re-run the targeted tests, then the full suite, to make sure nothing else broke.
5. Call finish with a short summary: the cause, what you changed, and how you verified it.

Rules:
- Never modify, delete, or skip existing tests to make them pass.
- Budgets (iterations, test runs, tokens, time) are hard limits; the remaining budget is shown \
after each round of tool results. Test runs are the scarcest; prefer targeted selectors.
- edit_file needs old_str to match the file exactly; copy it from read_file output without the \
line-number prefix.
- Tool output may be truncated; narrow the request (line range, selector, glob) to see more.
- If you cannot make progress, call finish and explain what you found."""

NUDGE_NO_TOOL = "No tool was called. Keep working using the tools, or call finish if you are done."
NUDGE_TOOL_REQUIRED = (
    "Every turn must be a tool call. If the fix is done and verified, call finish with your "
    "summary now; otherwise continue with the next tool call."
)
FINISH_ONLY_REJECTED = (
    "Error: the test-run budget is exhausted, so only finish is accepted now. "
    "This call was not run and the attempt ends here."
)


def test_budget_exhausted(used: str) -> str:
    """Error result for a run_tests request past the budget (the model gets one last turn)."""
    return (
        f"Error: test-run budget exhausted ({used} used); this run was not executed. "
        "You have one final turn: call finish now with your summary. Any other tool call "
        "ends the attempt."
    )


NUDGE_CUT_OFF = (
    "Your reply was cut off by the output-token limit. Continue, keeping text brief and "
    "using tools."
)
TRUNCATED_CALL = (
    "This tool call was cut off by the output-token limit, so it was not run. "
    "Retry with smaller arguments (e.g. a shorter old_str/new_str)."
)


def initial_prompt(title: str, body: str, tree: str) -> str:
    """First user message: the issue plus a file-tree summary of the repository."""
    return (
        f"<issue>\n{title.strip()}\n\n{body.strip()}\n</issue>\n\n"
        "<repository>\nAll paths are relative to the repository root.\n"
        f"{tree.rstrip()}\n</repository>\n\n"
        "Fix the issue described above."
    )


def budget_status(
    *,
    iteration: int,
    max_iterations: int,
    test_runs: int,
    max_test_runs: int,
    tokens: int,
    max_tokens: int,
    elapsed_s: float,
    timeout_s: float,
    cost_usd: float | None = None,
    max_cost_usd: float | None = None,
) -> str:
    """One-line budget reminder appended after each round of tool results."""
    line = (
        f"[budget] iteration {iteration}/{max_iterations} · test runs {test_runs}/{max_test_runs}"
        f" · tokens {tokens:,}/{max_tokens:,} · time {_mmss(elapsed_s)}/{_mmss(timeout_s)}"
    )
    if max_cost_usd is not None:
        line += f" · cost ${cost_usd or 0:.4f}/${max_cost_usd:.4f}"
    return line


def _mmss(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"
