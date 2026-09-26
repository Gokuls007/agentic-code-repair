# Results

Only numbers from actual runs appear here. Each row links to the trace and result JSON it
came from (`runs/` is gitignored, so those files are local).

## Smoke tests (single runs, not benchmarks)

These show the pipeline works end to end. One task, one attempt, so there's no resolve
rate or variance here. Benchmark numbers start in Phase 4.

| Date | Task | Provider / model | Resolved | Outcome (stop) | Iterations | Test runs | Tokens in / out (cache read) | Cost | Wall / LLM time | Run id |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-26 | calc-mean-001 | Groq / `openai/gpt-oss-120b` (temperature: default, effort: medium) | ✅ yes | `stopped_tests_pass` (`llm_error`) | 10 | 3 | 15,474 / 595 (1,024) | $0.00 free tier (list-price equivalent $0.0027) | 84.4 s / 5.6 s | `20260926-015547-2d98df` |
| 2026-09-26 | calc-mean-001 | Groq / `openai/gpt-oss-120b` (temperature: default, effort: medium, tool_choice: required) | ✅ yes | `finished_tests_pass` (`finished`) | 9 | 2 | 13,871 / 1,127 (3,072) | $0.00 free tier (list-price equivalent $0.0028) | 124.0 s / 6.4 s | `20260926-153756-d4f86c` |

**Notes on run `20260926-153756-d4f86c`** (after the Phase 4 changes: `tool_choice=required`
with the no-tool-call nudge, graded-ids-only grading, and the new budgets)
- The agent fixed the denominator in `mean` and called `finish`, so it ended cleanly.
- Grading passed all 6 FAIL_TO_PASS and 22 PASS_TO_PASS tests.
- The agent's own last test run was targeted (`tests/test_stats.py`, 16 passed) and agreed
  with grading.
- Budget tokens (uncached input + cache writes + output): 14,998.

**Notes on run `20260926-015547-2d98df`**
- **What the agent did.** It explored the repo, reproduced the failure with a targeted test
  run, fixed the denominator in `mean` with a one-line edit, then re-ran the targeted tests
  and the full suite (6 of 6 pass). It edited no test files. Grading, with the original
  tests restored, gives 6 of 6 passing.
- **It did not finish cleanly.** Turn 9 wrote the summary as text instead of calling
  `finish`, which triggered a nudge. Turn 10 failed with Groq 400 `output_parse_failed`,
  which at the time wasn't classed as retryable. So `success` is false even though
  `resolved` is true. That error code has been retryable since this run (DECISIONS #20).
- **Rate limits.** There were 8 free-tier 429s. Each was retried after the server's
  `retry-after` wait (1–18 s) and succeeded. That's why wall time (84 s) is far above model
  time (5.6 s).
- **Earlier attempts.** The two Anthropic attempts before this (2026-09-26) never reached
  the model: the account had no credit (HTTP 400). They aren't results and aren't listed.
