# Design decisions

Each entry: the decision, the alternatives considered, and why.

## 1. Own neutral message types instead of reusing one provider's SDK types (Phase 1)

**Decision.** `llm/base.py` defines `Message`, `TextBlock`, `ToolCall`, `ToolResult`,
`ToolSpec`, `Usage`, and `LLMResponse`. Each provider translates to and from these.

**Alternatives.** (a) Use Anthropic SDK types everywhere and adapt Groq to them. (b) Use
LiteLLM or a similar abstraction layer.

**Why.** The agent loop, tracing, and eval should not depend on any one vendor, and the
Groq-vs-Anthropic comparison in Phase 6 needs an even footing. A dependency like LiteLLM
would hide the exact request shape, and that matters when debugging tool-use behavior. The
translation layer is small and fully unit-tested with a fake client.

## 2. Keep native assistant blocks for verbatim replay (`provider_raw`) (Phase 1)

**Decision.** Assistant messages carry the provider's native content blocks next to the
neutral ones. A provider replays them only when the message came from that same provider.

**Alternatives.** Drop anything the neutral format can't express.

**Why.** Current Claude models emit thinking blocks with signatures that should be passed
back unchanged on the next turn. Dropping them would degrade multi-turn reasoning without
any error. When a conversation moves across providers, the neutral blocks are used instead.

## 3. Manual tool-use loop, not the SDK's beta tool runner (Phase 1, affects Phase 3)

**Decision.** The agent loop (Phase 3) calls `LLMProvider.complete()` once per iteration
and runs the tools itself.

**Alternatives.** `client.beta.messages.tool_runner`.

**Why.** The spec needs things a runner hides: hard budgets checked between every call
(tokens, test runs, wall clock), per-step tracing, output truncation, and provider
independence. The tool runner is also Anthropic-only and in beta.

## 4. Settings layout: `REPAIR_` prefix, `__` nesting, secrets unprefixed (Phase 1)

**Decision.** `pydantic-settings` with nested models (`llm`, `budget`, `tools`, `github`).
Secrets use their standard env names and are held as `SecretStr`.

**Alternatives.** A flat settings class, or a YAML config file.

**Why.** Nesting keeps related limits together and makes validation errors readable. Using
standard secret names means existing shells and CI setups work unchanged. `SecretStr`
keeps keys out of reprs and dumps, and `repair-agent config` masks them explicitly.

## 5. Tracing: one JSONL file per task, flushed per event, redacted before write (Phase 1)

**Decision.** `runs/<run_id>/<task_id>.jsonl`. Each event is validated by Pydantic, scrubbed
of secrets, written, and flushed right away.

**Alternatives.** SQLite, OpenTelemetry, or buffering in memory and writing at the end.

**Why.** JSONL is append-only, safe when a run crashes, easy to `grep`, and trivial for the
Phase 5 dashboard to stream. Redaction covers both the configured secret values and
common credential patterns (`sk-ant-…`, `gsk_…`, `ghp_…`, `github_pat_…`), because tool
output such as `env` or a stack trace can leak keys the agent never saw directly. IDs are
checked so a malicious task id can't escape `runs/`.

## 6. Cost is an estimate from a config price table, `None` when unknown (Phase 1)

**Decision.** `llm/pricing.py` multiplies usage by a per-model price table in config, with
Anthropic's standard cache multipliers (0.1× for reads, 1.25× for writes).

**Alternatives.** Hardcode prices, or skip cost tracking.

**Why.** Prices change, so they live in config and can be overridden. An unknown model
returns `None`, not `0.0`, so a report never shows a fake "$0.00".

## 7. No thinking/effort overrides by default (Phase 1)

**Decision.** The model comes from config (default changed in #8). The provider sends
`output_config.effort` only when `REPAIR_LLM__EFFORT` is set. It never sends a `thinking`
parameter, so the model's default adaptive thinking applies.

**Why.** It keeps the request minimal and valid across current models. Effort is the knob
worth sweeping in Phase 6. Refusals come back as `StopReason.REFUSAL`, so the loop can
record them as their own failure mode. Server-side refusal fallbacks are not enabled yet.

## 8. Default model `claude-sonnet-5` during development (Phase 1, revised)

**Decision.** The default model changed from `claude-opus-5` to `claude-sonnet-5`
($2 / $10 per MTok vs $5 / $25).

**Alternatives.** Keep Opus as the default, or use Haiku 4.5 for everything.

**Why.** Cost during development. One agent loop can use tens of thousands of tokens per
task, and a full eval runs 20-30 tasks, so the Opus price adds up quickly while the loop is
still being debugged. Haiku would be cheaper but likely weakens the baseline so much that
later improvements would be hard to read. Model choice is not settled: Phase 6 compares
Sonnet and Opus on the same benchmark, and each gets its own row in RESULTS.md.

## 9. Host workspace + fresh container per test run, no bind mounts (Phase 2)

**Decision.** The agent's working copy is a git clone on the host. list/read/search/edit
operate on it directly. Each `run_tests` call copies a tar snapshot into a brand-new
container, runs pytest, collects JUnit XML, and removes the container.

**Alternatives.** (a) Bind-mount the workspace into a long-lived container and run every
tool through `docker exec`. (b) A long-lived container per task with files synced in.

**Why.** On Windows, bind mounts mean path translation, slow 9p file sharing, and UID
mismatches. Mounts are also read-write, so a test could modify the agent's source of truth.
The file tools never execute repo code, so running them on the host doesn't weaken the
sandbox, and it keeps them fast and testable without Docker. A fresh container per run means
no state leaks between runs. The cost is about 1 second of startup per run, which is fine
with a cap of 10 runs per task. Timeouts are enforced from the host (`wait(timeout)` then
`kill`), so a hung process can't outlive the budget. Measured on this machine: a 3-second
limit killed an infinite loop at 3.4 seconds, and a 2 GB allocation under a 256 MB limit was
killed by the kernel's out-of-memory killer (exit 137).

## 10. Tool contract: never raise, errors are data, tail-heavy truncation (Phase 2)

**Decision.** `ToolRegistry.execute` always returns a `ToolOutput`.
- Unknown tools, invalid arguments (the Pydantic models reject extra fields), and expected
  failures (`ToolError`) become `is_error=True` results with a message the model can act on.
- Unexpected exceptions show only the exception type and message to the model; the full
  traceback goes to `metadata` for the trace.
- Failing tests are **not** errors. They're a valid `run_tests` result.
- Truncation has two layers. Each tool truncates in a way that fits its output (line pages,
  match caps, entry caps) and says how to see more. The registry then caps everything at
  `max_output_chars`, keeping the first third and the last two thirds.

**Alternatives.** Raise and let the loop catch; truncate from the head only.

**Why.** A tool failure is information the model can use: a re-read file, a unique
`old_str`, a narrower selector. Keeping the tail keeps the pytest summary and the end of
tracebacks, and the `run_tests` summary line comes first, so it survives either way.

## 11. Target repos via PYTHONPATH, not `pip install` (Phase 2)

**Decision.** The sandbox image contains only Python and pinned pytest. Target repos are
imported through `PYTHONPATH=/workspace/src:/workspace`.

**Alternatives.** `pip install -e .` per run, or a per-task image built from the repo.

**Why.** Installing would run the repo's build code and needs network access, which the
containers don't have. The benchmark repos are pure Python by design. Per-task images (an
`image` field in the task YAML) can come later for the optional SWE-bench adapter.

## 12. Byte-exact line endings (Phase 2)

**Decision.**
- All file tool I/O is in bytes.
- Workspace git commands always pass `core.autocrlf=false`.
- `edit_file` converts the model's LF text to CRLF for files that use CRLF throughout.
- ripgrep runs with `--path-separator /`.
- The repo's `.gitattributes` forces LF.

**Why.** On Windows, Python text mode and `autocrlf=true` rewrite line endings without
telling you. That corrupts diffs and makes exact-match edits fail. The CRLF test fixtures are
generated at test time, because committed CRLF files would be normalized to LF and the tests
would pass without testing anything.
## 13. Loop and stop semantics (Phase 3)

**Decision.** Each iteration is one model call followed by all of its tool calls. The
results come back as one user message, with a one-line budget status appended; it's only
ever appended, never edited, so it doesn't break caching.

Stop reasons:
- `finished`
- `max_iterations`, `token_budget`, and `timeout`: checked before every model call
  (`timeout` also before every tool call)
- `test_budget`: the model asks for a test run beyond the limit; it gets an error result
  and the loop ends
- `refusal`
- `no_action`: two text-only turns in a row, the first of which got a nudge
- `llm_error`
- `sandbox_error`

A reply cut off at `max_tokens` gets a "continue" nudge, and any tool calls in it are
answered with errors instead of being run, because their arguments may be incomplete. When
several calls come in one turn, `finish` always runs last.

**Why.** Every limit in the spec becomes a named, testable stop reason, and the eval can
break failures down by it. The token budget counts all billed tokens, cached reads
included. That's conservative, but it matches what a task really consumes.

## 14. Prompt caching plus batched elision of old tool results (Phase 3)

**Decision.**
- The Anthropic provider puts one cache marker on the system block, which also covers the
  tools because they are rendered before it. Top-level automatic caching covers the growing
  history.
- When a request's input exceeds `agent.context_elide_tokens` (60K), tool results older
  than the last 4 turns are replaced with one-line stubs; `run_tests` stubs keep their
  `Result:` line.
- The issue, the assistant turns, and tool-call arguments are never touched.

**Alternatives.** Anthropic's server-side context editing; summarizing old turns with an
LLM; eliding on every turn.

**Why.**
- Caching is the cheapest cost lever. The tool definitions and system prompt are identical
  on every call, and each turn resends the whole history.
- Elision is a history edit, so it invalidates the cache after the edit point. Doing it in
  batches, only when the threshold is crossed, keeps those cache misses rare.
- Doing it client-side keeps behavior identical across providers, which matters for the
  Groq comparison in Phase 6.

## 15. Our own retry layer; SDK retries off (Phase 3)

**Decision.** The SDK runs with `max_retries=0`. `agent/retry.py` retries errors marked
retryable (408, 409, 429, 5xx, connection errors, timeouts) with exponential backoff and
full jitter (2 s base, 30 s cap). A `retry-after` header sets a minimum wait. It never
sleeps past the task's wall-clock deadline. Each retry is traced. Anything else stops the
loop with `llm_error`, and the attempt is still graded and recorded.

**Why.** Two retry layers multiply each other, and SDK retries don't show up in our trace.
The wall-clock budget has to bound the backoff as well. Observed on the first real run: an
exhausted-credit 400 was, correctly, not retried.

## 16. `finish` triggers grading; grading restores original tests (Phase 3)

**Decision.**
- **Grading always runs** after the loop ends, whether through `finish` or another stop. It
  uses the full suite, and those runs don't count against the agent's test budget. Outcomes
  are `finished_tests_pass`, `finished_tests_fail`, `stopped_tests_pass`,
  `stopped_tests_fail`, and `no_final_tests`.
- **Test restore.** Before grading, every baseline file that is a graded test file or looks
  like a test file (`tests/`, `test_*.py`, `*_test.py`, `conftest.py`) is reset to its
  baseline content. The files the agent had changed are listed in `modified_test_files`.
- **Resolved** means every FAIL_TO_PASS and PASS_TO_PASS test passes in that restored run.
- **New test files** the agent adds stay in place.
- **Diff.** `diff` is captured before the restore, so an attempt to edit tests stays visible.

**Alternatives.** Trust the agent's last test run; reject `finish` when tests fail.

**Why.** An agent can "resolve" a bug by weakening the test. A unit test and a real-Docker
test both script exactly that cheat and confirm `resolved=False`. Rejecting `finish` is a
behavior change and belongs in Phase 6, where it can be measured.

**Known limitation.** A *new* file that monkeypatches the code, such as a new root
`conftest.py`, isn't covered. Phase 4 could also discard new conftest files before grading.

## 17. Temperature and model id recorded as sent and as returned (Phase 3)

**Decision.**
- `llm.temperature` defaults to `None`, meaning the parameter is not sent, so the model's
  default sampling applies.
- `AgentResult` records `temperature` as it was actually sent, plus both `model` (what was
  requested) and `model_id` (what the API reported).

**Why.** Reproducibility. The eval report can say exactly which model and sampling
settings produced each number.

**What was verified (corrected 2026-09-26).** An earlier version said "Sonnet 5 rejects
temperature with a 400". That was never tested. When checked:

| Check | Result |
|---|---|
| anthropic SDK 1.8 | Takes no `temperature` argument: `Messages.create()` raises `TypeError` client-side. The provider now sends a configured value raw via `extra_body`, so the API decides. |
| Anthropic API | **Not verified.** Two minimal requests to `claude-sonnet-5`, with and without `temperature`, were both rejected by the account's credit check (400 "credit balance is too low") before parameters were evaluated. Anthropic's docs say sampling parameters are rejected on Sonnet 5, but that is documentation, not a measurement. |
| Groq | Accepts it: `openai/gpt-oss-120b` with `temperature=0.2` returned a normal completion. |

The default stays "not sent" on every provider.

## 18. Phase 4: per-repo dependency images (planned)

The containers have no network, so third-party dependencies are baked into per-repo images
at build time:
- `FROM` the base sandbox image, running
  `pip install --require-hashes --only-binary=:all: -r requirements.txt`. Wheels only, so no
  package build code runs.
- Tagged by a hash of the Dockerfile and requirements, and rebuilt only when those change.
- A task can override the image. The repo itself is still loaded via PYTHONPATH.
- The SWE-bench adapter, if built, would use its published images with the same run-time
  restrictions.
## 19. Development provider switched to Groq free tier, `openai/gpt-oss-120b` (revised)

**Decision.** The default provider is now Groq with `openai/gpt-oss-120b`, and `.env.example`
ships a Groq profile. Anthropic remains fully supported: its profile is commented out in
`.env.example`, and switching back is two lines. This supersedes #8 as the default.

**Why.**
- **Cost.** There are no Anthropic API credits available right now. Development has to
  continue, and Groq's free tier costs nothing.
- **Model choice.** Groq's docs (checked 2026-09-25) list the free-tier models that support
  tool calling: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`,
  and `qwen/qwen3.8-27b`. `gpt-oss-120b` is the largest and a production model (131K
  context), so it's the strongest fit for code. MiniMax M2.7 and Llama 3.3 70B also support
  tools, but they aren't on the free-tier table. The model id comes from config
  (`REPAIR_LLM__MODEL`).
- **Parallel calls.** gpt-oss can't make parallel tool calls. The loop already handles one
  call per turn.
- The Groq-vs-Anthropic comparison stays a Phase 6 experiment.

## 20. Groq free-tier specifics: TPM sizing, error classes, $0.00 cost (Groq provider)

**Decision.**
- **TPM sizing.** The free tier charges each request its prompt *plus the declared
  `max_completion_tokens`* against a limit of 8K tokens per minute. A request over it gets
  413, which backing off can't fix. The provider estimates the prompt size (JSON length ÷
  3.2) and shrinks `max_completion_tokens` to fit `groq_tpm_limit`. If less than
  `min_output_tokens` would be left, it raises a non-retryable 413 without sending. The loop
  then trims old tool results once, down to the most recent turn, and retries.
- **Error classes.**
  - 429s are retried, waiting at least the server's `retry-after`.
  - A 413 that mentions the per-minute window is retried after 20 s; a plain 413 is not.
  - A 400 `tool_use_failed` or `output_parse_failed` is a bad sample, so it's retried.
- **No prompt caching.** Nothing cache-related is sent to Groq. Any `cached_tokens` it
  reports are split out as cache reads, so input + cache + output still equals the billed
  total and the token budget stays accurate.
- **Cost.** It's recorded as `cost_usd = 0.0` with a `cost_note`, and `list_price_usd` keeps
  the on-demand price equivalent ($0.15 / $0.60 per 1M tokens for gpt-oss-120b). A model with
  no price entry still records `None` with a note.
- **Groq profile tuning.** Smaller tool outputs (6K chars, 200-line pages), elision at 4K
  prompt tokens, 8 retries, a 30-minute wall clock, and a 150K per-task token cap (the daily
  free quota is about 200K).

**Found in the first real run.** The model's final turn failed with 400 `output_parse_failed`.
Only `tool_use_failed` was treated as retryable at the time, so the loop stopped with
`llm_error` after the fix was already in and verified. That code is now retried too, with a
test.
## 21. `llm.tool_choice`: "required" on Groq, "auto" on Anthropic

**Decision.** `REPAIR_LLM__TOOL_CHOICE` is `auto` or `required`. When unset, each provider
uses its own default: `required` for Groq and `auto` for Anthropic.
- **Groq** receives OpenAI-style `tool_choice: "required"`.
- **Anthropic** receives `{"type": "any"}`.
- It is only sent when tools are present; `ping` sends no tools. The value actually used is
  recorded as `tool_choice` in `AgentResult`.

**Why.**
- **Groq.** The first real Groq run fixed the bug, then wrote its summary as plain text
  instead of calling `finish`; that turn ended in `output_parse_failed`, as recorded in
  RESULTS.md. With `required`, every turn is a tool call, so the only way to stop is
  `finish`, which is what grading and the `success` metric expect.
- **Anthropic.** Current Claude models reject forced tool use while thinking is on, and some
  (Opus 5.5, Fable 5.1) reject `any` altogether. Claude also reliably calls `finish` with
  `auto`.

**Effect on the no-action logic.** The nudge-then-`no_action` path stays, as a safety net for
a provider that returns a text-only turn anyway. It no longer shapes normal Groq behavior. The
realistic failure mode under `required` is a model that keeps calling tools without
finishing. That's bounded by `max_iterations`, the token budget, and the wall clock, and the
eval breaks it down by stop reason.
## 22. Benchmark design: 27 seeded-bug tasks over 4 small repos (Phase 4)

**Decision.**
- **Repos.** Four small repos with **correct code checked in**: `calc`, `textkit`,
  `inventory` (stdlib) and `schedule`, which needs `python-dateutil` and `six`.
- **Tasks.** Each task is a YAML file with `bug_type`, `difficulty`, a symptom-only issue,
  a `seed_patch` that plants one bug, and **measured** FAIL_TO_PASS / PASS_TO_PASS lists.
- **Mix.** Off-by-one 6, wrong-conditional 5, missing-edge-case 6, API-misuse 7, multi-file
  3. Easy 12, medium 11, hard 4.
- **How the patches are made.** Seed patches are generated as exact diffs of string
  replacements against the correct code, never hand-typed.

**Alternatives.** Hand-written buggy repos, or real historical bugs (SWE-bench style).

**Why.**
- Checking in the correct code means the reference solution is the checked-in code itself,
  so "the fix passes" can be proven mechanically.
- Seeded patches keep every task independent: one repo, many bugs.
- The test lists are *measured* by running the suite before and after the patch, not guessed.
  That caught real propagation: the `mean` bug also breaks four `variance` tests, so
  calc-mean-001 has 6 FAIL_TO_PASS tests, not 2.

**Limits.** The tasks are small and synthetic, and the agent sees each repo in full. These
numbers measure the loop and tools on self-contained bugs; they are not comparable to
SWE-bench.

## 23. Grading hardening: restore config, delete collection hooks, grade only graded ids (Phase 4)

**Decision.** Before grading:
1. Baseline test files and test-config files (`conftest.py`, `pytest.ini`, `tox.ini`,
   `setup.cfg`, `pyproject.toml`) are restored.
2. Newly added `conftest.py`, pytest/tox/setup config, `sitecustomize.py`,
   `usercustomize.py` and `*.pth` files are deleted and listed in `removed_files`.
3. pytest runs **only the FAIL_TO_PASS + PASS_TO_PASS node ids**.

**Why.** Restoring edited tests (#16) wasn't enough. A new root `conftest.py`, a
`sitecustomize.py` on `PYTHONPATH`, or a new test module that monkeypatches the code at
import time could all make the graded tests pass without fixing anything. Grading only the
graded ids means new test modules are never imported, so they can stay. All three cheats are
real-Docker tests: in each one the agent's own test run passes, and grading says
`resolved=False`.

**Recording and tests (extended).**
- The result JSON has three separate lists:
  - `modified_test_files`: original tests the agent edited, restored before grading;
  - `restored_config_files`: original `conftest.py`, `pytest.ini`, `pyproject.toml`,
    `setup.cfg` or `tox.ini` the agent edited, restored before grading;
  - `removed_files`: new conftests, config files, start-up hooks and `.pth` files the agent
    added, deleted before grading.
- New test files the agent adds stay but are never collected; new conftests never stay.
- A non-Docker unit test has the scripted agent "fix" calc-mean-001 with a new root
  `conftest.py` that replaces `calc.stats.mean`. Its own test run passes. Grading deletes
  the conftest, records it in `removed_files`, and returns `resolved=False` with
  `agent_disagrees_with_grading=True`.
- Another test edits an original `pytest.ini`: it is restored and listed in
  `restored_config_files`.

## 24. Benchmark validation gate (Phase 4)

**Decision.** `repair-agent benchmark validate` proves, for every task, with no LLM:
- the correct code passes its suite, the same way on 2 runs;
- the seed patch applies and touches no test files;
- FAIL_TO_PASS and PASS_TO_PASS exactly match what the patch changes;
- `grade()` says the buggy code isn't resolved and the reverted patch is;
- the issue has no leaks: no patched file name, no changed line of code, no cause hints.

Results are cached by a content hash of the task file, repo tree and image, and `eval`
refuses to start on an invalid or unvalidated task. A Docker test re-validates all 27 tasks
in the project's own suite.

**Why.** A broken task silently corrupts every metric computed from it. On first run the gate
caught three corrupt seed patches (a YAML-stripped trailing blank line, now repaired from the
hunk counts) and one leak-check false positive. After those fixes, all 27 tasks are valid.

## 25. Eval runner: resumable, round-ordered, stops on infrastructure failures (Phase 4)

**Decision.**
- **Layout.** `runs/eval-<id>/manifest.json` records the config snapshot, git sha and dirty
  flag, task hashes, and N. Each attempt gets `<task>/run-<k>/{trace.jsonl,result.json}`.
- **Resume.** Attempts that already have `result.json` are skipped. An interrupted trace is
  set aside as `trace.partial-*`. `--resume` refuses a changed config or task set unless
  `--force` is given.
- **Order.** Runs go in rounds: every task's run 1, then run 2, and so on.
- **Concurrency.** Sequential by default; `--parallel N` is available.
- **Infrastructure failures.** An attempt lost to infrastructure (an LLM error other than
  400, meaning quota, rate limit, outage or auth; or a sandbox failure) is saved as
  `result.infra.json`, excluded from metrics, and re-run on resume. The eval **stops** at
  that point.

**Why.** Groq's free tier caps `gpt-oss-120b` at about 200K tokens/day, so a full 81-attempt
eval will be interrupted many times. Stopping on quota exhaustion, instead of recording
dozens of fake failures, keeps the numbers honest. Round order means a partial eval still
covers every task.

## 26. Metric definitions (Phase 4)

**Decision.**
- **Resolve rate.** Resolved over valid attempts, with a 95% Wilson interval, which stays
  meaningful at small n.
- **pass@k.** The unbiased estimator 1 − C(n−c, k)/C(n, k), averaged over tasks, counting
  only tasks with n ≥ k.
- **Variance.** Per-run resolve rates with mean ± std, and a list of flaky tasks
  (0 < c < n).
- **Failure mode.** Each unresolved attempt gets the first match in this order:
  `broke_other_tests`, `timeout`, `budget_exceeded`, `gave_up` (no source change, or a
  no-action/refusal stop), `llm_error`, `wrong_fix`.
- **Cost.** Reported both as charged ($0.00 on the free tier) and at list-price equivalent,
  per resolved task.
- **Tamper attempts.** Counted separately.

**Why.** Every number can be recomputed with `repair-agent report <id>` from the saved
result files. None comes from logs or memory.
## 27. Text-only answers under `tool_choice=required` are handled, not resampled (Phase 4)

**Decision.** Groq rejects a text-only turn under `tool_choice=required` with 400
`tool_use_failed` ("did not call a tool"). This is now its own error kind, `no_tool_call`.
It is not retried. The model's text, which Groq returns as `failed_generation`, is kept as
an assistant turn, followed by a nudge: call `finish` if done, otherwise continue. A second
such turn in a row stops the loop with `no_action`.

**Why.** The first real Phase 4 check showed `gpt-oss-120b` finishing correct fixes with a
text summary. The retry layer re-sent the identical prompt up to 8 times, got the same
answer each time, burned rate-limit waits and quota, and recorded `llm_error`. After the
fix, every attempt in that check that resolved also ended cleanly with `finish`.

## 28. Host sleep is prevented during evals, and detected afterwards (Phase 4)

**Decision.**
- **Prevention.** `repair-agent eval` holds `SetThreadExecutionState(ES_SYSTEM_REQUIRED)` on
  Windows while it runs. On other platforms this is a no-op.
- **Detection.** An attempt whose wall time exceeds its budget plus 600 s is classified as
  infrastructure: excluded from metrics, and moved aside and re-run on resume. The loop checks
  the budget before every call, so it can only overrun by about one request timeout plus
  grading.

**Why.** Two attempts in the first real check "timed out" after 3.0 h and 7.4 h, against a
30-minute budget. The Windows power log shows the machine went to sleep at the exact second
each request started and woke hours later. Scoring those as agent timeouts would have been
a false failure mode.
## 29. Token budget excludes cache reads; optional cost cap

**Decision.**
- `max_tokens_per_task` now counts **uncached input + cache writes + output**
  (`Usage.budget_tokens`). Cache reads are excluded.
- A new optional `max_cost_usd_per_task` caps the list-price cost estimate, using the
  cache-read and cache-write multipliers from #6. It applies on free tiers too, where the
  charge is $0 but the list price isn't. When it trips, the stop reason is `cost_budget`,
  which counts as `budget_exceeded` in failure modes.
- The per-request elision threshold is unchanged: it still looks at the full prompt size.
- `AgentResult` still reports all four token counts: input, output, cache read, cache write.

**Why.** Every turn re-sends the whole context. Counting cache reads meant summing the
context once per turn, so `token_budget` fired on long runs that were actually cheap.
Budgeting billable tokens, plus an explicit dollar cap, limits what a run really costs.

## 30. One source of truth for results

**Decision.**
- The **grading run** (after the loop, original tests and config restored, graded ids only)
  alone drives `outcome`, `success`, `resolved` and `final_tests`.
- The agent's own last test run is kept separately as `agent_last_test_result`, with its
  selectors, since it may have been targeted. `agent_disagrees_with_grading` flags when that
  run and the grading run disagree on pass or fail. Neither affects the outcome.
- The `AgentResult` docstring lists which run or component every field comes from.

**Why.** The agent's runs can be targeted, stale, or distorted by its own edits (for
example a new conftest). Using them for the outcome would let a cheat or a partial run
count as a pass. Recording the disagreement makes those cases visible instead.

## 31. Test budget: one finish-only turn before stopping (amends #13)

**Decision.** A `run_tests` request past `max_test_runs` gets an error result and the model
gets **one final turn** where only `finish` is accepted.
- **`finish` is called:** the run ends normally as `finished`, and grading still decides
  `resolved`.
- **Any other tool call:** it is rejected without running, and the loop stops with
  `test_budget`.
- **A text-only reply (or a rejected text answer):** the loop stops with `test_budget`.

**Why.** Running out of test runs usually happens right after the fix is verified.
Stopping immediately threw away a finished attempt's summary and turned would-be successes
into budget failures.

## 32. Minor: elision stubs keep edit locations; image hash includes Python version

**Decision.**
- **Elision stubs.** An elided `edit_file` result keeps its path and line range, for
  example `edit_file src/calc/stats.py lines 3-9`, or `(created, 12 lines)` for new files.
  The line range is read from the tool's own output.
- **Dependency-image tag.** The tag now also hashes the base image's `PYTHON_VERSION`
  (read from the image's environment). `--only-binary` wheels are resolved for the
  interpreter they'll run on, and a base-image Python upgrade forces a rebuild. The
  validation cache key includes it too.
## 33. Baseline runs on Groq's Developer tier with default limits

**Decision.**
- **Model.** The 27 × 3 baseline uses Groq `openai/gpt-oss-120b` on the Developer tier
  (250K tokens/min, 1K requests/min).
- **Free-tier workarounds removed.** Max output tokens go back to 16,000, and the TPM clamp
  is 250,000 instead of 8,000. The context and tool limits are reset to their defaults:
  elision at 60K tokens keeping 4 turns, 12,000-char tool output, 400-line pages, 100
  search matches.
- **Cost.** `groq_free_tier=false`, so `cost_usd` is what's charged, and there's a
  `max_cost_usd_per_task` safety cap of $0.05.

The eval manifest records all of this:
- the git SHA and dirty flag;
- the effective config snapshot: model, `tool_choice`, effort, temperature, output tokens,
  Groq TPM and tier, retries, timeout, price, budgets, and agent, tool and sandbox limits;
- an environment block: sandbox image, base Python version, per-repo dependency images,
  and Docker version.

The report adds the model ids the API actually returned.

**Why.** The free-tier limits existed only to survive 8K tokens per minute. Leaving them in
would handicap the agent and make the baseline a measure of truncation. Every
free-tier-affected run so far was a smoke test, so nothing reported needs to be
re-labelled.
## 34. The baseline uses the free-tier profile; later experiments must use the same profile (supersedes #33's baseline choice)

**Decision.**
- **Profile.** The 27-task baseline runs on Groq's **free tier** with `openai/gpt-oss-120b`,
  using the free-tier profile in `.env.example`:

  | Group | Settings |
  |---|---|
  | Model | `tool_choice=required`, `effort=medium`, `MAX_OUTPUT_TOKENS=4096`, `GROQ_TPM_LIMIT=8000`, `GROQ_FREE_TIER=true`, `MAX_RETRIES=8`, `MAX_RETRY_WAIT_S=120` |
  | Context | elision at 4,000 tokens, keeping 2 turns |
  | Tool output | 6,000 chars, 200-line pages, 50 search matches |
  | Budgets | 30 iterations, 10 test runs, 150K billable tokens, 1,800 s, $0.05 list-price cap |

  The Developer-tier profile from #33 stays documented but is **not** the baseline.
- **Runs.** One run per task (`--runs 1`), resumed daily. Because of that, pass@3 and
  across-run variance are not available for the baseline.
- **Comparability rule.** Every later experiment that claims an improvement over the
  baseline must use this exact profile, with only the variable under test changed. That
  covers retrieval and reflection in Phase 6, and model or provider comparisons.
  - The eval manifest records the full config snapshot, so a mismatch is visible.
  - `--resume` refuses a changed config, so one eval never mixes profiles.
  - A profile change (for example the Developer tier) needs its own baseline row.
- **Daily-limit handling.** A retry that would wait longer than `max_retry_wait_s` (120 s)
  gives up instead of sleeping.
  - The free tier's per-minute 429s wait 1–25 s and are unaffected.
  - The daily-limit 429 has a retry-after of many minutes. It ends the attempt as an LLM
    error with status 429, which the runner classifies as infrastructure: the attempt is
    excluded, the eval stops, and that attempt re-runs on resume.

**Why.**
- The Developer-tier upgrade isn't happening. Running the baseline on the profile that will
  actually be available keeps it reproducible and cheap ($0).
- Fixing the profile makes Phase 6 deltas attributable to the change under test, not to
  different limits.
- Without the retry-wait cap, a daily-limit wait could consume an attempt's 30-minute
  budget and be mis-scored as an agent timeout.

**Costs of this choice.** The small context and tool-output limits may lower the resolve
rate compared with the defaults. That is part of what this baseline measures, and it will be
stated next to the numbers.
## 35. Provider pool: the same model on several free backends (amends #34's runtime, not its profile)

**Decision.**
- **What it is.** `REPAIR_LLM__PROVIDER=pool` puts several endpoints serving the **same
  model** behind one provider. The default pool is `openai/gpt-oss-120b` on Groq's free tier,
  then on NVIDIA's free API catalog (`integrate.api.nvidia.com/v1`, OpenAI-compatible).
  More backends (OpenRouter, Cerebras, a local vLLM) are one JSON entry each in
  `REPAIR_LLM__POOL`, with their key named by `api_key_env`.
- **Routing.** `round_robin` (default) rotates the first backend on every request to spread
  per-minute limits; `failover` always prefers the first.
- **Benching.**
  - Benched until the cooldown ends:
    - a quota-exhausted backend (429 with a retry-after above `max_retry_wait_s`, or a
      daily/credit message; or 402), for the retry-after, else `pool_quota_cooldown_s`
      (1 h);
    - a transient failure (5xx, connection, per-minute 429), briefly
      (`pool_transient_cooldown_s`, 30 s).
  - Benched for the rest of the process: a rejected key (401/403) or a model the host
    doesn't serve (404).
  - In every case the same request goes to the next backend at once, with the caller's
    remaining time.
- **Not failed over.** 400s and "no tool call" are about the request, and another host of
  the same model would answer the same way. A non-retryable 413 (over one backend's
  per-minute cap) *is* tried on the next backend. If every backend says 413, the loop's
  context trimming takes over as before.
- **Pool exhausted.** When every backend is benched, the pool raises a retryable 429 whose
  retry-after is the earliest reopening. As with a single provider, a long wait ends the
  attempt as infrastructure, and the eval stops and resumes later.
- **Profile.** The NVIDIA backend gets the same 8,000-token request shaping as Groq, although
  NVIDIA has no such per-minute cap. Every request is built identically whichever backend
  serves it, so a pool eval runs the #34 profile. Only the host varies.
- **Recording.**
  - Every `llm_call` trace event names the backend that answered, plus any backends that
    failed first and why.
  - `AgentResult.backend_usage` has requests, failovers and tokens per backend.
  - Cost is charged per backend; free-tier backends are $0.00 with the list-price
    equivalent kept.
  - The manifest records the strategy and each backend with a key (endpoint, model id,
    clamp, tool_choice), so adding a backend is a config change that `--resume` refuses
    without `--force`. Single-provider snapshots are unchanged, and existing evals still
    resume.
  - The report gets a "Backends" table: request share, failovers, tokens, and the resolve
    rate of attempts answered **entirely** by one backend.
- **`ping`** checks each backend separately *with a tool*, because the agent needs tool
  calling and `tool_choice=required`.

**Why.** On Groq's free tier alone, the baseline gets 6–10 attempts a day, so 27 attempts
take about 5 days of resumes. The same weights on a second free host roughly double the
daily token budget and halve per-minute waiting. The comparability rule in #34 is about
what the agent sees: model, sampling, limits. A pool changes none of that.

**Risks and how they're handled.**
- **Hosts can differ** in quantization, chat template, or tool-call parsing, even with the
  same weights. So a pool eval is its own eval id and its own RESULTS row, not a resume of
  the Groq-only eval. The per-backend "sole attempts resolved" column shows whether one host
  does worse.
- **Unverified live.** At the time of writing there was no NVIDIA key. The NVIDIA path is
  covered by unit tests against the real `openai` SDK's error types (429 with retry-after,
  401, connection errors). Still unverified:
  - whether NVIDIA's `openai/gpt-oss-120b` accepts `tool_choice: "required"` and
    `reasoning_effort`;
  - its exact error bodies.

  `repair-agent ping` with the pool checks the first two before an eval. Each backend can
  turn either off (`tool_choice`, `reasoning_effort`) without code changes.
- **One account per provider.** A pool is for different providers. Several free accounts at
  one provider to multiply its quota would break most providers' terms, and it isn't
  supported as a default.

**Alternatives.**
- A different free model (e.g. GLM or DeepSeek on NVIDIA). It would be faster to get, but it
  is a different model, so it's a separate baseline. It can be done later with
  `REPAIR_LLM__MODEL` and a single-backend pool.
- Hermes Agent (Nous Research) was considered as a component and rejected. It's a
  complete personal-assistant agent, so using it would replace the loop, sandbox and grading
  this project measures. A Hermes *model* on an OpenAI-compatible host could still be a pool
  backend or a comparison model.
