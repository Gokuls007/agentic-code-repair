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
- `llm.temperature` defaults to `None`, meaning the parameter is not sent. Current Claude
  models such as Sonnet 5 reject sampling parameters with a 400.
- `AgentResult` records `temperature` as it was actually sent, plus both `model` (what was
  requested) and `model_id` (what the API reported).

**Why.** Reproducibility. The eval report can say exactly which model and sampling
settings produced each number.

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