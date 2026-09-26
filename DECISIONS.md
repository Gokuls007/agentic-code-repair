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