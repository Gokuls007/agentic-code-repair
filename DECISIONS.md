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

## 7. Default model `claude-opus-5`, no thinking/effort overrides by default (Phase 1)

**Decision.** The model comes from config (default `claude-opus-5`). The provider sends
`output_config.effort` only when `REPAIR_LLM__EFFORT` is set. It never sends a `thinking`
parameter, so the model's default adaptive thinking applies.

**Why.** It keeps the request minimal and valid across current models. Effort is the knob
worth sweeping in Phase 6. Refusals come back as `StopReason.REFUSAL`, so the loop can
record them as their own failure mode. Server-side refusal fallbacks are not enabled yet.
