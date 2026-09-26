# repair-agent

An autonomous agent that takes a GitHub issue, explores the repository, writes a fix, runs
the tests inside a Docker sandbox, iterates on failures, and opens a pull request. It comes
with an evaluation harness that measures how well the agent actually works.

> **Status: Phase 4 of 6 (benchmark + eval).** A validated 27-task benchmark, tamper-proof
> grading, a resumable `repair-agent eval` runner, and a metrics report. Development runs on
> **Groq's free tier** (`openai/gpt-oss-120b`); Anthropic is still supported. Baseline
> numbers are in [RESULTS.md](RESULTS.md) once a full eval has run.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker (Docker Desktop with WSL2 on
Windows), and [ripgrep](https://github.com/BurntSushi/ripgrep) on `PATH`.

```bash
uv sync
cp .env.example .env              # then add GROQ_API_KEY (or ANTHROPIC_API_KEY)
uv run repair-agent sandbox build # one-time: builds the sandbox image
uv run pytest                     # Docker tests are skipped if the daemon is down
```

## Usage

```bash
uv run repair-agent config           # show effective config (secrets masked)
uv run repair-agent ping             # one tiny LLM request to check credentials
uv run repair-agent sandbox build    # (re)build the sandbox image
uv run repair-agent sandbox cleanup  # remove containers left by interrupted runs
uv run repair-agent solve --task benchmark/tasks/calc-mean-001.yaml [--keep-workspace]
uv run repair-agent benchmark validate                 # prove every task is well-formed (no LLM)
uv run repair-agent eval --runs 3                      # all tasks x 3 runs -> runs/eval-<id>/report.md
uv run repair-agent eval --resume eval-<id> --runs 3   # continue after a crash or quota stop
uv run repair-agent report eval-<id> [--append-results]
```

### Configuration

All settings come from environment variables or `.env` (see `.env.example`). Non-secret
settings use the `REPAIR_` prefix with `__` for nesting, e.g. `REPAIR_LLM__MODEL`,
`REPAIR_SANDBOX__MEMORY_MB`. Secrets use their standard names (`ANTHROPIC_API_KEY`,
`GROQ_API_KEY`, `GITHUB_TOKEN`), are held as `SecretStr`, and are scrubbed from traces.

## Architecture

```mermaid
flowchart LR
    CLI[cli.py solve / eval] --> Loop[agent/loop.py]
    Loop -->|neutral Message / ToolSpec| LLM[llm/base.py]
    LLM --> A[llm/anthropic.py]
    LLM --> G[llm/groq.py]
    Loop --> Reg[tools/ ToolRegistry]
    Reg --> FT[list / read / edit]
    Reg --> S[search_code: ripgrep]
    Reg --> RT[run_tests]
    FT & S --> WS[(host workspace: git clone)]
    RT --> SB[sandbox/ DockerSandbox]
    WS -->|tar snapshot| SB
    SB --> C[[fresh container: no network, uid 1000, CPU/mem/PID limits]]
    Loop --> Trace[tracing.py]
    Loop --> Grade[agent/grading.py: restore tests + config, drop hooks, run graded ids]
    Grade --> SB
    Trace --> Runs[(runs/run_id/task_id.jsonl)]
    Eval[eval/runner.py] --> Loop
    Val[eval/validate.py] --> SB
    Eval --> Rep[eval/metrics.py + report.py]
    Runs --> Rep
```

Solid arrows are built; dotted ones come in later phases.

### Sandbox

Repo code runs **only** inside Docker. The agent's working copy is a git clone on the host.
File tools read and edit it directly and never execute anything. Each `run_tests` call:

1. packs the workspace into a tar (POSIX names, uid 1000, bytes untouched);
2. creates a fresh container with `network_mode=none`, a non-root user, `cap_drop=ALL`,
   `no-new-privileges`, and limits on CPU, memory (no swap), and PIDs;
3. copies the tar in, runs pytest with a host-enforced timeout, and reads JUnit XML for
   per-test outcomes;
4. always removes the container.

There are no bind mounts, so Windows paths never reach Docker, and line endings are preserved
byte for byte.

### Tools

| Tool | Returns |
|---|---|
| `list_files(path=".", depth=2)` | Indented tree, dirs end with `/`, capped at 500 entries |
| `read_file(path, start_line=1, end_line=None)` | `cat -n`-style lines, paged at 400 with a "call again with start_line=N" hint |
| `search_code(pattern, glob=None)` | `path:line:text`, capped at 100 matches; no match is not an error |
| `edit_file(path, old_str, new_str)` | Exact match, which must be unique; shows the edited region. `old_str=""` creates a new file |
| `run_tests(test_selector=None)` | `Result: PASSED/FAILED/TIMEOUT/...` line, failing ids, pytest output |
| `finish(summary)` | Signals completion to the loop |

Expected failures come back to the model as `is_error` results with an actionable message;
the registry never raises. Outputs over 12,000 chars keep head and tail with an explicit
truncation note.

| Module | Responsibility | Phase |
|---|---|---|
| `config.py` | Pydantic settings: LLM, budgets, tools, sandbox, GitHub allowlist, pricing | 1 ✅ |
| `llm/` | Provider-agnostic interface; Anthropic and Groq providers; cost estimation | 1 ✅ (Groq: 3 ✅) |
| `tracing.py` | Append-only JSONL trace per task, flushed per event, with secret redaction | 1 ✅ |
| `sandbox/` | Host workspace (git), Docker sandbox, JUnit parsing | 2 ✅ |
| `tools/` | Six agent tools + registry (validation, errors, truncation) | 2 ✅ |
| `cli.py` | Typer CLI: config, ping, sandbox, solve, benchmark validate, eval, report | 4 ✅ |
| `agent/` | ReAct loop, budgets, retries, caching, context elision, grading | 3 ✅ |
| `eval/`, `benchmark/` | 27 tasks, validation gate, resumable runner, metrics, report | 4 ✅ |
| `github/`, `dashboard/` | Issue → PR flow, FastAPI + React dashboard | 5 |

### Agent loop

Each iteration is one model call, then all of its tool calls. The results go back as a
single message ending with a budget line, for example
`[budget] iteration 7/30 · test runs 2/10 · tokens 41,200/500,000 · time 1m12s/15m00s`.

- **Stop reasons:** `finished`, `max_iterations`, `token_budget`, `test_budget`, `timeout`,
  `refusal`, `no_action`, `llm_error`, `sandbox_error`.
- **Retries:** retryable API errors are retried with exponential backoff and jitter, and
  the wait honors `retry-after`.
- **Caching:** prompt caching covers the tools, system prompt, and history. Old tool results
  are replaced with stubs once the context passes 60K tokens.
- **Grading:** after the loop, grading runs with protections against test tampering:
  - original test and test-config files are restored;
  - newly added collection hooks (`conftest.py`, pytest config, `sitecustomize.py`, `*.pth`)
    are deleted;
  - only the FAIL_TO_PASS + PASS_TO_PASS tests run.

  `resolved` requires all of them to pass, so editing tests or monkeypatching from a new
  file can't count as a fix.

Each attempt writes `runs/<run_id>/<task_id>.jsonl` (the trace) and
`runs/<run_id>/<task_id>.result.json`, which records:
- outcome, stop reason, and resolved
- iterations, test runs, and tool calls
- tokens (including cache), estimated cost, and timings
- the model id returned by the API and the temperature sent
- test files the agent edited
- final per-test outcomes and the diff

Design decisions and the alternatives considered are in [DECISIONS.md](DECISIONS.md).

### Providers

| | Groq (default) | Anthropic |
|---|---|---|
| Model | `openai/gpt-oss-120b` | `claude-sonnet-5` |
| Tool calls | one per turn (gpt-oss has no parallel calls) | parallel |
| Prompt caching | not sent (Groq may cache automatically; cached tokens are recorded) | explicit markers on the system prompt and history |
| Limits handled | 8K tokens/min: `max_completion_tokens` sized to fit, 429 `retry-after`, 413, unparseable-output 400s | 429 / 5xx / `retry-after` |
| Cost recorded | $0.00 on the free tier, plus the list-price equivalent | list price |

Switch providers with `REPAIR_LLM__PROVIDER` and `REPAIR_LLM__MODEL`; both profiles are in
`.env.example`.

## Benchmark

27 seeded-bug tasks across 4 small repos. `schedule` depends on `python-dateutil`, which is
baked into its own sandbox image at build time; test containers still have no network.

| Repo | Tasks | Bug types |
|---|---|---|
| `calc` | 6 | off-by-one, wrong conditional, missing edge case, API misuse |
| `textkit` | 7 | off-by-one, wrong conditional, missing edge case, API misuse, multi-file |
| `inventory` | 7 | wrong conditional, missing edge case, API misuse, multi-file |
| `schedule` | 7 | off-by-one, wrong conditional, missing edge case, API misuse, multi-file |

Difficulty: 12 easy, 11 medium, 4 hard.

Every task passes `repair-agent benchmark validate`, which proves each of the following
before any agent runs:
- the correct code passes its suite;
- the seed patch fails exactly FAIL_TO_PASS and keeps PASS_TO_PASS passing;
- grading marks the buggy code unresolved and the correct code resolved;
- the issue describes symptoms only.

### Metrics

`report.md` includes:
- **rates:** resolve rate with a 95% Wilson CI, pass@1, pass@3 (unbiased estimator),
  success rate, and graded-test pass rate;
- **variance:** per-run resolve rates (mean ± std) and flaky tasks;
- **effort and cost:** iterations, test runs, tokens, cost charged and at list price, and
  latency p50/p95;
- **failure modes:** broke other tests, timeout, budget exceeded, gave up, LLM error, wrong
  fix;
- **breakdowns:** by bug type, difficulty and repo, plus a per-task ✓/✗ table.

Attempts lost to infrastructure (quota, outages, Docker) are excluded and re-run on resume.

## Results

One smoke-test run so far: Groq `gpt-oss-120b` resolved `calc-mean-001` for $0.00 on the free
tier, but didn't finish cleanly. Benchmark numbers go into [RESULTS.md](RESULTS.md), and only
from real `eval` reports.
