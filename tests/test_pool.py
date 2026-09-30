"""Provider pool: balancing, benching, failover, config, and a full agent run (no API cost)."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from fakes import FakeClock, FakeSandbox, RecordingSleep, ScriptedProvider, reply, tc
from repair_agent.agent import load_task, solve_task
from repair_agent.config import BackendSettings, LLMSettings, Settings, default_pool
from repair_agent.eval.metrics import backend_stats
from repair_agent.eval.runner import config_snapshot
from repair_agent.llm import create_backend, create_pool, pool_backends_available
from repair_agent.llm.base import NO_TOOL_CALL, LLMError, Message, TextBlock
from repair_agent.llm.openai_compat import OpenAICompatibleProvider
from repair_agent.llm.pool import POOL_EXHAUSTED, Backend, PoolProvider
from repair_agent.sandbox.workspace import Workspace
from repair_agent.tracing import EventKind, Tracer, read_trace

ROOT = Path(__file__).resolve().parents[1]
MSGS = [Message(role="user", content=[TextBlock(text="hi")])]


class Named(ScriptedProvider):
    def __init__(self, name: str, script, **kw):
        super().__init__(script, **kw)
        self.name = name


def ok(text: str = "ok"):
    return reply(tc("finish", summary=text))


def quota(retry_after: float | None = 3600.0, message: str = "rate limit") -> LLMError:
    return LLMError(message, retryable=True, status_code=429, retry_after_s=retry_after)


def pool(*backends: Named, strategy: str = "failover", clock=None, **llm) -> PoolProvider:
    settings = LLMSettings(pool_strategy=strategy, **llm)
    return PoolProvider(
        [Backend(name=b.name, provider=b) for b in backends], settings, clock=clock or FakeClock()
    )


def call(p: PoolProvider, timeout_s: float | None = None):
    return p.complete(system="s", messages=MSGS, tools=[], timeout_s=timeout_s)


# --- routing -----------------------------------------------------------------------


def test_failover_strategy_prefers_first_backend() -> None:
    a, b = Named("a", [ok()], repeat_last=True), Named("b", [ok()], repeat_last=True)
    p = pool(a, b)
    assert [call(p).provider for _ in range(3)] == ["a", "a", "a"]
    assert not b.requests


def test_round_robin_rotates_first_backend() -> None:
    a, b = Named("a", [ok()], repeat_last=True), Named("b", [ok()], repeat_last=True)
    p = pool(a, b, strategy="round_robin")
    assert [call(p).provider for _ in range(4)] == ["a", "b", "a", "b"]


# --- benching and failover -----------------------------------------------------------


def test_daily_quota_fails_over_and_benches_until_retry_after() -> None:
    clock = FakeClock()
    a = Named("a", [quota(retry_after=3600), ok("a again")])
    b = Named("b", [ok()], repeat_last=True)
    p = pool(a, b, clock=clock)

    first = call(p)
    assert first.provider == "b"
    assert first.fallbacks[0]["backend"] == "a" and first.fallbacks[0]["status"] == 429
    assert call(p).provider == "b" and len(a.requests) == 1  # a is benched, not retried
    status = {s["name"]: s for s in p.status()}
    assert status["a"]["bench_reason"] == "quota exhausted (429)"

    clock.now += 3601
    assert call(p).provider == "a"  # cooldown over: back to the preferred backend


def test_daily_limit_message_without_retry_after_uses_quota_cooldown() -> None:
    clock = FakeClock()
    a = Named("a", [quota(None, "Rate limit reached on tokens per day (TPD)")])
    b = Named("b", [ok()], repeat_last=True)
    p = pool(a, b, clock=clock, pool_quota_cooldown_s=900)
    call(p)
    clock.now += 899
    call(p)
    assert len(a.requests) == 1
    assert {s["name"]: s for s in p.status()}["a"]["benched_for_s"] == pytest.approx(1)


def test_per_minute_429_benches_briefly_then_backend_returns() -> None:
    clock = FakeClock()
    a = Named("a", [quota(retry_after=7), ok()])
    b = Named("b", [ok()], repeat_last=True)
    p = pool(a, b, clock=clock)
    assert call(p).provider == "b"
    clock.now += 8
    assert call(p).provider == "a"


@pytest.mark.parametrize("status", [401, 403, 404])
def test_bad_key_or_missing_model_benches_for_good(status) -> None:
    clock = FakeClock()
    a = Named("a", [LLMError("nope", retryable=False, status_code=status)])
    b = Named("b", [ok()], repeat_last=True)
    p = pool(a, b, clock=clock)
    call(p)
    clock.now += 10**9
    call(p)
    assert len(a.requests) == 1
    assert math.isinf({s["name"]: s for s in p.status()}["a"]["benched_for_s"])


def test_server_error_fails_over_immediately() -> None:
    a = Named("a", [LLMError("boom", retryable=True, status_code=503)])
    b = Named("b", [ok()])
    assert call(pool(a, b)).provider == "b"


@pytest.mark.parametrize(
    "error",
    [
        LLMError("bad sample", retryable=True, status_code=400),
        LLMError("text only", retryable=False, status_code=400, kind=NO_TOOL_CALL),
    ],
)
def test_errors_about_the_request_are_not_failed_over(error) -> None:
    a = Named("a", [error])
    b = Named("b", [ok()])
    with pytest.raises(LLMError) as info:
        call(pool(a, b))
    assert info.value is error
    assert not b.requests


def test_prompt_too_big_for_one_backend_goes_to_the_next_without_benching() -> None:
    too_big = LLMError("over TPM cap", retryable=False, status_code=413)
    a = Named("a", [too_big, ok()])
    b = Named("b", [ok()])
    p = pool(a, b)
    assert call(p).provider == "b"
    assert call(p).provider == "a"  # a was not benched


def test_too_big_everywhere_raises_the_413_so_the_loop_trims_context() -> None:
    err = lambda: LLMError("over TPM cap", retryable=False, status_code=413)  # noqa: E731
    with pytest.raises(LLMError) as info:
        call(pool(Named("a", [err()]), Named("b", [err()])))
    assert info.value.status_code == 413 and not info.value.retryable


def test_every_backend_benched_is_a_retryable_429_with_earliest_reopening() -> None:
    clock = FakeClock()
    a = Named("a", [quota(retry_after=500)])
    b = Named("b", [quota(retry_after=200)])
    p = pool(a, b, clock=clock)
    with pytest.raises(LLMError) as info:
        call(p)
    assert info.value.kind == POOL_EXHAUSTED
    assert info.value.retryable and info.value.status_code == 429
    assert info.value.retry_after_s == pytest.approx(200)
    clock.now += 50
    with pytest.raises(LLMError) as info:  # still benched: no request is sent
        call(p)
    assert info.value.retry_after_s == pytest.approx(150)
    assert len(a.requests) == len(b.requests) == 1


def test_every_key_rejected_is_not_retryable() -> None:
    bad = lambda: LLMError("no", retryable=False, status_code=401)  # noqa: E731
    with pytest.raises(LLMError) as info:
        call(pool(Named("a", [bad()]), Named("b", [bad()])))
    assert info.value.kind == POOL_EXHAUSTED and not info.value.retryable


def test_failover_passes_the_remaining_time_budget() -> None:
    clock = FakeClock(step=2.0)  # every clock read advances 2 s
    a = Named("a", [LLMError("boom", retryable=True, status_code=500)])
    b = Named("b", [ok()])
    call(pool(a, b, clock=clock), timeout_s=60)
    assert a.requests[0]["timeout_s"] > b.requests[0]["timeout_s"]
    assert b.requests[0]["timeout_s"] < 60


# --- config and construction -----------------------------------------------------------


def settings_with(tmp_path: Path, monkeypatch, **env: str) -> Settings:
    for name in ("GROQ_API_KEY", "NVIDIA_API_KEY", "CEREBRAS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)  # no stray .env
    return Settings(_env_file=None, llm=LLMSettings(provider="pool"))


def test_default_pool_is_groq_then_cerebras_for_gpt_oss() -> None:
    pool_ = default_pool()
    assert [(b.name, b.kind, b.base_url) for b in pool_] == [
        ("groq", "groq", None),
        ("cerebras", "openai_compatible", "https://api.cerebras.ai/v1"),
    ]
    model = LLMSettings().model
    assert model == "openai/gpt-oss-120b"
    # Cerebras names the same model without the org prefix.
    assert [b.model_for(model) for b in pool_] == ["openai/gpt-oss-120b", "gpt-oss-120b"]
    assert all(b.tpm_limit == 8000 for b in pool_)  # same request shaping everywhere


def test_pool_uses_only_backends_with_keys(tmp_path, monkeypatch) -> None:
    s = settings_with(tmp_path, monkeypatch, CEREBRAS_API_KEY="csk-key-123")
    usable, skipped = pool_backends_available(s)
    assert [b.name for b in usable] == ["cerebras"]
    assert skipped == ["groq (GROQ_API_KEY not set)"]
    provider = create_pool(s)
    assert [b.name for b in provider.backends] == ["cerebras"]
    assert isinstance(provider.backends[0].provider, OpenAICompatibleProvider)


def test_pool_without_any_key_is_a_clear_error(tmp_path, monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="no pool backend has an API key"):
        create_pool(settings_with(tmp_path, monkeypatch))


def test_backend_keys_are_redacted_and_read_from_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("NVIDIA_API_KEY=nv-from-dotenv\n", encoding="utf-8")
    s = Settings(llm=LLMSettings(provider="pool"))
    assert s.env_secret("NVIDIA_API_KEY") == "nv-from-dotenv"
    # Not a default pool backend any more, but a known provider key: still redacted.
    assert "nv-from-dotenv" in s.secret_values()


def test_pool_backend_key_with_custom_env_name_is_redacted(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MY_LOCAL_VLLM_KEY", "local-secret-9")
    backend = BackendSettings(
        name="local", base_url="http://localhost:8000/v1", api_key_env="MY_LOCAL_VLLM_KEY"
    )
    s = Settings(_env_file=None, llm=LLMSettings(provider="pool", pool=[backend]))
    assert "local-secret-9" in s.secret_values()


def test_pool_from_env_json(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REPAIR_LLM__PROVIDER", "pool")
    monkeypatch.setenv(
        "REPAIR_LLM__POOL",
        json.dumps(
            [
                {
                    "name": "nvidia",
                    "base_url": "https://integrate.api.nvidia.com/v1",
                    "api_key_env": "NVIDIA_API_KEY",
                    "tool_choice": "auto",
                },
                {
                    "name": "openrouter",
                    "base_url": "https://openrouter.ai/api/v1",
                    "model": "openai/gpt-oss-120b:free",
                    "api_key_env": "OPENROUTER_API_KEY",
                },
            ]
        ),
    )
    s = Settings(_env_file=None)
    assert [b.name for b in s.llm.pool] == ["nvidia", "openrouter"]
    assert s.llm.pool[1].model_for(s.llm.model) == "openai/gpt-oss-120b:free"


def test_duplicate_backend_names_rejected() -> None:
    b = BackendSettings(name="x", api_key_env="K", base_url="https://x.test/v1")
    with pytest.raises(ValueError, match="unique"):
        LLMSettings(pool=[b, b])


def test_backend_gets_its_own_model_tpm_and_tool_choice(tmp_path, monkeypatch) -> None:
    s = settings_with(tmp_path, monkeypatch, NVIDIA_API_KEY="k")
    backend = BackendSettings(
        name="nv",
        base_url="https://x.test/v1",
        api_key_env="NVIDIA_API_KEY",
        model="org/other-id",
        tpm_limit=None,
        reasoning_effort=False,
    )
    provider = create_backend(s, backend)
    params = provider.build_params(system="s", messages=MSGS, tools=[])
    assert provider.name == "nv"
    assert params["model"] == "org/other-id"
    assert "reasoning_effort" not in params


def test_manifest_snapshot_records_pool_members_only_for_pools(tmp_path, monkeypatch) -> None:
    s = settings_with(tmp_path, monkeypatch, CEREBRAS_API_KEY="k")
    snap = config_snapshot(s, "pool", backends=["cerebras"])
    assert snap["tool_choice"] == "required"
    assert [b["name"] for b in snap["pool"]["backends"]] == ["cerebras"]
    assert snap["pool"]["backends"][0]["model"] == "gpt-oss-120b"
    assert snap["pool"]["strategy"] == "round_robin"
    assert "pool" not in config_snapshot(s, "groq")  # old evals still resume


# --- OpenAI-compatible provider -----------------------------------------------------------


def _status_error(status: int, message: str, headers: dict | None = None) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return openai.APIStatusError(message, response=response, body={"error": {"message": message}})


class RaisingClient:
    def __init__(self, exc: Exception):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._raise))
        self._exc = exc

    def _raise(self, **_: object):
        raise self._exc


def compat(exc: Exception) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        LLMSettings(groq_tpm_limit=None), "k", RaisingClient(exc), name="nvidia"
    )


def test_openai_compatible_maps_rate_limit_with_retry_after() -> None:
    with pytest.raises(LLMError) as info:
        compat(_status_error(429, "Too Many Requests", {"retry-after": "7"})).complete(
            system="s", messages=MSGS, tools=[]
        )
    assert info.value.status_code == 429 and info.value.retryable
    assert info.value.retry_after_s == 7


def test_openai_compatible_maps_bad_key_and_connection_errors() -> None:
    with pytest.raises(LLMError) as info:
        compat(_status_error(401, "Unauthorized")).complete(system="s", messages=MSGS, tools=[])
    assert info.value.status_code == 401 and not info.value.retryable
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    with pytest.raises(LLMError) as info:
        compat(openai.APIConnectionError(request=request)).complete(
            system="s", messages=MSGS, tools=[]
        )
    assert info.value.retryable and info.value.status_code is None


def test_openai_compatible_requires_base_url() -> None:
    with pytest.raises(RuntimeError, match="base_url"):
        OpenAICompatibleProvider(LLMSettings(), "k", name="nv")


# --- a full attempt through the pool ---------------------------------------------------

BUG = "return sum(xs) / (len(xs) - 1)"
FIX = "return sum(xs) / len(xs)"


@pytest.fixture
def task():
    return load_task(ROOT / "benchmark" / "tasks" / "calc-mean-001.yaml")


@pytest.fixture
def ws(task, tmp_path: Path) -> Iterator[Workspace]:
    parent = tmp_path / "ws"
    parent.mkdir()
    workspace = Workspace.from_directory(task.repo_dir(), patch=task.seed_patch, parent_dir=parent)
    yield workspace
    workspace.cleanup()


def test_attempt_survives_mid_run_quota_and_records_backends(task, ws) -> None:
    groq = Named(
        "groq",
        [
            reply(tc("read_file", path="src/calc/stats.py")),
            quota(retry_after=86_400, message="tokens per day (TPD) exceeded"),
        ],
    )
    nvidia = Named(
        "nvidia",
        [
            reply(tc("edit_file", path="src/calc/stats.py", old_str=BUG, new_str=FIX)),
            reply(tc("run_tests", test_selector="tests/test_stats.py")),
            reply(tc("finish", summary="divide by len")),
        ],
    )
    settings = Settings(
        _env_file=None,
        llm=LLMSettings(provider="pool", model="openai/gpt-oss-120b", pool_strategy="failover"),
    )
    provider = PoolProvider(
        [Backend(name="groq", provider=groq), Backend(name="nvidia", provider=nvidia)],
        settings.llm,
        clock=FakeClock(),
    )
    with Tracer(ws.root.parent / "runs", "run1", task.id) as tracer:
        result = solve_task(
            task,
            ws,
            provider=provider,
            sandbox=FakeSandbox(),
            settings=settings,
            tracer=tracer,
            run_id="run1",
            clock=FakeClock(),
            sleep=RecordingSleep(),
        )

    assert result.resolved and result.provider == "pool" and result.tool_choice == "required"
    assert result.backend_usage["groq"].requests == 1
    assert result.backend_usage["groq"].failovers_from == 1
    assert result.backend_usage["nvidia"].requests == 3
    assert result.cost_usd == 0.0 and result.cost_note.startswith("free-tier backends only")

    calls = [e for e in read_trace(tracer.path) if e.kind == EventKind.LLM_CALL]
    assert [c.data["provider"] for c in calls] == ["groq", "nvidia", "nvidia", "nvidia"]
    assert calls[1].data["failed_over_from"][0]["backend"] == "groq"

    stats = backend_stats([result])
    assert stats["nvidia"].request_share == pytest.approx(0.75)
    assert stats["groq"].sole_attempts == 0 and stats["nvidia"].sole_attempts == 0


# --- fixes found in the first live Nemotron run ---------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["<function=read_file", "<function=read_file>", "functions.read_file", "read_file</function>"],
)
def test_wrapped_tool_names_are_unwrapped_when_they_match_a_tool(raw) -> None:
    from repair_agent.llm.base import ToolCall
    from repair_agent.llm.groq import _unwrap_tool_names

    msg = Message(role="assistant", content=[ToolCall(id="1", name=raw, arguments={})])
    _unwrap_tool_names(msg, {"read_file", "finish"})
    assert msg.tool_calls[0].name == "read_file"


@pytest.mark.parametrize("raw", ["<function=delete_repo", "read_files", "run shell"])
def test_unknown_tool_names_are_left_for_the_registry_to_reject(raw) -> None:
    from repair_agent.llm.base import ToolCall
    from repair_agent.llm.groq import _unwrap_tool_names

    msg = Message(role="assistant", content=[ToolCall(id="1", name=raw, arguments={})])
    _unwrap_tool_names(msg, {"read_file", "finish"})
    assert msg.tool_calls[0].name == raw


def test_lone_backend_server_errors_report_their_real_status() -> None:
    a = Named("a", [LLMError("Internal server error", retryable=True, status_code=500)])
    with pytest.raises(LLMError) as info:
        call(pool(a, clock=FakeClock()))
    assert info.value.status_code == 500 and info.value.retryable  # not a fake 429
    assert info.value.retry_after_s == pytest.approx(5.0)  # short transient cooldown


def test_exhausted_by_rate_limit_still_reports_429() -> None:
    with pytest.raises(LLMError) as info:
        call(pool(Named("a", [quota(retry_after=9)]), clock=FakeClock()))
    assert info.value.status_code == 429


def test_404_after_the_backend_has_answered_is_transient() -> None:
    clock = FakeClock()
    a = Named("a", [ok(), LLMError("Not Found", retryable=False, status_code=404), ok()])
    p = pool(a, clock=clock)
    assert call(p).provider == "a"
    with pytest.raises(LLMError) as info:
        call(p)
    assert info.value.retryable and info.value.status_code == 404
    clock.now += 6
    assert call(p).provider == "a"  # back after the short cooldown, not benched for good
