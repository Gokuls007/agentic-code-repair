from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from repair_agent.config import DEFAULT_PRICING, Settings


def test_defaults_are_sane() -> None:
    s = Settings()
    assert s.llm.provider == "anthropic"
    assert s.llm.model == "claude-opus-5"
    assert s.budget.max_iterations > 0
    assert s.tools.max_output_chars > 0
    assert s.github.repo_allowlist == []
    assert s.anthropic_api_key is None
    assert s.pricing.keys() == DEFAULT_PRICING.keys()


def test_nested_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPAIR_LLM__MODEL", "claude-sonnet-5")
    monkeypatch.setenv("REPAIR_LLM__EFFORT", "high")
    monkeypatch.setenv("REPAIR_BUDGET__MAX_ITERATIONS", "7")
    monkeypatch.setenv("REPAIR_TOOLS__MAX_OUTPUT_CHARS", "500")
    monkeypatch.setenv("REPAIR_RUNS_DIR", "out/traces")
    s = Settings()
    assert s.llm.model == "claude-sonnet-5"
    assert s.llm.effort == "high"
    assert s.budget.max_iterations == 7
    assert s.tools.max_output_chars == 500
    assert s.runs_dir == Path("out/traces")


def test_loads_from_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-ant-from-dotenv-123456\nREPAIR_BUDGET__MAX_TEST_RUNS=3\n",
        encoding="utf-8",
    )
    s = Settings()  # cwd is tmp_path via the autouse fixture
    assert s.api_key_for("anthropic") == "sk-ant-from-dotenv-123456"
    assert s.budget.max_test_runs == 3


def test_secrets_never_appear_in_repr_or_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "sk-ant-supersecretvalue-abcdef"
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    s = Settings()
    assert key not in repr(s)
    assert key not in str(s.model_dump())
    assert key not in s.model_dump_json()
    assert s.secret_values() == [key]


def test_missing_api_key_gives_clear_error() -> None:
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        Settings().api_key_for("anthropic")


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("REPAIR_BUDGET__MAX_ITERATIONS", "0"),
        ("REPAIR_BUDGET__WALL_CLOCK_TIMEOUT_S", "-1"),
        ("REPAIR_LLM__PROVIDER", "openai"),
        ("REPAIR_LLM__EFFORT", "extreme"),
    ],
)
def test_invalid_values_rejected(monkeypatch: pytest.MonkeyPatch, env: str, value: str) -> None:
    monkeypatch.setenv(env, value)
    with pytest.raises(ValidationError):
        Settings()


def test_allowlist_parses_csv_and_matches_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REPAIR_GITHUB__REPO_ALLOWLIST", " me/repo-a , Me/Repo-B ,")
    gh = Settings().github
    assert gh.repo_allowlist == ["me/repo-a", "Me/Repo-B"]
    assert gh.is_allowed("ME/REPO-A")
    assert gh.is_allowed("me/repo-b")
    assert not gh.is_allowed("someone-else/repo-a")


@pytest.mark.parametrize("bad", ["justaname", "owner/", "/repo", "a/b/c"])
def test_allowlist_rejects_malformed_slugs(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("REPAIR_GITHUB__REPO_ALLOWLIST", bad)
    with pytest.raises(ValidationError):
        Settings()
