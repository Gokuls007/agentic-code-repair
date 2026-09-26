from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from repair_agent.cli import app
from repair_agent.config import ModelPrice
from repair_agent.llm.base import LLMError, Usage
from repair_agent.llm.pricing import estimate_cost

PRICING = {"m": ModelPrice(input_per_mtok=2.0, output_per_mtok=10.0)}


def test_estimate_cost_includes_cache_multipliers() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    # 2.00 input + 1.00 output + 0.20 cache read + 2.50 cache write
    assert estimate_cost(usage, "m", PRICING) == pytest.approx(5.70)


def test_estimate_cost_unknown_model_is_none() -> None:
    assert estimate_cost(Usage(input_tokens=10), "unknown", PRICING) is None


def test_usage_addition() -> None:
    total = Usage(input_tokens=1, output_tokens=2) + Usage(
        input_tokens=3, cache_read_input_tokens=4
    )
    assert total == Usage(input_tokens=4, output_tokens=2, cache_read_input_tokens=4)
    assert total.total_tokens == 10


runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_cli_config_masks_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-print-123456")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "sk-ant-should-not-print" not in result.stdout
    data = json.loads(result.stdout)
    assert data["anthropic_api_key"] == "set"
    assert data["github_token"] == "unset"


def test_cli_ping_without_key_fails_cleanly() -> None:
    result = runner.invoke(app, ["ping"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY is not set" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "authentication failed: check ANTHROPIC_API_KEY"),
        (404, "model not found"),
        (None, "could not reach the API"),
    ],
)
def test_cli_ping_reports_api_errors_in_one_line(
    monkeypatch: pytest.MonkeyPatch, status: int | None, expected: str
) -> None:
    class FailingProvider:
        def complete(self, **_: object) -> None:
            raise LLMError("boom", retryable=False, status_code=status)

    monkeypatch.setattr("repair_agent.llm.create_provider", lambda _settings: FailingProvider())
    result = runner.invoke(app, ["ping"])
    assert result.exit_code == 1
    assert expected in result.output
    assert "Traceback" not in result.output


def test_cli_unimplemented_commands_exit_nonzero() -> None:
    assert runner.invoke(app, ["solve", "--task", "x.yaml"]).exit_code == 2
    assert runner.invoke(app, ["eval"]).exit_code == 2
