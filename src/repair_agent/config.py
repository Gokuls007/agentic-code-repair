"""Application settings, loaded from environment variables and an optional .env file.

Non-secret settings use the ``REPAIR_`` prefix with ``__`` as the nesting delimiter,
e.g. ``REPAIR_LLM__MODEL`` or ``REPAIR_BUDGET__MAX_ITERATIONS``. Secrets use their
conventional unprefixed names (``ANTHROPIC_API_KEY``, ``GROQ_API_KEY``, ``GITHUB_TOKEN``)
and are stored as ``SecretStr`` so they never appear in reprs, logs, or dumps.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ProviderName = Literal["anthropic", "groq"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ModelPrice(BaseModel):
    """USD price per million tokens for one model."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


# Anthropic list prices (USD / 1M tokens) as published at the time of writing.
# Override via REPAIR_PRICING='{"model-id": {"input_per_mtok": .., "output_per_mtok": ..}}'.
DEFAULT_PRICING: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-5-5": ModelPrice(input_per_mtok=4.0, output_per_mtok=20.0),
    "claude-sonnet-5": ModelPrice(input_per_mtok=2.0, output_per_mtok=10.0),
    "claude-haiku-4-5": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0),
}


class LLMSettings(BaseModel):
    """Which provider/model to call and per-request generation limits."""

    provider: ProviderName = "anthropic"
    model: str = "claude-sonnet-5"
    max_output_tokens: int = Field(default=16000, gt=0)
    effort: Effort | None = None
    request_timeout_s: float = Field(default=300.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class BudgetSettings(BaseModel):
    """Hard per-task limits enforced by the agent loop."""

    max_iterations: int = Field(default=30, gt=0)
    max_tokens_per_task: int = Field(default=500_000, gt=0)
    max_test_runs: int = Field(default=10, gt=0)
    wall_clock_timeout_s: float = Field(default=900.0, gt=0)


class ToolSettings(BaseModel):
    """Limits applied to tool outputs before they are shown to the model."""

    max_output_chars: int = Field(default=12_000, gt=0)
    max_read_lines: int = Field(default=400, gt=0)
    max_search_results: int = Field(default=100, gt=0)
    max_list_entries: int = Field(default=500, gt=0)
    search_timeout_s: float = Field(default=20.0, gt=0)


class SandboxSettings(BaseModel):
    """Docker sandbox limits. Repo code and tests only ever run under these."""

    image: str = "repair-agent-sandbox:py3.11"
    cpus: float = Field(default=1.0, gt=0)
    memory_mb: int = Field(default=1024, ge=64)
    pids_limit: int = Field(default=256, gt=0)
    test_timeout_s: float = Field(default=120.0, gt=0)
    max_log_bytes: int = Field(default=1_000_000, gt=0)


class GitHubSettings(BaseModel):
    """GitHub integration. PRs are only ever opened on allowlisted repos."""

    repo_allowlist: Annotated[list[str], NoDecode] = Field(default_factory=list)

    @field_validator("repo_allowlist", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept a comma-separated string (the natural .env format) as well as a list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("repo_allowlist")
    @classmethod
    def _validate_slugs(cls, value: list[str]) -> list[str]:
        for slug in value:
            owner, sep, name = slug.partition("/")
            if not sep or not owner or not name or "/" in name:
                raise ValueError(f"allowlist entries must be 'owner/repo', got {slug!r}")
        return value

    def is_allowed(self, repo: str) -> bool:
        """Return True if ``owner/repo`` is on the allowlist (case-insensitive)."""
        return repo.lower() in {slug.lower() for slug in self.repo_allowlist}


class Settings(BaseSettings):
    """Top-level settings object. Construct via :func:`get_settings` in application code."""

    model_config = SettingsConfigDict(
        env_prefix="REPAIR_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    github_token: SecretStr | None = Field(default=None, validation_alias="GITHUB_TOKEN")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    pricing: dict[str, ModelPrice] = Field(default_factory=lambda: dict(DEFAULT_PRICING))

    runs_dir: Path = Path("runs")

    def secret_values(self) -> list[str]:
        """All configured secret values, for redaction in logs and traces."""
        secrets = (self.anthropic_api_key, self.groq_api_key, self.github_token)
        return [s.get_secret_value() for s in secrets if s and s.get_secret_value()]

    def api_key_for(self, provider: ProviderName) -> str:
        """Return the API key for ``provider`` or raise a clear error if it is missing."""
        secret = {"anthropic": self.anthropic_api_key, "groq": self.groq_api_key}[provider]
        if secret is None or not secret.get_secret_value():
            env_name = f"{provider.upper()}_API_KEY"
            raise RuntimeError(f"{env_name} is not set; add it to .env")
        return secret.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
