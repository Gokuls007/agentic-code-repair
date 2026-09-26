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
ToolChoice = Literal["auto", "required"]

DEFAULT_TOOL_CHOICE: dict[str, ToolChoice] = {"groq": "required", "anthropic": "auto"}


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
    # Groq on-demand list prices (console.groq.com/docs/models, checked 2026-09-25). On the
    # free tier nothing is charged; these give a list-price equivalent for comparisons.
    "openai/gpt-oss-120b": ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60),
    "openai/gpt-oss-20b": ModelPrice(input_per_mtok=0.075, output_per_mtok=0.30),
    "qwen/qwen3.8-27b": ModelPrice(input_per_mtok=0.80, output_per_mtok=4.00),
}


class LLMSettings(BaseModel):
    """Which provider/model to call and per-request generation limits."""

    provider: ProviderName = "groq"
    # Strongest tool-calling model on Groq's free tier for code (see DECISIONS.md #19).
    model: str = "openai/gpt-oss-120b"
    max_output_tokens: int = Field(default=16000, gt=0)
    # Groq free tier: requests are charged prompt + max_tokens against a per-minute token
    # limit, and a single request over it fails with 413. The Groq provider shrinks
    # max_completion_tokens to fit. None disables the clamp (paid tiers).
    groq_tpm_limit: int | None = Field(default=8000, gt=0)
    min_output_tokens: int = Field(default=1024, gt=0)
    # Record Groq usage as $0.00 (free tier) while still reporting the list-price equivalent.
    groq_free_tier: bool = True

    def tool_choice_for(self, provider: str) -> ToolChoice:
        """The tool_choice ``provider`` sends: explicit setting, else that provider's default."""
        return self.tool_choice or DEFAULT_TOOL_CHOICE.get(provider, "auto")

    # "required" forces a tool call every turn, so the agent can only stop via finish.
    # None = provider default: "required" for Groq, "auto" for Anthropic (forced tool use
    # is rejected by current Claude models while thinking is on; see DECISIONS.md #21).
    tool_choice: ToolChoice | None = None
    effort: Effort | None = None
    # Not sent when None (the default: model default sampling). Groq accepts it (verified by a
    # live request). The anthropic SDK 1.x no longer takes it as an argument, so it is sent
    # raw; Anthropic's docs say current models (e.g. Sonnet 5) reject it with a 400, which we
    # could not confirm live (no API credit). See DECISIONS.md #17.
    temperature: float | None = Field(default=None, ge=0, le=1)
    prompt_caching: bool = True
    request_timeout_s: float = Field(default=300.0, gt=0)
    # Retries are done by the agent (agent/retry.py) so each attempt is traced.
    max_retries: int = Field(default=3, ge=0)
    retry_base_delay_s: float = Field(default=2.0, gt=0)
    retry_max_delay_s: float = Field(default=30.0, gt=0)


class BudgetSettings(BaseModel):
    """Hard per-task limits enforced by the agent loop."""

    max_iterations: int = Field(default=30, gt=0)
    # Counts uncached input + cache writes + output (not cache reads); see Usage.budget_tokens.
    max_tokens_per_task: int = Field(default=500_000, gt=0)
    # Optional cap on the list-price cost estimate (applies on free tiers too). None = off.
    max_cost_usd_per_task: float | None = Field(default=None, gt=0)
    max_test_runs: int = Field(default=10, gt=0)
    wall_clock_timeout_s: float = Field(default=900.0, gt=0)


class AgentSettings(BaseModel):
    """Agent-loop behaviour beyond the hard budgets."""

    tree_depth: int = Field(default=3, ge=1, le=5)
    tree_max_entries: int = Field(default=200, gt=0)
    # When a request's input exceeds this, old tool results are replaced with stubs.
    context_elide_tokens: int = Field(default=60_000, gt=0)
    keep_recent_turns: int = Field(default=4, ge=1)


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
    agent: AgentSettings = Field(default_factory=AgentSettings)
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
