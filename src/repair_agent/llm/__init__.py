"""LLM providers behind a common interface."""

from __future__ import annotations

from repair_agent.config import Settings
from repair_agent.llm.base import LLMProvider


def create_provider(settings: Settings) -> LLMProvider:
    """Build the provider named in ``settings.llm.provider``."""
    provider = settings.llm.provider
    if provider == "anthropic":
        from repair_agent.llm.anthropic import AnthropicProvider

        return AnthropicProvider(settings.llm, api_key=settings.api_key_for("anthropic"))
    if provider == "groq":
        raise NotImplementedError("The Groq provider is added in Phase 6")
    raise ValueError(f"Unknown LLM provider: {provider!r}")
