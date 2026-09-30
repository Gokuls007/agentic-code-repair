"""LLM providers behind a common interface."""

from __future__ import annotations

import logging

from repair_agent.config import BackendSettings, Settings
from repair_agent.llm.base import LLMProvider

logger = logging.getLogger(__name__)


def create_provider(settings: Settings) -> LLMProvider:
    """Build the provider named in ``settings.llm.provider``."""
    provider = settings.llm.provider
    if provider == "anthropic":
        from repair_agent.llm.anthropic import AnthropicProvider

        return AnthropicProvider(settings.llm, api_key=settings.api_key_for("anthropic"))
    if provider == "groq":
        from repair_agent.llm.groq import GroqProvider

        return GroqProvider(settings.llm, api_key=settings.api_key_for("groq"))
    if provider == "pool":
        return create_pool(settings)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def pool_backends_available(settings: Settings) -> tuple[list[BackendSettings], list[str]]:
    """(backends that have an API key, labels of those skipped because theirs is unset)."""
    usable, skipped = [], []
    for backend in settings.llm.pool:
        (usable if settings.env_secret(backend.api_key_env) else skipped).append(backend)
    return usable, [f"{b.name} ({b.api_key_env} not set)" for b in skipped]


def create_backend(settings: Settings, backend: BackendSettings) -> LLMProvider:
    """One pool member as a standalone provider (also used by ``ping`` per backend)."""
    from repair_agent.llm.groq import GroqProvider
    from repair_agent.llm.openai_compat import OpenAICompatibleProvider

    key = settings.env_secret(backend.api_key_env)
    if not key:
        raise RuntimeError(f"{backend.api_key_env} is not set; add it to .env")
    llm = settings.llm
    tuned = llm.model_copy(
        update={
            "model": backend.model_for(llm.model),
            "groq_tpm_limit": backend.tpm_limit,
            "tool_choice": backend.tool_choice or llm.tool_choice or "required",
        }
    )
    if backend.kind == "groq":
        return GroqProvider(
            tuned, key, name=backend.name, send_reasoning_effort=backend.reasoning_effort
        )
    return OpenAICompatibleProvider(
        tuned,
        key,
        base_url=backend.base_url,
        name=backend.name,
        send_reasoning_effort=backend.reasoning_effort,
    )


def create_pool(settings: Settings) -> LLMProvider:
    """A PoolProvider over every configured backend that has an API key."""
    from repair_agent.llm.pool import Backend, PoolProvider

    usable, skipped = pool_backends_available(settings)
    if not usable:
        raise RuntimeError(
            "no pool backend has an API key: " + ", ".join(skipped) + "; add one to .env"
        )
    for note in skipped:
        logger.warning("pool backend skipped: %s", note)
    return PoolProvider(
        [
            Backend(name=b.name, provider=create_backend(settings, b), free_tier=b.free_tier)
            for b in usable
        ],
        settings.llm,
    )
