"""Any OpenAI-compatible chat-completions endpoint (Cerebras, NVIDIA, OpenRouter, vLLM, ...).

Same wire format and error handling as :class:`GroqProvider`; only the client differs.
Groq-specific 400 codes (``tool_use_failed``) simply never occur on other hosts, and a
host that returns them gets the same treatment.
"""

from __future__ import annotations

from typing import Any

import openai

from repair_agent.config import LLMSettings
from repair_agent.llm.groq import GroqProvider


class OpenAICompatibleProvider(GroqProvider):
    """Calls an OpenAI-compatible endpoint at ``base_url`` via the official ``openai`` SDK."""

    name = "openai_compatible"
    _status_error = openai.APIStatusError
    _connection_error = openai.APIConnectionError  # includes APITimeoutError
    _tpm_setting = "the backend's tpm_limit in REPAIR_LLM__POOL"

    def __init__(
        self,
        settings: LLMSettings,
        api_key: str,
        client: Any | None = None,
        *,
        base_url: str | None = None,
        name: str | None = None,
        send_reasoning_effort: bool = True,
    ):
        self._base_url = base_url
        super().__init__(
            settings,
            api_key,
            client,
            name=name,
            send_reasoning_effort=send_reasoning_effort,
        )

    def _make_client(self, settings: LLMSettings, api_key: str) -> Any:
        if not self._base_url:
            raise RuntimeError(f"backend {self.name!r} needs a base_url")
        return openai.OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            timeout=settings.request_timeout_s,
            max_retries=0,
        )
