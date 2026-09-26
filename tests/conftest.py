from __future__ import annotations

import pytest

_ENV_VARS = ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "GITHUB_TOKEN")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Keep tests independent of the developer's shell env and .env file."""
    import os

    for name in list(os.environ):
        if name.startswith("REPAIR_") or name in _ENV_VARS:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in the working directory

    from repair_agent.config import get_settings

    get_settings.cache_clear()
