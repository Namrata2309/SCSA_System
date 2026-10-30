"""Environment-driven LLM provider selection without mandatory credentials."""

import os
from pathlib import Path
from typing import MutableMapping

from .base import AssessmentProvider
from .mock import MockProvider
from .gemini import GeminiProvider
from .openai_responses import GroqResponsesProvider, OpenAIResponsesProvider


def load_local_environment(
    environ: MutableMapping[str, str] | None = None,
    env_file: str | Path = ".env",
) -> None:
    """Read simple KEY=VALUE entries without overriding existing environment values."""
    target = os.environ if environ is None else environ
    path = Path(env_file)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            target.setdefault(key, value)


def create_provider(
    provider_name: str | None = None,
    *,
    env_file: str | Path | None = ".env",
    environ: MutableMapping[str, str] | None = None,
) -> AssessmentProvider:
    """Create a configured provider; mock is the safe no-credentials default."""
    target = os.environ if environ is None else environ
    # Explicit mock mode never needs credentials, so avoid loading the local
    # environment file (which may contain secrets) for offline runs.
    if env_file is not None and provider_name != "mock":
        load_local_environment(target, env_file)
    selected = (provider_name or target.get("LLM_PROVIDER", "mock")).strip().lower()
    model = target.get("LLM_MODEL")
    if selected == "mock":
        return MockProvider()
    if selected == "openai":
        return OpenAIResponsesProvider(
            api_key=target.get("OPENAI_API_KEY", ""),
            model=model or "gpt-6-astra",
        )
    if selected == "groq":
        return GroqResponsesProvider(
            api_key=target.get("LLM_API_KEY") or target.get("GROQ_API_KEY", ""),
            model=model or "openai/gpt-oss-20b",
        )
    if selected == "gemini":
        return GeminiProvider(
            api_key=target.get("GEMINI_API_KEY", ""),
            model=model or "gemini-3.8-flash",
            thinking_level=target.get("GEMINI_THINKING_LEVEL", "low"),
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {selected}")
