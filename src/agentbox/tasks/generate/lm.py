"""LM backend wiring for task generation (OpenAI-compatible)."""

from __future__ import annotations

import os
from typing import Any

from agentbox.config import ModelConfig
from agentbox.errors import TaskGenerationError


def build_openai_client(config: ModelConfig) -> Any:
    from openai import AsyncOpenAI

    api_key = config.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": config.timeout_s}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.extra_headers:
        kwargs["default_headers"] = config.extra_headers
    return AsyncOpenAI(**kwargs)


def configure_dspy_lm(config: ModelConfig) -> Any:
    """Configure a DSPy LM for OpenAI-compatible endpoints."""
    try:
        import dspy
    except ImportError as exc:
        raise TaskGenerationError(
            "dspy is required for TaskGenerator. Install with: pip install agentbox[generate]"
        ) from exc

    api_key = config.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    kwargs: dict[str, Any] = {
        "model": config.model,
        "api_key": api_key,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens or 8192,
    }
    if config.base_url:
        # dspy OpenAI-compatible: model often as openai/name with api_base
        kwargs["api_base"] = config.base_url
        # Prefer explicit provider prefix if not present
        if not str(config.model).startswith(("openai/", "ollama/", "litellm/")):
            kwargs["model"] = f"openai/{config.model}"

    try:
        lm = dspy.LM(**kwargs)
    except TypeError:
        # Older dspy signatures
        lm = dspy.LM(config.model, api_key=api_key, api_base=config.base_url)
    dspy.configure(lm=lm)
    return lm
