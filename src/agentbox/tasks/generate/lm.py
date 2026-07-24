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


def build_dspy_lm(config: ModelConfig) -> Any:
    """Construct a DSPy LM without mutating global settings (safe under concurrency)."""
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
        kwargs["api_base"] = config.base_url
        mid = str(config.model)
        # Paths like /data/models/... must still get openai/ for LiteLLM custom endpoints
        if not mid.startswith(("openai/", "ollama/", "litellm/")):
            kwargs["model"] = f"openai/{mid}"
    if config.extra_body:
        kwargs["extra_body"] = dict(config.extra_body)
    if config.extra_headers:
        kwargs["extra_headers"] = config.extra_headers
    if config.timeout_s:
        kwargs["timeout"] = config.timeout_s

    try:
        return dspy.LM(**kwargs)
    except TypeError:
        return dspy.LM(config.model, api_key=api_key, api_base=config.base_url)


def configure_dspy_lm(config: ModelConfig, *, set_global: bool = True) -> Any:
    """Build a DSPy LM; optionally set process-wide default (main thread only).

    Under concurrent asyncio tasks, prefer ``build_dspy_lm`` + ``dspy.context(lm=...)``
    instead of calling this with set_global=True from multiple tasks.
    """
    try:
        import dspy
    except ImportError as exc:
        raise TaskGenerationError(
            "dspy is required for TaskGenerator. Install with: pip install agentbox[generate]"
        ) from exc

    lm = build_dspy_lm(config)
    if set_global:
        try:
            dspy.configure(lm=lm)
        except RuntimeError:
            # Already configured from another async task — caller should use context()
            pass
    return lm
