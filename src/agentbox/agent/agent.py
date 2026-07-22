"""High-level Agent facade."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agentbox.config import AgentConfig, ModelConfig
from agentbox.model.base import ModelClient
from agentbox.model.openai_compat import OpenAICompatClient
from agentbox.tools.base import BaseTool
from agentbox.types import ToolMode


class Agent:
    """Ergonomic agent configuration for rollouts."""

    def __init__(
        self,
        model: str | ModelConfig | ModelClient,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        tools: ToolMode | list[str] | str = ToolMode.STRUCTURED,
        custom_tools: Sequence[BaseTool] | None = None,
        builtins: bool = True,
        max_steps: int = 40,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        if isinstance(model, str):
            self.model_config = ModelConfig(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
            )
            self.model_client: ModelClient = OpenAICompatClient(self.model_config)
        elif isinstance(model, ModelConfig):
            self.model_config = model
            self.model_client = OpenAICompatClient(model)
        else:
            self.model_config = ModelConfig(model=getattr(model, "model", "unknown"))
            self.model_client = model

        tool_mode: ToolMode | list[str]
        if isinstance(tools, str) and tools in {m.value for m in ToolMode}:
            tool_mode = ToolMode(tools)
        else:
            tool_mode = tools  # type: ignore[assignment]

        agent_kwargs = {
            "tools": tool_mode,
            "custom_tools": list(custom_tools or []),
            "builtins": builtins,
            "max_steps": max_steps,
            **kwargs,
        }
        self.config = AgentConfig(**agent_kwargs)
