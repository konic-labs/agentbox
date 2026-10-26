"""High-level Agent facade."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agentbox.config import AgentConfig, ModelConfig
from agentbox.model.base import ModelClient
from agentbox.model.openai_compat import OpenAICompatClient
from agentbox.tools.base import BaseTool
from agentbox.types import ToolMode

_MODEL_FIELDS = set(ModelConfig.model_fields) - {"model", "base_url", "api_key", "temperature"}


class Agent:
    """Ergonomic agent configuration for rollouts.

    Keyword arguments that are :class:`ModelConfig` fields (``max_tokens``,
    ``top_p``, ``extra_body``, ...) configure the model client; everything else
    configures the :class:`AgentConfig`.
    """

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
        model_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _MODEL_FIELDS}
        if isinstance(model, str):
            self.model_config = ModelConfig(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                **model_kwargs,
            )
            self.model_client: ModelClient = OpenAICompatClient(self.model_config)
        elif isinstance(model, ModelConfig):
            self.model_config = model.model_copy(update=model_kwargs) if model_kwargs else model
            self.model_client = OpenAICompatClient(self.model_config)
        else:
            self.model_config = ModelConfig(model=getattr(model, "model", "unknown"))
            self.model_client = model

        tool_mode: ToolMode | list[str]
        if isinstance(tools, str):
            tool_mode = ToolMode(tools)
        else:
            tool_mode = tools

        self.config = AgentConfig(
            tools=tool_mode,
            custom_tools=list(custom_tools or []),
            builtins=builtins,
            max_steps=max_steps,
            **kwargs,
        )

    async def aclose(self) -> None:
        close = getattr(self.model_client, "aclose", None)
        if close is not None:
            await close()
