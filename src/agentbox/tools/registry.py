"""Tool registry: builtins, custom tools, overrides."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Any

from agentbox.tools.base import BaseTool
from agentbox.tools.builtins import default_tools, shell_tools
from agentbox.types import ToolMode


class ToolRegistry:
    """Named collection of tools exposed to the model."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool, *, override: bool = True) -> None:
        if not override and tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]

    def drop(self, names: Iterable[str]) -> None:
        for name in names:
            self._tools.pop(name, None)


def _as_tool(obj: Any) -> BaseTool:
    if isinstance(obj, BaseTool):
        return obj
    raise TypeError(f"Expected BaseTool, got {type(obj)!r}")


def build_tool_registry(
    mode: ToolMode | list[str] | str = ToolMode.STRUCTURED,
    *,
    custom_tools: Sequence[Any] | None = None,
    include_builtins: bool = True,
    drop: Iterable[str] | None = None,
    drop_prob: float = 0.0,
    rng: random.Random | None = None,
) -> ToolRegistry:
    """Compose a tool set from mode, custom tools, and drop rules."""
    registry = ToolRegistry()
    rng = rng or random.Random()

    if isinstance(mode, str) and mode not in {m.value for m in ToolMode}:
        # treat as invalid — fall through
        pass

    if include_builtins:
        if mode == ToolMode.STRUCTURED or mode == "structured":
            for t in default_tools():
                registry.register(t)
        elif mode == ToolMode.SHELL or mode == "shell":
            for t in shell_tools():
                registry.register(t)
        elif mode == ToolMode.CUSTOM or mode == "custom":
            pass
        elif isinstance(mode, list):
            all_builtins = {t.name: t for t in default_tools()}
            for name in mode:
                if name in all_builtins:
                    registry.register(all_builtins[name])
                else:
                    raise ValueError(f"Unknown builtin tool: {name}")
        else:
            for t in default_tools():
                registry.register(t)

    if custom_tools:
        for obj in custom_tools:
            registry.register(_as_tool(obj), override=True)

    if drop:
        registry.drop(drop)

    if drop_prob > 0:
        to_drop = [n for n in registry.names() if rng.random() < drop_prob]
        registry.drop(to_drop)

    return registry
