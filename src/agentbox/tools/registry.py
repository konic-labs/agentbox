"""Tool registry: builtins, custom tools, overrides."""

from __future__ import annotations

import builtins
import random
from collections.abc import Iterable, Sequence
from typing import Any

from agentbox.tools.base import BaseTool
from agentbox.tools.builtins import all_builtins, default_tools, extended_tools, shell_tools
from agentbox.types import ToolMode


class ToolRegistry:
    """Named collection of tools exposed to the model."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._schemas: list[dict[str, Any]] | None = None

    def register(self, tool: BaseTool, *, override: bool = True) -> None:
        if not override and tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self._schemas = None

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._schemas = None

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list(self) -> builtins.list[BaseTool]:
        return builtins.list(self._tools.values())

    def names(self) -> builtins.list[str]:
        return builtins.list(self._tools.keys())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def openai_tools(self) -> builtins.list[dict[str, Any]]:
        """Schemas, built once and cached until the registry changes."""
        if self._schemas is None:
            self._schemas = [t.openai_schema() for t in self._tools.values()]
        return self._schemas

    def drop(self, names: Iterable[str]) -> None:
        for name in names:
            self._tools.pop(name, None)
        self._schemas = None


def _as_tool(obj: Any) -> BaseTool:
    if isinstance(obj, BaseTool):
        return obj
    raise TypeError(f"Expected BaseTool, got {type(obj)!r}")


def _mode_value(mode: ToolMode | list[str] | str) -> ToolMode | list[str] | str:
    if isinstance(mode, ToolMode):
        return mode
    if isinstance(mode, str):
        try:
            return ToolMode(mode)
        except ValueError as exc:
            raise ValueError(
                f"Unknown tool mode {mode!r}; use one of {[m.value for m in ToolMode]} "
                "or a list of builtin tool names"
            ) from exc
    return mode


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
    mode = _mode_value(mode)

    if include_builtins:
        if mode == ToolMode.STRUCTURED:
            for t in default_tools():
                registry.register(t)
        elif mode == ToolMode.EXTENDED:
            for t in extended_tools():
                registry.register(t)
        elif mode == ToolMode.SHELL:
            for t in shell_tools():
                registry.register(t)
        elif mode == ToolMode.CUSTOM:
            pass
        elif isinstance(mode, list):
            builtins = all_builtins()
            for name in mode:
                if name in builtins:
                    registry.register(builtins[name])
                elif not any(getattr(c, "name", None) == name for c in custom_tools or ()):
                    raise ValueError(f"Unknown builtin tool: {name}")

    if custom_tools:
        for obj in custom_tools:
            registry.register(_as_tool(obj), override=True)

    if drop:
        registry.drop(drop)

    if drop_prob > 0 and len(registry) > 1:
        names = registry.names()
        to_drop = [n for n in names if rng.random() < drop_prob]
        if len(to_drop) >= len(names):  # always leave the agent at least one tool
            to_drop.remove(rng.choice(to_drop))
        registry.drop(to_drop)

    return registry
