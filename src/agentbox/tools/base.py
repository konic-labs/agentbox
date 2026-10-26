"""Tool base types and ABC."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Normalized tool execution result."""

    content: str
    is_error: bool = False
    terminal: bool = False  # ends the episode (e.g. submit)
    meta: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ToolContext:
    """Execution context for a tool call (available via :func:`get_tool_context`)."""

    sandbox: Any
    manager: Any = None
    workspace_dir: str = "/workspace"
    step: int = 0
    run_id: str = ""
    task_id: str = ""
    deadline: float | None = None  # time.monotonic() deadline of the episode
    tool_timeout_s: float = 120.0  # upper bound for any single tool call
    max_output_chars: int = 16_000  # budget for a tool result shown to the model

    def remaining_s(self) -> float | None:
        if self.deadline is None:
            return None
        return self.deadline - time.monotonic()

    def clamp_timeout(self, requested: float | None, *, default: float = 60.0) -> float:
        """Clamp a model-requested timeout to [1, tool_timeout_s] and the episode budget."""
        try:
            value = float(requested) if requested is not None else default
        except (TypeError, ValueError):
            value = default
        if value != value or value <= 0:  # NaN or non-positive
            value = default
        value = max(1.0, min(value, self.tool_timeout_s))
        remaining = self.remaining_s()
        if remaining is not None:
            value = max(0.5, min(value, remaining))
        return value


_CURRENT: ContextVar[ToolContext | None] = ContextVar("agentbox_tool_context", default=None)


def get_tool_context() -> ToolContext | None:
    """The :class:`ToolContext` of the tool call currently executing (or ``None``)."""
    return _CURRENT.get()


class BaseTool(ABC):
    """Public extension point for custom tools."""

    name: str
    description: str
    read_only: bool = False  # safe to run concurrently with other read-only tools
    terminal: bool = False  # calling it ends the episode

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for function.parameters (OpenAI tools format)."""

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters(),
            },
        }

    @abstractmethod
    async def execute(self, sandbox: Any, **kwargs: Any) -> str | ToolResult:
        """Run tool I/O via the sandbox only."""
