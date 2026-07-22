"""Tool base types and ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolResult(BaseModel):
    """Normalized tool execution result."""

    content: str
    is_error: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class ToolContext(BaseModel):
    """Internal execution context for builtins."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sandbox: Any
    manager: Any
    workspace_dir: str
    step: int = 0
    run_id: str = ""
    task_id: str = ""


class BaseTool(ABC):
    """Public extension point for custom tools."""

    name: str
    description: str

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
