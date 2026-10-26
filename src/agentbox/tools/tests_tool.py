"""Convenience test runner tool."""

from __future__ import annotations

from typing import Any

from agentbox.tools.base import BaseTool, ToolResult
from agentbox.tools.shell import RunCommandTool

DEFAULT_TEST_COMMAND = "python -m pytest -q"


class RunTestsTool(BaseTool):
    name = "run_tests"
    description = "Run the test suite inside the container (default: python -m pytest -q)."

    def __init__(self, default_command: str = DEFAULT_TEST_COMMAND, timeout_s: float = 120.0):
        self.default_command = default_command
        self.timeout_s = timeout_s
        self._runner = RunCommandTool()

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Test command to run",
                    "default": self.default_command,
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        sandbox: Any,
        command: str | None = None,
        **_: Any,
    ) -> str | ToolResult:
        return await self._runner.execute(
            sandbox, command=command or self.default_command, timeout_s=self.timeout_s
        )
