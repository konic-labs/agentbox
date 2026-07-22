"""Convenience test runner tool."""

from __future__ import annotations

from typing import Any

from agentbox.tools.base import BaseTool, ToolResult
from agentbox.tools.shell import RunCommandTool


class RunTestsTool(BaseTool):
    name = "run_tests"
    description = "Run the test suite inside the container (default: pytest)."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Test command to run",
                    "default": "python -m pytest -q",
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        sandbox: Any,
        command: str = "python -m pytest -q",
        **_: Any,
    ) -> str | ToolResult:
        runner = RunCommandTool()
        return await runner.execute(sandbox, command=command, timeout_s=120.0)
