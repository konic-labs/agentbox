"""Shell command tool."""

from __future__ import annotations

from typing import Any

from agentbox.tools.base import BaseTool, ToolResult

OUTPUT_LIMIT = 50_000


def _truncate(text: str, limit: int = OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


class RunCommandTool(BaseTool):
    name = "run_command"
    description = "Execute a shell command inside the container workspace."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout_s": {
                    "type": "number",
                    "description": "Timeout in seconds",
                    "default": 60,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        sandbox: Any,
        command: str,
        timeout_s: float = 60.0,
        **_: Any,
    ) -> str | ToolResult:
        try:
            result = await sandbox.exec(command, timeout_s=float(timeout_s))
        except Exception as exc:
            return ToolResult(
                content=f"ERROR: run_command failed: {exc}", is_error=True
            )

        parts = [
            f"exit_code={result.exit_code}",
        ]
        if result.timed_out:
            parts.append("timed_out=true")
        if result.stdout:
            parts.append("--- stdout ---\n" + _truncate(result.stdout))
        if result.stderr:
            parts.append("--- stderr ---\n" + _truncate(result.stderr))
        content = "\n".join(parts)
        if result.exit_code != 0 or result.timed_out:
            return ToolResult(content=content, is_error=True)
        return content
