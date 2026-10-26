"""Shell command tool."""

from __future__ import annotations

from typing import Any

from agentbox.tools.base import BaseTool, ToolContext, ToolResult, get_tool_context
from agentbox.utils.text import truncate_middle

OUTPUT_LIMIT = 16_000
DEFAULT_TIMEOUT_S = 60.0


def _truncate(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Keep head and tail (errors and test summaries are at the end)."""
    return truncate_middle(text, limit)


def format_exec_result(result: Any, *, limit: int) -> str:
    parts = [f"exit_code={result.exit_code}"]
    if result.timed_out:
        parts.append("timed_out=true")
    budget = max(200, limit // 2) if result.stdout and result.stderr else limit
    if result.stdout:
        parts.append("--- stdout ---\n" + _truncate(result.stdout, budget))
    if result.stderr:
        parts.append("--- stderr ---\n" + _truncate(result.stderr, budget))
    return "\n".join(parts)


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
                    "description": "Timeout in seconds (capped by the agent's tool limit)",
                    "default": DEFAULT_TIMEOUT_S,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        sandbox: Any,
        command: str,
        timeout_s: float | None = None,
        **_: Any,
    ) -> str | ToolResult:
        if not isinstance(command, str) or not command.strip():
            return ToolResult(content="ERROR: run_command failed: empty command", is_error=True)
        ctx = get_tool_context() or ToolContext(sandbox=sandbox)
        effective = ctx.clamp_timeout(timeout_s, default=DEFAULT_TIMEOUT_S)
        try:
            result = await sandbox.exec(command, timeout_s=effective)
        except Exception as exc:
            return ToolResult(content=f"ERROR: run_command failed: {exc}", is_error=True)
        content = format_exec_result(result, limit=ctx.max_output_chars)
        if result.exit_code != 0 or result.timed_out:
            return ToolResult(
                content=content,
                is_error=True,
                meta={"exit_code": result.exit_code, "timed_out": result.timed_out},
            )
        return content
