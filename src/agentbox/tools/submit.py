"""submit: explicit episode termination."""

from __future__ import annotations

from typing import Any

from agentbox.tools.base import BaseTool, ToolResult


class SubmitTool(BaseTool):
    name = "submit"
    description = (
        "Finish the task. Call this exactly once when your work is complete; "
        "the episode ends immediately and your changes are evaluated."
    )
    terminal = True

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Optional final answer or summary of the changes",
                }
            },
            "additionalProperties": False,
        }

    async def execute(self, sandbox: Any, answer: str = "", **_: Any) -> str | ToolResult:  # noqa: ARG002
        return ToolResult(content="Submitted.", terminal=True, meta={"answer": answer})
