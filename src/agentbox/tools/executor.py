"""Execute tool calls from the model."""

from __future__ import annotations

import json
import time
from typing import Any

from agentbox.tools.base import ToolContext, ToolResult
from agentbox.tools.registry import ToolRegistry
from agentbox.trajectory.schema import Message, ToolCall


class ToolExecutor:
    """Dispatch OpenAI tool_calls to registered tools."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def schemas(self) -> list[dict[str, Any]]:
        return self.registry.openai_tools()

    def names(self) -> list[str]:
        return self.registry.names()

    async def execute_one(self, tool_call: ToolCall, ctx: ToolContext) -> Message:
        name = tool_call.function.name
        tool = self.registry.get(name)
        start = time.monotonic()

        if tool is None:
            content = f"ERROR: unknown tool: {name}"
            return Message(
                role="tool",  # type: ignore[arg-type]
                content=content,
                tool_call_id=tool_call.id,
                name=name,
            )

        try:
            args = json.loads(tool_call.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("tool arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            content = f"ERROR: invalid arguments for {name}: {exc}"
            return Message(
                role="tool",  # type: ignore[arg-type]
                content=content,
                tool_call_id=tool_call.id,
                name=name,
            )

        try:
            result = await tool.execute(ctx.sandbox, **args)
        except Exception as exc:
            result = ToolResult(content=f"ERROR: {name} failed: {exc}", is_error=True)

        if isinstance(result, ToolResult):
            content = result.content
        else:
            content = str(result)

        _ = time.monotonic() - start
        return Message(
            role="tool",  # type: ignore[arg-type]
            content=content,
            tool_call_id=tool_call.id,
            name=name,
        )

    async def execute_many(
        self,
        tool_calls: list[ToolCall],
        ctx: ToolContext,
        *,
        parallel: bool = True,
    ) -> list[Message]:
        if not tool_calls:
            return []
        if not parallel or len(tool_calls) == 1:
            return [await self.execute_one(tc, ctx) for tc in tool_calls]

        import asyncio

        return list(
            await asyncio.gather(*(self.execute_one(tc, ctx) for tc in tool_calls))
        )
