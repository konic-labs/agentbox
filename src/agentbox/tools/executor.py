"""Execute tool calls from the model.

Semantics (GH #27, #28, #30):

* calls run **in emission order**; only consecutive ``read_only`` tools may run
  concurrently, so ``[write_file, run_tests]`` always observes the write;
* every call yields exactly one tool message, even on timeout, error, unknown
  tool or after a terminal tool — the conversation stays well-formed;
* per-call timeouts are bounded by the episode deadline;
* results are truncated head+tail and recorded with ``is_error``/duration.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from dataclasses import dataclass
from typing import Any

from agentbox.tools.base import _CURRENT, BaseTool, ToolContext, ToolResult
from agentbox.tools.registry import ToolRegistry
from agentbox.trajectory.schema import Message, ToolCall, ToolCallRecord
from agentbox.types import MessageRole
from agentbox.utils.jsonparse import loads_lenient
from agentbox.utils.text import truncate_middle

_TIMEOUT_GRACE_S = 5.0
RECORD_RESULT_CHARS = 2_000  # tool records keep a preview; the message keeps the full text


@dataclass(slots=True)
class ToolOutcome:
    message: Message
    record: ToolCallRecord
    terminal: bool = False
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)


def parse_arguments(raw: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Parse OpenAI ``function.arguments``; returns ``(args, error)``."""
    if raw is None or raw.strip() == "":
        return {}, None
    try:
        value = loads_lenient(raw)
    except ValueError as exc:
        return None, f"arguments are not valid JSON ({exc})"
    if not isinstance(value, dict):
        return None, "arguments must be a JSON object"
    return value, None


class ToolExecutor:
    """Dispatch OpenAI tool_calls to registered tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_output_chars: int = 16_000,
        tool_timeout_s: float = 120.0,
        parallel_read_only: bool = True,
        record_result_chars: int = RECORD_RESULT_CHARS,
    ) -> None:
        self.registry = registry
        self.max_output_chars = max_output_chars
        self.tool_timeout_s = tool_timeout_s
        self.parallel_read_only = parallel_read_only
        self.record_result_chars = record_result_chars

    def schemas(self) -> list[dict[str, Any]]:
        return self.registry.openai_tools()

    def names(self) -> list[str]:
        return self.registry.names()

    # ------------------------------------------------------------------ core
    def _outcome(
        self,
        tc: ToolCall,
        step: int,
        content: str,
        *,
        is_error: bool,
        args: dict[str, Any] | None,
        duration_s: float = 0.0,
        terminal: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> ToolOutcome:
        shown = truncate_middle(content, self.max_output_chars)
        truncated = len(shown) != len(content)
        message = Message(
            role=MessageRole.TOOL,
            content=shown,
            tool_call_id=tc.id,
            name=tc.function.name,
        )
        record = ToolCallRecord(
            step=step,
            tool_call_id=tc.id,
            name=tc.function.name,
            arguments=args if args is not None else {"_raw": tc.function.arguments},
            result=truncate_middle(shown, self.record_result_chars),
            is_error=is_error,
            duration_s=duration_s,
            truncated=truncated,
        )
        return ToolOutcome(message=message, record=record, terminal=terminal, meta=meta or {})

    async def run_call(
        self, tc: ToolCall, ctx: ToolContext, *, deadline: float | None = None
    ) -> ToolOutcome:
        name = tc.function.name
        step = ctx.step
        tool = self.registry.get(name)
        if tool is None:
            available = ", ".join(self.registry.names()) or "(none)"
            return self._outcome(
                tc,
                step,
                f"ERROR: unknown tool {name!r}. Available tools: {available}",
                is_error=True,
                args=None,
            )
        args, err = parse_arguments(tc.function.arguments)
        if err is not None or args is None:
            return self._outcome(
                tc, step, f"ERROR: invalid arguments for {name}: {err}", is_error=True, args=None
            )
        required = tool.parameters().get("required") or []
        missing = [r for r in required if r not in args]
        if missing:
            return self._outcome(
                tc,
                step,
                f"ERROR: {name} is missing required argument(s): {', '.join(missing)}",
                is_error=True,
                args=args,
            )
        deadline = deadline if deadline is not None else ctx.deadline
        budget = self.tool_timeout_s
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._outcome(
                    tc,
                    step,
                    "ERROR: episode time limit reached; tool call not executed",
                    is_error=True,
                    args=args,
                )
            budget = min(budget, remaining)
        call_ctx = dataclasses.replace(
            ctx,
            deadline=deadline,
            tool_timeout_s=min(ctx.tool_timeout_s, self.tool_timeout_s),
            max_output_chars=self.max_output_chars,
        )
        token = _CURRENT.set(call_ctx)
        t0 = time.monotonic()
        try:
            result: str | ToolResult = await asyncio.wait_for(
                tool.execute(ctx.sandbox, **args), budget + _TIMEOUT_GRACE_S
            )
        except TimeoutError:
            return self._outcome(
                tc,
                step,
                f"ERROR: {name} timed out after {budget:.0f}s",
                is_error=True,
                args=args,
                duration_s=time.monotonic() - t0,
            )
        except TypeError as exc:
            return self._outcome(
                tc,
                step,
                f"ERROR: invalid arguments for {name}: {exc}",
                is_error=True,
                args=args,
                duration_s=time.monotonic() - t0,
            )
        except Exception as exc:
            return self._outcome(
                tc,
                step,
                f"ERROR: {name} failed: {exc}",
                is_error=True,
                args=args,
                duration_s=time.monotonic() - t0,
            )
        finally:
            _CURRENT.reset(token)
        duration = time.monotonic() - t0
        if isinstance(result, ToolResult):
            return self._outcome(
                tc,
                step,
                result.content,
                is_error=result.is_error,
                args=args,
                duration_s=duration,
                terminal=result.terminal or tool.terminal,
                meta=result.meta,
            )
        return self._outcome(
            tc,
            step,
            str(result),
            is_error=False,
            args=args,
            duration_s=duration,
            terminal=tool.terminal,
        )

    def _is_read_only(self, tc: ToolCall) -> bool:
        tool: BaseTool | None = self.registry.get(tc.function.name)
        return bool(tool is not None and tool.read_only)

    async def run_calls(
        self,
        tool_calls: list[ToolCall],
        ctx: ToolContext,
        *,
        deadline: float | None = None,
        parallel: bool | None = None,
    ) -> list[ToolOutcome]:
        parallel = self.parallel_read_only if parallel is None else parallel
        outcomes: list[ToolOutcome] = []
        i = 0
        n = len(tool_calls)
        while i < n:
            if outcomes and outcomes[-1].terminal:
                for tc in tool_calls[i:]:
                    outcomes.append(
                        self._outcome(
                            tc,
                            ctx.step,
                            "Skipped: the episode was already submitted.",
                            is_error=True,
                            args=None,
                        )
                    )
                break
            if parallel and self._is_read_only(tool_calls[i]):
                j = i
                while j < n and self._is_read_only(tool_calls[j]):
                    j += 1
                if j - i > 1:
                    outcomes.extend(
                        await asyncio.gather(
                            *(self.run_call(tc, ctx, deadline=deadline) for tc in tool_calls[i:j])
                        )
                    )
                    i = j
                    continue
            outcomes.append(await self.run_call(tool_calls[i], ctx, deadline=deadline))
            i += 1
        return outcomes

    # --------------------------------------------------------------- compat
    async def execute_one(self, tool_call: ToolCall, ctx: ToolContext) -> Message:
        return (await self.run_call(tool_call, ctx)).message

    async def execute_many(
        self,
        tool_calls: list[ToolCall],
        ctx: ToolContext,
        *,
        parallel: bool = True,
    ) -> list[Message]:
        if not tool_calls:
            return []
        outcomes = await self.run_calls(tool_calls, ctx, parallel=parallel)
        return [o.message for o in outcomes]
