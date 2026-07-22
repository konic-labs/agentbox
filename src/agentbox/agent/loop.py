"""Multi-turn agent loop with OpenAI tool calling."""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from agentbox.config import AgentConfig
from agentbox.model.base import ModelClient, ModelResponse
from agentbox.tools.base import ToolContext
from agentbox.tools.executor import ToolExecutor
from agentbox.trajectory.schema import Message, ToolCallRecord
from agentbox.types import FinalStatus, MessageRole


class AgentLoopResult(BaseModel):
    messages: list[Message]
    tool_call_records: list[ToolCallRecord]
    final_status: FinalStatus
    steps: int
    model_calls: int
    stop_reason: str
    error: str | None = None
    usage_totals: dict[str, int] = Field(default_factory=dict)


class AgentLoop:
    def __init__(
        self,
        model: ModelClient,
        tools: ToolExecutor,
        config: AgentConfig,
        *,
        on_step: Callable[[int, Message], Awaitable[None] | None] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config
        self.on_step = on_step

    async def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        ctx: ToolContext,
        episode_deadline: float | None = None,
    ) -> AgentLoopResult:
        messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_prompt),
        ]
        tool_records: list[ToolCallRecord] = []
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        model_calls = 0
        step = 0
        schemas = self.tools.schemas()

        try:
            while step < self.config.max_steps:
                if episode_deadline is not None and time.monotonic() >= episode_deadline:
                    return AgentLoopResult(
                        messages=messages,
                        tool_call_records=tool_records,
                        final_status=FinalStatus.TIMEOUT,
                        steps=step,
                        model_calls=model_calls,
                        stop_reason="timeout",
                        usage_totals=usage_totals,
                    )

                openai_messages = [m.to_openai_dict() for m in messages]
                response = await self.model.complete(
                    openai_messages,
                    tools=schemas or None,
                    tool_choice="auto",
                )
                model_calls += 1
                self._accumulate_usage(usage_totals, response)

                assistant = self._to_assistant_message(response)
                messages.append(assistant)

                if self.on_step:
                    maybe = self.on_step(step, assistant)
                    if maybe is not None:
                        await maybe

                if not response.tool_calls:
                    return AgentLoopResult(
                        messages=messages,
                        tool_call_records=tool_records,
                        final_status=FinalStatus.SUCCESS,  # provisional; verifier decides reward
                        steps=step,
                        model_calls=model_calls,
                        stop_reason="final_answer",
                        usage_totals=usage_totals,
                    )

                ctx.step = step
                tool_messages = await self.tools.execute_many(
                    response.tool_calls,
                    ctx,
                    parallel=self.config.parallel_tool_calls,
                )
                for tc, tm in zip(response.tool_calls, tool_messages, strict=True):
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {"_raw": tc.function.arguments}
                    is_error = bool(
                        tm.content and str(tm.content).startswith("ERROR:")
                    )
                    tool_records.append(
                        ToolCallRecord(
                            step=step,
                            tool_call_id=tc.id,
                            name=tc.function.name,
                            arguments=args if isinstance(args, dict) else {},
                            result=tm.content or "",
                            is_error=is_error,
                        )
                    )
                messages.extend(tool_messages)
                step += 1

            return AgentLoopResult(
                messages=messages,
                tool_call_records=tool_records,
                final_status=FinalStatus.MAX_STEPS,
                steps=step,
                model_calls=model_calls,
                stop_reason="max_steps",
                usage_totals=usage_totals,
            )
        except Exception as exc:
            return AgentLoopResult(
                messages=messages,
                tool_call_records=tool_records,
                final_status=FinalStatus.ERROR,
                steps=step,
                model_calls=model_calls,
                stop_reason="error",
                error=str(exc),
                usage_totals=usage_totals,
            )

    def _to_assistant_message(self, response: ModelResponse) -> Message:
        return Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls or None,
            reasoning_content=response.reasoning_content
            if self.config.include_thinking
            else None,
            raw=response.raw_choice,
        )

    @staticmethod
    def _accumulate_usage(
        totals: dict[str, int], response: ModelResponse
    ) -> None:
        if not response.usage:
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[key] = totals.get(key, 0) + int(response.usage.get(key, 0))
