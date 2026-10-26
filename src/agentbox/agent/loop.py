"""Multi-turn agent loop with OpenAI (or Hermes text) tool calling.

The produced conversation is training data, so the loop guarantees (GH #27-#33):

* every assistant ``tool_call`` is followed by exactly one tool message;
* model calls and tools are bounded by the *remaining* episode budget;
* context-window overflow is a scored policy failure, not an infra error;
* the wire payload is built incrementally (no O(n²) re-serialisation).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from agentbox.agent.protocols import hermes_tool_response, parse_hermes_tool_calls, render_hermes_tools
from agentbox.config import AgentConfig
from agentbox.errors import ContextLengthError, ModelError
from agentbox.model.base import ModelClient, ModelResponse
from agentbox.tools.base import ToolContext
from agentbox.tools.executor import ToolExecutor, ToolOutcome
from agentbox.trajectory.schema import Message, ToolCallRecord
from agentbox.trajectory.tokens import TokenTracker
from agentbox.types import FailureKind, FinalStatus, MessageRole, StopReason


class AgentLoopResult(BaseModel):
    messages: list[Message]
    tool_call_records: list[ToolCallRecord]
    final_status: FinalStatus
    steps: int
    model_calls: int
    stop_reason: str
    error: str | None = None
    usage_totals: dict[str, int] = Field(default_factory=dict)
    failure_kind: FailureKind = FailureKind.NONE
    submitted_answer: str | None = None


class AgentHooks:
    """Override any subset to observe or steer the loop (GH #38)."""

    async def before_model(self, step: int, messages: list[Message]) -> None:  # noqa: B027
        """Called before each model request."""

    async def after_model(self, step: int, message: Message) -> None:  # noqa: B027
        """Called with every assistant message."""

    async def after_tools(self, step: int, outcomes: list[ToolOutcome]) -> None:  # noqa: B027
        """Called after the tool calls of a turn ran."""


class Conversation:
    """Messages plus their wire dicts, kept in lockstep (each converted once)."""

    def __init__(
        self,
        *,
        include_reasoning: bool = False,
        hermes: bool = False,
        tool_response_role: str = "tool",
    ) -> None:
        self.messages: list[Message] = []
        self.wire: list[dict[str, Any]] = []
        self.include_reasoning = include_reasoning
        self.hermes = hermes
        self.tool_response_role = tool_response_role

    def add(self, message: Message) -> None:
        self.messages.append(message)
        role = message.role.value if isinstance(message.role, MessageRole) else message.role
        if self.hermes and role == "assistant":
            self.wire.append({"role": "assistant", "content": message.content or ""})
        elif self.hermes and role == "tool" and self.tool_response_role == "user":
            block = hermes_tool_response(message.content or "")
            last = self.wire[-1] if self.wire else None
            if last is not None and last.get("_tool_batch"):
                last["content"] += "\n" + block
            else:
                self.wire.append({"role": "user", "content": block, "_tool_batch": True})
        else:
            self.wire.append(message.to_openai_dict(include_reasoning=self.include_reasoning))

    def payload(self) -> list[dict[str, Any]]:
        if not self.hermes or self.tool_response_role != "user":
            return self.wire
        return [{k: v for k, v in m.items() if k != "_tool_batch"} for m in self.wire]

    def close_dangling(self, step: int, note: str) -> list[ToolCallRecord]:
        """Answer tool calls of the last assistant turn that have no tool message."""
        records: list[ToolCallRecord] = []
        for idx in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[idx]
            if msg.role in (MessageRole.ASSISTANT, "assistant"):
                answered = {
                    m.tool_call_id for m in self.messages[idx + 1 :] if m.tool_call_id is not None
                }
                for tc in msg.tool_calls or []:
                    if tc.id in answered:
                        continue
                    self.add(
                        Message(
                            role=MessageRole.TOOL,
                            content=f"ERROR: {note}",
                            tool_call_id=tc.id,
                            name=tc.function.name,
                        )
                    )
                    records.append(
                        ToolCallRecord(
                            step=step,
                            tool_call_id=tc.id,
                            name=tc.function.name,
                            arguments={"_raw": tc.function.arguments},
                            result=f"ERROR: {note}",
                            is_error=True,
                        )
                    )
                break
        return records


class AgentLoop:
    def __init__(
        self,
        model: ModelClient,
        tools: ToolExecutor,
        config: AgentConfig,
        *,
        on_step: Callable[[int, Message], Awaitable[None] | None] | None = None,
        hooks: AgentHooks | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config
        self.on_step = on_step
        self.hooks = hooks

    def _budget(self, deadline: float | None) -> float:
        budget = self.config.step_timeout_s
        if deadline is not None:
            budget = min(budget, deadline - time.monotonic())
        return budget

    async def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        ctx: ToolContext,
        episode_deadline: float | None = None,
    ) -> AgentLoopResult:
        cfg = self.config
        schemas = self.tools.schemas()
        hermes = cfg.tool_protocol == "hermes"
        if hermes and schemas:
            system_prompt = system_prompt.rstrip() + "\n\n" + render_hermes_tools(schemas)
        convo = Conversation(
            include_reasoning=cfg.send_reasoning,
            hermes=hermes,
            tool_response_role=cfg.tool_response_role,
        )
        convo.add(Message(role=MessageRole.SYSTEM, content=system_prompt))
        convo.add(Message(role=MessageRole.USER, content=user_prompt))
        records: list[ToolCallRecord] = []
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        model_calls = 0
        step = 0
        ctx.deadline = episode_deadline
        tokens = TokenTracker()

        def finish(
            status: FinalStatus,
            reason: StopReason,
            *,
            error: str | None = None,
            failure: FailureKind = FailureKind.NONE,
            answer: str | None = None,
        ) -> AgentLoopResult:
            records.extend(convo.close_dangling(step, error or reason.value))
            return AgentLoopResult(
                messages=convo.messages,
                tool_call_records=records,
                final_status=status,
                steps=step,
                model_calls=model_calls,
                stop_reason=reason.value,
                error=error,
                usage_totals=usage_totals,
                failure_kind=failure,
                submitted_answer=answer,
            )

        try:
            while step < cfg.max_steps:
                budget = self._budget(episode_deadline)
                if budget <= 0:
                    return finish(
                        FinalStatus.TIMEOUT, StopReason.TIMEOUT, failure=FailureKind.POLICY
                    )
                if self.hooks is not None:
                    await self.hooks.before_model(step, convo.messages)
                t0 = time.monotonic()
                try:
                    response = await asyncio.wait_for(
                        self.model.complete(
                            convo.payload(),
                            tools=None if hermes else (schemas or None),
                            tool_choice=None,
                        ),
                        timeout=budget,
                    )
                except TimeoutError:
                    return finish(
                        FinalStatus.TIMEOUT,
                        StopReason.TIMEOUT,
                        error=f"model call timed out after {budget:.0f}s",
                        failure=FailureKind.POLICY,
                    )
                except ContextLengthError as exc:
                    return finish(
                        FinalStatus.FAILED,
                        StopReason.CONTEXT_LENGTH,
                        error=str(exc),
                        failure=FailureKind.POLICY,
                    )
                except ModelError as exc:
                    return finish(
                        FinalStatus.ERROR,
                        StopReason.ERROR,
                        error=str(exc),
                        failure=FailureKind.MODEL_API,
                    )
                model_calls += 1
                self._accumulate_usage(usage_totals, response)
                assistant = self._to_assistant_message(response, time.monotonic() - t0)
                assistant.tokens = tokens.observe(
                    response.prompt_token_ids,
                    response.completion_token_ids,
                    response.completion_logprobs,
                )
                if hermes and not assistant.tool_calls:
                    _, parsed = parse_hermes_tool_calls(assistant.content)
                    assistant.tool_calls = parsed or None
                convo.add(assistant)

                if self.on_step:
                    maybe = self.on_step(step, assistant)
                    if maybe is not None:
                        await maybe
                if self.hooks is not None:
                    await self.hooks.after_model(step, assistant)

                if not assistant.tool_calls:
                    if response.finish_reason == "length":
                        return finish(
                            FinalStatus.FAILED,
                            StopReason.CONTEXT_LENGTH,
                            error="generation truncated (finish_reason=length)",
                            failure=FailureKind.POLICY,
                        )
                    return finish(FinalStatus.SUCCESS, StopReason.FINAL_ANSWER)

                ctx.step = step
                outcomes = await self.tools.run_calls(
                    assistant.tool_calls,
                    ctx,
                    deadline=episode_deadline,
                    parallel=cfg.parallel_tool_calls,
                )
                submitted: str | None = None
                was_submitted = False
                for outcome in outcomes:
                    convo.add(outcome.message)
                    records.append(outcome.record)
                    if outcome.terminal:
                        was_submitted = True
                        submitted = outcome.meta.get("answer")
                if self.hooks is not None:
                    await self.hooks.after_tools(step, outcomes)
                step += 1
                if was_submitted:
                    return finish(FinalStatus.SUCCESS, StopReason.SUBMITTED, answer=submitted)

            return finish(FinalStatus.MAX_STEPS, StopReason.MAX_STEPS, failure=FailureKind.POLICY)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return finish(
                FinalStatus.ERROR,
                StopReason.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                failure=FailureKind.INTERNAL,
            )

    def _to_assistant_message(self, response: ModelResponse, latency_s: float) -> Message:
        return Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls or None,
            reasoning_content=response.reasoning_content if self.config.include_thinking else None,
            raw=response.raw_choice,
            finish_reason=response.finish_reason,
            usage=response.usage,
            latency_s=round(latency_s, 4),
        )

    @staticmethod
    def _accumulate_usage(totals: dict[str, int], response: ModelResponse) -> None:
        if not response.usage:
            return
        for key, value in response.usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
