"""Step-wise environment API (GH #70).

For trainer-driven loops — the trainer owns generation (TRL, verl, OpenRLHF,
SkyRL, custom); AgentBox owns the sandbox, tools and reward::

    env = AgentBoxEnv(sandbox=SandboxConfig(...))
    obs = await env.reset(task)                  # messages + tool schemas
    while True:
        text = my_policy(obs.messages, obs.tools)
        res = await env.step(text)               # assistant text, message dict or tool calls
        if res.done:
            break
    result = await env.evaluate()                # verifier + reward components
    traj = env.trajectory()                      # same schema as Rollout
    await env.close()
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from agentbox.agent.loop import AgentLoopResult, Conversation
from agentbox.agent.prompts import render_system_prompt
from agentbox.agent.protocols import parse_hermes_tool_calls, render_hermes_tools
from agentbox.config import AgentConfig, SandboxConfig
from agentbox.rewards import RewardContext, RewardSpec, compute_reward, default_reward_specs
from agentbox.runner.provision import Provisioned, manager_for, provision, release
from agentbox.runner.rollout import (
    build_tools,
    final_status_for,
    resolve_agent_config,
    task_hash,
    tool_mode_label,
    verify_info,
)
from agentbox.sandbox.manager import SandboxManager
from agentbox.tasks.schema import Task
from agentbox.tasks.verifier import Verifier, VerifyResult
from agentbox.tools.base import ToolContext
from agentbox.tools.executor import ToolExecutor, ToolOutcome
from agentbox.trajectory.schema import (
    FunctionCall,
    Message,
    ToolCall,
    ToolCallRecord,
    Trajectory,
    TrajectoryMetrics,
    VerifyInfo,
)
from agentbox.types import FailureKind, FinalStatus, MessageRole, StopReason
from agentbox.utils.aio import BackgroundLoop, shared_loop


class Observation(BaseModel):
    task_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    system_prompt: str
    prompt: str


class StepResult(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    stop_reason: str | None = None
    tool_calls: int = 0
    info: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    reward: float
    breakdown: dict[str, float] = Field(default_factory=dict)
    success: bool
    final_status: FinalStatus
    failure_kind: FailureKind
    verify: VerifyInfo | None = None
    error: str | None = None


class AgentBoxEnv:
    """One sandboxed episode at a time; reusable across ``reset`` calls."""

    def __init__(
        self,
        *,
        sandbox: SandboxConfig | None = None,
        agent: AgentConfig | None = None,
        rewards: list[RewardSpec] | None = None,
        manager: SandboxManager | None = None,
        run_id: str | None = None,
        setup_checks: list[Any] | None = None,
    ) -> None:
        self.sandbox_config = sandbox or (manager.config if manager else SandboxConfig())
        self.agent_config = agent or AgentConfig()
        self.rewards = rewards
        self.manager = manager
        self.run_id = run_id or uuid.uuid4().hex
        self.setup_checks = setup_checks
        self._prov: Provisioned | None = None
        self._task: Task | None = None
        self._executor: ToolExecutor | None = None
        self._ctx: ToolContext | None = None
        self._convo: Conversation | None = None
        self._cfg = self.agent_config
        self._records: list[ToolCallRecord] = []
        self._steps = 0
        self._stop_reason: str | None = None
        self._deadline: float | None = None
        self._ids: dict[str, Any] = {}
        self._metrics = TrajectoryMetrics()
        self._evaluation: EvalResult | None = None
        self._verify: VerifyResult | None = None
        self._t0 = 0.0
        self._created_at = datetime.now(UTC)

    # ----------------------------------------------------------------- state
    @property
    def task(self) -> Task:
        if self._task is None:
            raise RuntimeError("call reset() first")
        return self._task

    @property
    def done(self) -> bool:
        return self._stop_reason is not None

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._executor.schemas() if self._executor else []

    @property
    def messages(self) -> list[Message]:
        return self._convo.messages if self._convo else []

    @property
    def sandbox(self) -> Any:
        if self._prov is None:
            raise RuntimeError("call reset() first")
        return self._prov.sandbox

    # --------------------------------------------------------------- episode
    async def reset(
        self,
        task: Task,
        *,
        group_id: str | None = None,
        sample_index: int = 0,
        rollout_id: str | None = None,
        seed: int | None = None,
    ) -> Observation:
        await self.close()
        mgr = manager_for(task, self.sandbox_config, self.manager)
        cfg = resolve_agent_config(self.agent_config, None, task)
        registry, executor = build_tools(cfg, seed)
        rid = rollout_id or uuid.uuid4().hex
        self._t0 = time.monotonic()
        self._created_at = datetime.now(UTC)
        self._metrics = TrajectoryMetrics()
        self._prov = await provision(
            task,
            manager=mgr,
            run_id=self.run_id,
            rollout_id=rid,
            group_id=group_id,
            setup_checks=self.setup_checks,
        )
        self._metrics.sandbox_create_s = self._prov.create_s
        self._metrics.seed_s = self._prov.seed_s
        self._task, self._cfg, self._executor = task, cfg, executor
        self._records, self._steps, self._stop_reason = [], 0, None
        self._evaluation, self._verify = None, None
        self._ids = {
            "rollout_id": rid,
            "group_id": group_id,
            "sample_index": sample_index,
            "tools_available": registry.names(),
        }
        hermes = cfg.tool_protocol == "hermes"
        system = render_system_prompt(
            workspace_dir=mgr.config.workspace_dir,
            mode=cfg.tools,
            extra=task.system_prompt_extra,
            override=cfg.system_prompt,
        )
        if hermes and executor.schemas():
            system = system.rstrip() + "\n\n" + render_hermes_tools(executor.schemas())
        self._convo = Conversation(
            include_reasoning=cfg.send_reasoning,
            hermes=hermes,
            tool_response_role=cfg.tool_response_role,
        )
        prompt = task.prompt_text()
        self._convo.add(Message(role=MessageRole.SYSTEM, content=system))
        self._convo.add(Message(role=MessageRole.USER, content=prompt))
        self._ctx = ToolContext(
            sandbox=self._prov.sandbox,
            manager=mgr,
            workspace_dir=mgr.config.workspace_dir,
            run_id=self.run_id,
            task_id=task.task_id,
            tool_timeout_s=cfg.tool_timeout_s,
            max_output_chars=cfg.max_tool_output_chars,
        )
        self._deadline = time.monotonic() + cfg.episode_timeout_s
        self._ctx.deadline = self._deadline
        return Observation(
            task_id=task.task_id,
            messages=list(self._convo.payload()),
            tools=executor.schemas(),
            system_prompt=system,
            prompt=prompt,
        )

    def _assistant_message(self, action: Any) -> Message:
        if isinstance(action, Message):
            msg = action
        elif isinstance(action, str):
            msg = Message(role=MessageRole.ASSISTANT, content=action)
        elif isinstance(action, dict):
            msg = Message.model_validate({**action, "role": "assistant"})
        elif isinstance(action, list):
            msg = Message(role=MessageRole.ASSISTANT, content=None, tool_calls=list(action))
        else:
            raise TypeError(f"unsupported action type {type(action)!r}")
        if not msg.tool_calls and msg.content and "<tool_call>" in msg.content:
            _, calls = parse_hermes_tool_calls(msg.content)
            msg.tool_calls = calls or None
        return msg

    def _finish(self, reason: StopReason) -> None:
        if self._stop_reason is None:
            self._stop_reason = reason.value

    async def step(self, action: str | dict[str, Any] | Message | list[ToolCall]) -> StepResult:
        """Record an assistant turn and execute its tool calls."""
        if self._convo is None or self._executor is None or self._ctx is None:
            raise RuntimeError("call reset() first")
        if self.done:
            return StepResult(done=True, stop_reason=self._stop_reason)
        msg = self._assistant_message(action)
        self._convo.add(msg)
        if not msg.tool_calls:
            self._finish(StopReason.FINAL_ANSWER)
            return StepResult(done=True, stop_reason=self._stop_reason)
        self._ctx.step = self._steps
        start = len(self._convo.wire)
        outcomes = await self._executor.run_calls(
            msg.tool_calls,
            self._ctx,
            deadline=self._deadline,
            parallel=self._cfg.parallel_tool_calls,
        )
        terminal = False
        for outcome in outcomes:
            self._convo.add(outcome.message)
            self._records.append(outcome.record)
            terminal = terminal or outcome.terminal
        self._steps += 1
        if terminal:
            self._finish(StopReason.SUBMITTED)
        elif self._steps >= self._cfg.max_steps:
            self._finish(StopReason.MAX_STEPS)
        elif self._deadline is not None and time.monotonic() >= self._deadline:
            self._finish(StopReason.TIMEOUT)
        new = [{k: v for k, v in m.items() if k != "_tool_batch"} for m in self._convo.wire[start:]]
        return StepResult(
            messages=new,
            done=self.done,
            stop_reason=self._stop_reason,
            tool_calls=len(outcomes),
            info={"is_error": [o.record.is_error for o in outcomes]},
        )

    async def call_tool(
        self, name: str, arguments: str | dict[str, Any], *, call_id: str | None = None
    ) -> ToolOutcome:
        """Execute one tool call for loops that parse tool calls themselves (verl)."""
        if self._executor is None or self._ctx is None:
            raise RuntimeError("call reset() first")
        args = arguments if isinstance(arguments, str) else json.dumps(arguments)
        tc = ToolCall(
            id=call_id or f"call_{uuid.uuid4().hex[:24]}",
            function=FunctionCall(name=name, arguments=args),
        )
        self._ctx.step = self._steps
        outcome = await self._executor.run_call(tc, self._ctx, deadline=self._deadline)
        self._records.append(outcome.record)
        if outcome.terminal:
            self._finish(StopReason.SUBMITTED)
        return outcome

    def end_turn(self) -> None:
        """Count an assistant turn when tools are executed via :meth:`call_tool`."""
        self._steps += 1
        if self._steps >= self._cfg.max_steps:
            self._finish(StopReason.MAX_STEPS)

    def _loop_result(self) -> AgentLoopResult:
        return AgentLoopResult(
            messages=self.messages,
            tool_call_records=self._records,
            final_status=FinalStatus.SUCCESS,
            steps=self._steps,
            model_calls=sum(
                1 for m in self.messages if m.role in (MessageRole.ASSISTANT, "assistant")
            ),
            stop_reason=self._stop_reason or StopReason.FINAL_ANSWER.value,
            failure_kind=FailureKind.NONE,
        )

    async def evaluate(self) -> EvalResult:
        """Run the verifier and reward components for the current episode."""
        if self._prov is None or self._task is None:
            raise RuntimeError("call reset() first")
        if self._evaluation is not None:
            return self._evaluation
        loop = self._loop_result()
        t0 = time.monotonic()
        try:
            verify = await Verifier(self._prov.manager).verify(
                self._prov.sandbox, self._task.verifier, self._task
            )
        except Exception as exc:
            self._metrics.verify_s = time.monotonic() - t0
            self._evaluation = EvalResult(
                reward=0.0,
                success=False,
                final_status=FinalStatus.ERROR,
                failure_kind=FailureKind.VERIFIER,
                error=f"verifier failed: {exc}",
            )
            return self._evaluation
        self._metrics.verify_s = time.monotonic() - t0
        specs = self.rewards or default_reward_specs(self._cfg)
        reward, breakdown = await compute_reward(
            specs, RewardContext(task=self._task, verify=verify, loop=loop)
        )
        status, failure = final_status_for(verify, loop)
        self._verify = verify
        self._evaluation = EvalResult(
            reward=reward,
            breakdown=breakdown,
            success=verify.success,
            final_status=status,
            failure_kind=failure,
            verify=verify_info(verify),
        )
        return self._evaluation

    def trajectory(self, *, model: str | None = None) -> Trajectory:
        """The episode as a :class:`Trajectory` (call after :meth:`evaluate`)."""
        task = self.task
        ev = self._evaluation
        loop = self._loop_result()
        m = self._metrics
        m.steps = loop.steps
        m.model_calls = loop.model_calls
        m.tool_calls = len(self._records)
        m.tool_errors = sum(1 for r in self._records if r.is_error)
        m.tool_latency_s = round(sum(r.duration_s for r in self._records), 4)
        m.duration_s = time.monotonic() - self._t0
        return Trajectory(
            task_id=task.task_id,
            run_id=self.run_id,
            rollout_id=self._ids.get("rollout_id", ""),
            group_id=self._ids.get("group_id"),
            sample_index=self._ids.get("sample_index", 0),
            messages=list(self.messages),
            tool_call_records=list(self._records),
            reward=ev.reward if ev else 0.0,
            reward_breakdown=ev.breakdown if ev else {},
            final_status=ev.final_status if ev else FinalStatus.FAILED,
            stop_reason=self._stop_reason,
            failure_kind=ev.failure_kind if ev else FailureKind.POLICY,
            verify=ev.verify if ev else None,
            metrics=m,
            metadata={
                "task_metadata": task.metadata,
                "tools_available": self._ids.get("tools_available", []),
            },
            error=ev.error if ev else None,
            model=model,
            tool_mode=tool_mode_label(self._cfg.tools),
            tools=self.tools,
            task_hash=task_hash(task),
            created_at=self._created_at,
            finished_at=datetime.now(UTC),
        )

    async def close(self) -> None:
        if self._prov is not None:
            prov, self._prov = self._prov, None
            await release(prov, keep=False)

    async def __aenter__(self) -> AgentBoxEnv:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


class SyncAgentBoxEnv:
    """Blocking facade over :class:`AgentBoxEnv` for synchronous trainers."""

    def __init__(self, *args: Any, loop: BackgroundLoop | None = None, **kwargs: Any) -> None:
        self._loop = loop or shared_loop()
        self.env = AgentBoxEnv(*args, **kwargs)

    def reset(self, task: Task, **kw: Any) -> Observation:
        return self._loop.run(self.env.reset(task, **kw))

    def step(self, action: Any) -> StepResult:
        return self._loop.run(self.env.step(action))

    def call_tool(self, name: str, arguments: str | dict[str, Any], **kw: Any) -> ToolOutcome:
        return self._loop.run(self.env.call_tool(name, arguments, **kw))

    def evaluate(self) -> EvalResult:
        return self._loop.run(self.env.evaluate())

    def trajectory(self, **kw: Any) -> Trajectory:
        return self.env.trajectory(**kw)

    def close(self) -> None:
        self._loop.run(self.env.close())

    @property
    def done(self) -> bool:
        return self.env.done

    def __enter__(self) -> SyncAgentBoxEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
