"""Single-rollout orchestrator.

Phases and their failure attribution (GH #40)::

    provision       -> SANDBOX | SEED | SETUP_CHECK   (infra)
    agent loop      -> POLICY | MODEL_API | INTERNAL
    verify          -> VERIFIER (infra) when the verifier itself cannot run

Only ``none``/``policy`` outcomes are trainable; runners retry the rest.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from agentbox.agent.agent import Agent
from agentbox.agent.loop import AgentHooks, AgentLoop, AgentLoopResult
from agentbox.agent.prompts import render_system_prompt
from agentbox.config import AgentConfig, ModelConfig, RolloutConfig, SandboxConfig
from agentbox.model.base import ModelClient
from agentbox.model.openai_compat import OpenAICompatClient
from agentbox.rewards import RewardContext, compute_reward, default_reward_specs
from agentbox.runner.provision import (
    Provisioned,
    ProvisionError,
    effective_sandbox,
    manager_for,
    provision,
    release,
)
from agentbox.sandbox.manager import SandboxManager
from agentbox.tasks.schema import Task
from agentbox.tasks.verifier import Verifier, VerifyResult
from agentbox.tito import SESSION_KEY
from agentbox.tools.base import ToolContext
from agentbox.tools.executor import ToolExecutor
from agentbox.tools.registry import ToolRegistry, build_tool_registry
from agentbox.trajectory.schema import Message, Trajectory, TrajectoryMetrics, VerifyInfo
from agentbox.types import FailureKind, FinalStatus, StopReason, ToolMode
from agentbox.utils.text import truncate_middle

__all__ = [
    "Rollout",
    "build_tools",
    "derive_seed",
    "effective_sandbox",
    "resolve_agent_config",
    "resolve_model",
    "task_hash",
    "verify_info",
]

logger = logging.getLogger("agentbox.rollout")

VERIFY_LOG_CHARS = 16_384


def derive_seed(seed: int | None, task_id: str, sample_index: int) -> int | None:
    """Deterministic per-sample seed (distinct across GRPO group members)."""
    if seed is None:
        return None
    digest = hashlib.sha256(f"{seed}:{task_id}:{sample_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def task_hash(task: Task) -> str:
    return hashlib.sha256(task.model_dump_json().encode()).hexdigest()[:16]


def resolve_model(
    model: ModelClient | ModelConfig | str | None,
    agent: Agent | AgentConfig | dict[str, Any] | None,
    config: RolloutConfig | None,
) -> tuple[ModelClient, bool]:
    """Return ``(client, owned)``; owned clients must be closed by the caller."""
    if model is not None and not isinstance(model, (str, ModelConfig)):
        return model, False
    if isinstance(model, ModelConfig):
        return OpenAICompatClient(model), True
    if isinstance(model, str):
        return OpenAICompatClient(ModelConfig(model=model)), True
    if isinstance(agent, Agent):
        return agent.model_client, False
    if config is not None:
        return OpenAICompatClient(config.model), True
    raise ValueError("Must provide model, agent with model, or RolloutConfig.model")


def resolve_agent_config(
    agent: Agent | AgentConfig | dict[str, Any] | None,
    config: RolloutConfig | None,
    task: Task,
) -> AgentConfig:
    if isinstance(agent, Agent):
        cfg = agent.config.model_copy(deep=True)
    elif isinstance(agent, AgentConfig):
        cfg = agent.model_copy(deep=True)
    elif isinstance(agent, dict):
        base = config.agent if config else AgentConfig()
        cfg = base.model_copy(update=agent)
    elif config is not None:
        cfg = config.agent.model_copy(deep=True)
    else:
        cfg = AgentConfig()
    if task.max_steps is not None:
        cfg.max_steps = task.max_steps
    if task.allowed_tools is not None:
        cfg.tools = task.allowed_tools
    if task.episode_timeout_s is not None:
        cfg.episode_timeout_s = task.episode_timeout_s
    return cfg


def build_tools(agent_cfg: AgentConfig, seed: int | None) -> tuple[ToolRegistry, ToolExecutor]:
    registry = build_tool_registry(
        agent_cfg.tools,
        custom_tools=agent_cfg.custom_tools,
        include_builtins=agent_cfg.builtins,
        drop=agent_cfg.drop_tools,
        drop_prob=agent_cfg.drop_tools_prob,
        rng=random.Random(seed),
    )
    executor = ToolExecutor(
        registry,
        max_output_chars=agent_cfg.max_tool_output_chars,
        tool_timeout_s=agent_cfg.tool_timeout_s,
        parallel_read_only=agent_cfg.parallel_tool_calls,
    )
    return registry, executor


def tool_mode_label(mode: ToolMode | list[str] | str) -> str:
    if isinstance(mode, ToolMode):
        return mode.value
    if isinstance(mode, list):
        return ",".join(mode)
    return str(mode)


def verify_info(verify: VerifyResult) -> VerifyInfo:
    return VerifyInfo(
        success=verify.success,
        reward=verify.reward,
        exit_code=verify.exit_code,
        command=verify.command,
        duration_s=verify.duration_s,
        timed_out=verify.timed_out,
        stdout=truncate_middle(verify.stdout or "", VERIFY_LOG_CHARS),
        stderr=truncate_middle(verify.stderr or "", VERIFY_LOG_CHARS // 2),
        tests_passed=verify.tests_passed,
        tests_failed=verify.tests_failed,
        tests_total=verify.tests_total,
        tampered=list(verify.tampered),
    )


def loop_metrics(metrics: TrajectoryMetrics, loop: AgentLoopResult) -> None:
    records = loop.tool_call_records
    usage = loop.usage_totals
    metrics.steps = loop.steps
    metrics.model_calls = loop.model_calls
    metrics.tool_calls = len(records)
    metrics.tool_errors = sum(1 for r in records if r.is_error)
    metrics.prompt_tokens = usage.get("prompt_tokens")
    metrics.completion_tokens = usage.get("completion_tokens")
    metrics.total_tokens = usage.get("total_tokens")
    metrics.cached_tokens = usage.get("cached_tokens")
    metrics.tool_latency_s = round(sum(r.duration_s for r in records), 4)
    metrics.model_latency_s = round(sum(m.latency_s or 0.0 for m in loop.messages), 4)


def final_status_for(
    verify: VerifyResult, loop: AgentLoopResult
) -> tuple[FinalStatus, FailureKind]:
    if verify.success:
        return FinalStatus.SUCCESS, FailureKind.NONE
    status = {
        StopReason.TIMEOUT.value: FinalStatus.TIMEOUT,
        StopReason.MAX_STEPS.value: FinalStatus.MAX_STEPS,
        StopReason.ERROR.value: FinalStatus.ERROR,
    }.get(loop.stop_reason, FinalStatus.FAILED)
    failure = loop.failure_kind if loop.failure_kind.is_infra else FailureKind.POLICY
    return status, failure


class Rollout:
    """Run one task in an isolated sandbox and return a :class:`Trajectory`."""

    @staticmethod
    async def run(
        task: Task,
        *,
        model: ModelClient | ModelConfig | str | None = None,
        agent: Agent | AgentConfig | dict[str, Any] | None = None,
        sandbox: SandboxConfig | None = None,
        config: RolloutConfig | None = None,
        manager: SandboxManager | None = None,
        setup_checks: list[Any] | None = None,
        run_id: str | None = None,
        rollout_id: str | None = None,
        group_id: str | None = None,
        sample_index: int = 0,
        hooks: AgentHooks | None = None,
    ) -> Trajectory:
        model_client, owned = resolve_model(model, agent, config)
        try:
            return await _run(
                task,
                model_client=model_client,
                agent=agent,
                sandbox=sandbox,
                config=config,
                manager=manager,
                setup_checks=setup_checks,
                run_id=run_id,
                rollout_id=rollout_id,
                group_id=group_id,
                sample_index=sample_index,
                hooks=hooks,
            )
        finally:
            if owned:
                close = getattr(model_client, "aclose", None)
                if close is not None:
                    await close()


async def _run(
    task: Task,
    *,
    model_client: ModelClient,
    agent: Agent | AgentConfig | dict[str, Any] | None,
    sandbox: SandboxConfig | None,
    config: RolloutConfig | None,
    manager: SandboxManager | None,
    setup_checks: list[Any] | None,
    run_id: str | None,
    rollout_id: str | None,
    group_id: str | None,
    sample_index: int,
    hooks: AgentHooks | None,
) -> Trajectory:
    t_start = time.monotonic()
    created_at = datetime.now(UTC)
    run_id = run_id or (config.run_id if config else None) or uuid.uuid4().hex
    rollout_id = rollout_id or uuid.uuid4().hex
    run_sandbox = (
        sandbox
        or (manager.config if manager else None)
        or (config.sandbox if config else SandboxConfig())
    )
    mgr = manager_for(task, run_sandbox, manager)
    sandbox_cfg = mgr.config
    agent_cfg = resolve_agent_config(agent, config, task)
    reward_specs = (config.rewards if config else None) or default_reward_specs(agent_cfg)
    sample_seed = derive_seed(config.seed if config else None, task.task_id, sample_index)
    registry, executor = build_tools(agent_cfg, sample_seed)
    metrics = TrajectoryMetrics()
    base: dict[str, Any] = {
        "task_id": task.task_id,
        "run_id": run_id,
        "rollout_id": rollout_id,
        "group_id": group_id,
        "sample_index": sample_index,
        "model": getattr(model_client, "model", None),
        "tool_mode": tool_mode_label(agent_cfg.tools),
        "tools": executor.schemas(),
        "task_hash": task_hash(task),
        "created_at": created_at,
    }
    meta: dict[str, Any] = {"task_metadata": task.metadata, "tools_available": registry.names()}
    if task.split is not None:
        meta["split"] = task.split
    if task.group_key is not None:
        meta["group_key"] = task.group_key
    built: list[Trajectory] = []

    def build(
        *,
        status: FinalStatus,
        failure: FailureKind,
        reward: float = 0.0,
        breakdown: dict[str, float] | None = None,
        messages: list[Message] | None = None,
        loop: AgentLoopResult | None = None,
        verify: VerifyResult | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Trajectory:
        metrics.duration_s = time.monotonic() - t_start
        if loop is not None:
            loop_metrics(metrics, loop)
        traj = Trajectory(
            **base,
            messages=messages if messages is not None else (loop.messages if loop else []),
            tool_call_records=loop.tool_call_records if loop else [],
            reward=reward,
            reward_breakdown=breakdown or {},
            final_status=status,
            stop_reason=loop.stop_reason if loop else None,
            failure_kind=failure,
            verify=verify_info(verify) if verify is not None else None,
            metrics=metrics,
            metadata={**meta, **(extra or {})},
            error=error,
            finished_at=datetime.now(UTC),
        )
        built.append(traj)
        return traj

    prov: Provisioned | None = None
    outcome: FinalStatus | None = None
    try:
        try:
            prov = await provision(
                task,
                manager=mgr,
                run_id=run_id,
                rollout_id=rollout_id,
                group_id=group_id,
                setup_checks=setup_checks,
            )
        except ProvisionError as exc:
            logger.warning("provisioning %s failed (%s): %s", task.task_id, exc.kind.value, exc)
            outcome = FinalStatus.ERROR
            return build(
                status=FinalStatus.ERROR, failure=exc.kind, error=str(exc), extra=exc.extra
            )
        metrics.sandbox_create_s = prov.create_s
        metrics.seed_s = prov.seed_s

        system = render_system_prompt(
            workspace_dir=sandbox_cfg.workspace_dir,
            mode=agent_cfg.tools,
            extra=task.system_prompt_extra,
            override=agent_cfg.system_prompt,
        )
        ctx = ToolContext(
            sandbox=prov.sandbox,
            manager=mgr,
            workspace_dir=sandbox_cfg.workspace_dir,
            run_id=run_id,
            task_id=task.task_id,
            tool_timeout_s=agent_cfg.tool_timeout_s,
            max_output_chars=agent_cfg.max_tool_output_chars,
        )
        loop = AgentLoop(
            model_client,
            executor,
            agent_cfg,
            hooks=hooks,
            model_kwargs={"seed": sample_seed} if sample_seed is not None else None,
        )
        session = SESSION_KEY.set(rollout_id)  # per-rollout state for TITO clients
        try:
            loop_result = await loop.run(
                system_prompt=system,
                user_prompt=task.prompt_text(),
                ctx=ctx,
                episode_deadline=time.monotonic() + agent_cfg.episode_timeout_s,
            )
        finally:
            SESSION_KEY.reset(session)
            end = getattr(getattr(model_client, "inner", model_client), "end_session", None)
            if end is not None:
                end(rollout_id)

        t_verify = time.monotonic()
        try:
            verify_result = await Verifier(mgr).verify(prov.sandbox, task.verifier, task)
        except Exception as exc:
            logger.warning("verifier crashed for %s: %s", task.task_id, exc)
            metrics.verify_s = time.monotonic() - t_verify
            outcome = FinalStatus.ERROR
            return build(
                status=FinalStatus.ERROR,
                failure=FailureKind.VERIFIER,
                loop=loop_result,
                error=f"verifier failed: {exc}",
            )
        metrics.verify_s = time.monotonic() - t_verify

        reward, breakdown = await compute_reward(
            reward_specs, RewardContext(task=task, verify=verify_result, loop=loop_result)
        )
        status, failure = final_status_for(verify_result, loop_result)
        outcome = status
        extra = (
            {"submitted_answer": loop_result.submitted_answer}
            if loop_result.submitted_answer
            else None
        )
        return build(
            status=status,
            failure=failure,
            reward=reward,
            breakdown=breakdown,
            loop=loop_result,
            verify=verify_result,
            error=loop_result.error,
            extra=extra,
        )
    except Exception as exc:
        logger.exception("rollout failed task_id=%s", task.task_id)
        outcome = FinalStatus.ERROR
        return build(status=FinalStatus.ERROR, failure=FailureKind.INTERNAL, error=str(exc))
    finally:
        if prov is not None:
            keep = not sandbox_cfg.auto_remove or (
                sandbox_cfg.keep_on_failure and outcome != FinalStatus.SUCCESS
            )
            await release(prov, keep=keep)
            if keep and built:  # `agentbox sandbox shell <name>` to inspect it
                built[-1].metadata["kept_sandbox"] = prov.sandbox.name
