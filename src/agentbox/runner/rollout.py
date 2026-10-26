"""Single-rollout orchestrator.

Phases and their failure attribution (GH #40)::

    create sandbox  -> FailureKind.SANDBOX      (infra)
    seed task       -> FailureKind.SEED         (infra / broken task)
    setup checks    -> FailureKind.SETUP_CHECK  (infra)
    agent loop      -> POLICY | MODEL_API | INTERNAL
    verify          -> FailureKind.VERIFIER     (infra) if the verifier cannot run

Only ``none``/``policy`` outcomes are trainable; runners retry the rest.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from agentbox.agent.agent import Agent
from agentbox.agent.loop import AgentHooks, AgentLoop, AgentLoopResult
from agentbox.agent.prompts import render_system_prompt
from agentbox.config import AgentConfig, ModelConfig, RolloutConfig, SandboxConfig
from agentbox.model.base import ModelClient
from agentbox.model.openai_compat import OpenAICompatClient
from agentbox.sandbox.manager import SandboxManager
from agentbox.sandbox.types import Sandbox
from agentbox.rewards import RewardContext, compute_reward, default_reward_specs
from agentbox.tasks.schema import Task
from agentbox.tasks.seeder import TaskSeeder
from agentbox.tasks.verifier import Verifier, VerifyResult
from agentbox.tools.base import ToolContext
from agentbox.tools.executor import ToolExecutor
from agentbox.tools.registry import build_tool_registry
from agentbox.trajectory.schema import Message, Trajectory, TrajectoryMetrics, VerifyInfo
from agentbox.types import FailureKind, FinalStatus, StopReason, ToolMode
from agentbox.utils.text import truncate_middle

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


def effective_sandbox(base: SandboxConfig, task: Task) -> SandboxConfig:
    """Merge a task's environment overrides over the run's sandbox config."""
    env = task.environment
    if env is None:
        return base
    update: dict[str, Any] = {}
    if env.image:
        update["image"] = env.image
    if env.env:
        update["env"] = {**base.env, **env.env}
    if env.working_dir:
        update["working_dir"] = env.working_dir
        update["workspace_dir"] = env.working_dir
    if env.ensure_pytest is not None:
        update["ensure_pytest"] = env.ensure_pytest
    limits: dict[str, Any] = {}
    if env.network is not None:
        limits["network_disabled"] = not env.network
    if env.cpu_count is not None:
        limits["cpu_count"] = env.cpu_count
    if env.memory_mb is not None:
        limits["memory_mb"] = env.memory_mb
    if limits:
        update["limits"] = base.limits.model_copy(update=limits)
    return base.model_copy(update=update) if update else base


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
    episode_timeout = getattr(task, "episode_timeout_s", None)
    if episode_timeout is not None:
        cfg.episode_timeout_s = episode_timeout
    return cfg


def _tool_mode_label(mode: ToolMode | list[str] | str) -> str:
    if isinstance(mode, ToolMode):
        return mode.value
    if isinstance(mode, list):
        return ",".join(mode)
    return str(mode)


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
    created_at = datetime.now(timezone.utc)
    run_id = run_id or (config.run_id if config else None) or uuid.uuid4().hex
    rollout_id = rollout_id or uuid.uuid4().hex
    run_sandbox = sandbox or (manager.config if manager else None) or (
        config.sandbox if config else SandboxConfig()
    )
    sandbox_cfg = effective_sandbox(run_sandbox, task)
    agent_cfg = resolve_agent_config(agent, config, task)
    if manager is not None and sandbox_cfg is manager.config:
        mgr = manager
    else:
        mgr = SandboxManager(sandbox_cfg, backend=manager.backend if manager else None)
    reward_specs = (config.rewards if config else None) or default_reward_specs(agent_cfg)
    base_seed = config.seed if config else None
    sample_seed = derive_seed(base_seed, task.task_id, sample_index)

    registry = build_tool_registry(
        agent_cfg.tools,
        custom_tools=agent_cfg.custom_tools,
        include_builtins=agent_cfg.builtins,
        drop=agent_cfg.drop_tools,
        drop_prob=agent_cfg.drop_tools_prob,
        rng=random.Random(sample_seed),
    )
    executor = ToolExecutor(
        registry,
        max_output_chars=agent_cfg.max_tool_output_chars,
        tool_timeout_s=agent_cfg.tool_timeout_s,
        parallel_read_only=agent_cfg.parallel_tool_calls,
    )
    metrics = TrajectoryMetrics()
    base: dict[str, Any] = {
        "task_id": task.task_id,
        "run_id": run_id,
        "rollout_id": rollout_id,
        "group_id": group_id,
        "sample_index": sample_index,
        "model": getattr(model_client, "model", None),
        "tool_mode": _tool_mode_label(agent_cfg.tools),
        "tools": executor.schemas(),
        "task_hash": task_hash(task),
        "created_at": created_at,
    }
    meta: dict[str, Any] = {"task_metadata": task.metadata, "tools_available": registry.names()}

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
        records = loop.tool_call_records if loop else []
        if loop is not None:
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
            metrics.model_latency_s = round(
                sum(m.latency_s or 0.0 for m in loop.messages if m.latency_s), 4
            )
        verify_info = None
        if verify is not None:
            verify_info = VerifyInfo(
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
        return Trajectory(
            **base,
            messages=messages if messages is not None else (loop.messages if loop else []),
            tool_call_records=records,
            reward=reward,
            reward_breakdown=breakdown or {},
            final_status=status,
            stop_reason=loop.stop_reason if loop else None,
            failure_kind=failure,
            verify=verify_info,
            metrics=metrics,
            metadata={**meta, **(extra or {})},
            error=error,
            finished_at=datetime.now(timezone.utc),
        )

    handle: Sandbox | None = None
    outcome: FinalStatus | None = None
    try:
        t0 = time.monotonic()
        cached_image: str | None = None
        if sandbox_cfg.cache_setup and task.setup_commands:
            # Run setup once per environment and start from the snapshot (GH #24).
            try:
                base_image = await mgr.resolve_image()
                cached_image = await mgr.backend.derive_image(
                    base_image,
                    steps=list(task.setup_commands),
                    files=task.seed_files(),
                    workdir=sandbox_cfg.workspace_dir,
                    env=sandbox_cfg.env,
                    network=sandbox_cfg.prepare_network or not sandbox_cfg.limits.network_disabled,
                    shell=mgr.shell_for(base_image),
                    step_timeout_s=300.0,
                    key_extra={"purpose": "setup"},
                    max_images=sandbox_cfg.cache_max_images,
                )
            except Exception as exc:
                metrics.seed_s = time.monotonic() - t0
                outcome = FinalStatus.ERROR
                return build(
                    status=FinalStatus.ERROR,
                    failure=FailureKind.SEED,
                    error=f"setup failed: {exc}",
                )
            metrics.seed_s = time.monotonic() - t0
        t_create = time.monotonic()
        try:
            handle = await mgr.create(
                task_id=task.task_id,
                run_id=run_id,
                rollout_id=rollout_id,
                labels={"agentbox.group_id": group_id} if group_id else None,
                image=cached_image,
                resolve=cached_image is None,
            )
        except Exception as exc:
            logger.warning("sandbox create failed for %s: %s", task.task_id, exc)
            outcome = FinalStatus.ERROR
            return build(status=FinalStatus.ERROR, failure=FailureKind.SANDBOX, error=str(exc))
        metrics.sandbox_create_s = time.monotonic() - t_create

        if cached_image is None:
            seed = await TaskSeeder(mgr).seed(handle, task)
            metrics.seed_s = seed.duration_s
            if not seed.ok:
                outcome = FinalStatus.ERROR
                return build(
                    status=FinalStatus.ERROR,
                    failure=FailureKind.SEED,
                    error=seed.error or "task seed failed",
                )

        if setup_checks:
            from agentbox.benchmark.setup_check import SetupChecker

            check = await SetupChecker(mgr).run(handle, setup_checks)
            if not check.ok:
                outcome = FinalStatus.ERROR
                return build(
                    status=FinalStatus.ERROR,
                    failure=FailureKind.SETUP_CHECK,
                    error=check.error or "setup check failed",
                    extra={"failed_check": check.failed_check},
                )

        system = render_system_prompt(
            workspace_dir=sandbox_cfg.workspace_dir,
            mode=agent_cfg.tools,
            extra=task.system_prompt_extra,
            override=agent_cfg.system_prompt,
        )
        ctx = ToolContext(
            sandbox=handle,
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
        deadline = time.monotonic() + agent_cfg.episode_timeout_s
        loop_result = await loop.run(
            system_prompt=system,
            user_prompt=task.prompt_text(),
            ctx=ctx,
            episode_deadline=deadline,
        )

        t1 = time.monotonic()
        try:
            verify_result = await Verifier(mgr).verify(handle, task.verifier, task)
        except Exception as exc:
            logger.warning("verifier crashed for %s: %s", task.task_id, exc)
            metrics.verify_s = time.monotonic() - t1
            outcome = FinalStatus.ERROR
            return build(
                status=FinalStatus.ERROR,
                failure=FailureKind.VERIFIER,
                loop=loop_result,
                error=f"verifier failed: {exc}",
            )
        metrics.verify_s = time.monotonic() - t1

        reward, breakdown = await compute_reward(
            reward_specs, RewardContext(task=task, verify=verify_result, loop=loop_result)
        )
        if verify_result.success:
            status, failure = FinalStatus.SUCCESS, FailureKind.NONE
        else:
            status = {
                StopReason.TIMEOUT.value: FinalStatus.TIMEOUT,
                StopReason.MAX_STEPS.value: FinalStatus.MAX_STEPS,
                StopReason.ERROR.value: FinalStatus.ERROR,
            }.get(loop_result.stop_reason, FinalStatus.FAILED)
            failure = (
                loop_result.failure_kind
                if loop_result.failure_kind.is_infra
                else FailureKind.POLICY
            )
        outcome = status
        return build(
            status=status,
            failure=failure,
            reward=reward,
            breakdown=breakdown,
            loop=loop_result,
            verify=verify_result,
            error=loop_result.error,
            extra={"submitted_answer": loop_result.submitted_answer}
            if loop_result.submitted_answer
            else None,
        )
    except Exception as exc:
        logger.exception("rollout failed task_id=%s", task.task_id)
        outcome = FinalStatus.ERROR
        return build(status=FinalStatus.ERROR, failure=FailureKind.INTERNAL, error=str(exc))
    finally:
        if handle is not None:
            keep = not sandbox_cfg.auto_remove or (
                sandbox_cfg.keep_on_failure and outcome != FinalStatus.SUCCESS
            )
            if keep:
                mgr.keep(handle)
                logger.warning(
                    "keeping sandbox %s (status=%s); inspect with: docker exec -it %s sh",
                    handle.name,
                    outcome.value if outcome else "unknown",
                    handle.name,
                )
            else:
                try:
                    await mgr.destroy(handle)
                except Exception as exc:  # pragma: no cover - teardown must not mask results
                    logger.warning("sandbox destroy failed for %s: %s", handle.name, exc)
