"""Single-rollout orchestrator."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from agentbox.agent.agent import Agent
from agentbox.agent.loop import AgentLoop
from agentbox.agent.prompts import render_system_prompt
from agentbox.config import AgentConfig, ModelConfig, RolloutConfig, SandboxConfig
from agentbox.model.base import ModelClient
from agentbox.model.openai_compat import OpenAICompatClient
from agentbox.sandbox.manager import SandboxManager
from agentbox.tasks.schema import Task
from agentbox.tasks.seeder import TaskSeeder
from agentbox.tasks.verifier import Verifier
from agentbox.tools.base import ToolContext
from agentbox.tools.executor import ToolExecutor
from agentbox.tools.registry import build_tool_registry
from agentbox.trajectory.recorder import TrajectoryRecorder
from agentbox.trajectory.schema import Trajectory
from agentbox.types import FinalStatus, ToolMode

logger = logging.getLogger("agentbox.rollout")


def _resolve_model(
    model: ModelClient | ModelConfig | str | None,
    agent: Agent | AgentConfig | None,
    config: RolloutConfig | None,
) -> ModelClient:
    if isinstance(model, (str, type(None))) is False and hasattr(model, "complete"):
        return model  # type: ignore[return-value]
    if isinstance(model, ModelConfig):
        return OpenAICompatClient(model)
    if isinstance(model, str):
        return OpenAICompatClient(ModelConfig(model=model))
    if isinstance(agent, Agent):
        return agent.model_client
    if config is not None:
        return OpenAICompatClient(config.model)
    raise ValueError("Must provide model, agent with model, or RolloutConfig.model")


def _resolve_agent_config(
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
    return cfg


class Rollout:
    """Run one task in an isolated sandbox and return a Trajectory."""

    @staticmethod
    async def run(
        task: Task,
        *,
        model: ModelClient | ModelConfig | str | None = None,
        agent: Agent | AgentConfig | dict[str, Any] | None = None,
        sandbox: SandboxConfig | None = None,
        config: RolloutConfig | None = None,
        manager: SandboxManager | None = None,
    ) -> Trajectory:
        run_id = (
            (config.run_id if config else None)
            or str(uuid.uuid4())
        )
        sandbox_cfg = sandbox or (config.sandbox if config else SandboxConfig())
        agent_cfg = _resolve_agent_config(agent, config, task)
        model_client = _resolve_model(model, agent, config)
        owns_manager = manager is None
        mgr = manager or SandboxManager(sandbox_cfg)

        tool_mode = agent_cfg.tools
        registry = build_tool_registry(
            tool_mode,
            custom_tools=agent_cfg.custom_tools,
            include_builtins=agent_cfg.builtins,
            drop=agent_cfg.drop_tools,
            drop_prob=agent_cfg.drop_tools_prob,
        )
        executor = ToolExecutor(registry)
        tool_schemas = executor.schemas()

        recorder = TrajectoryRecorder(
            task,
            run_id,
            model=getattr(model_client, "model", None),
            tool_mode=str(tool_mode.value if isinstance(tool_mode, ToolMode) else tool_mode),
            tools=tool_schemas,
        )

        sandbox_handle = None
        t_start = time.monotonic()
        try:
            t0 = time.monotonic()
            sandbox_handle = await mgr.create(task_id=task.task_id, run_id=run_id)
            recorder.set_metrics(sandbox_create_s=time.monotonic() - t0)

            seeder = TaskSeeder(mgr)
            seed = await seeder.seed(sandbox_handle, task)
            recorder.set_metrics(seed_s=seed.duration_s)
            if not seed.ok:
                recorder.set_messages([])
                return recorder.finalize(
                    reward=0.0,
                    final_status=FinalStatus.ERROR,
                    error=seed.error or "task seed failed",
                )

            system = render_system_prompt(
                workspace_dir=sandbox_cfg.workspace_dir,
                mode=tool_mode if isinstance(tool_mode, (ToolMode, str)) else ToolMode.STRUCTURED,
                extra=task.system_prompt_extra,
                override=agent_cfg.system_prompt,
            )
            ctx = ToolContext(
                sandbox=sandbox_handle,
                manager=mgr,
                workspace_dir=sandbox_cfg.workspace_dir,
                step=0,
                run_id=run_id,
                task_id=task.task_id,
            )
            loop = AgentLoop(model_client, executor, agent_cfg)
            deadline = time.monotonic() + agent_cfg.episode_timeout_s
            loop_result = await loop.run(
                system_prompt=system,
                user_prompt=task.description,
                ctx=ctx,
                episode_deadline=deadline,
            )
            recorder.set_messages(loop_result.messages)
            recorder.set_tool_records(loop_result.tool_call_records)
            recorder.set_metrics(
                steps=loop_result.steps,
                tool_calls=len(loop_result.tool_call_records),
                model_calls=loop_result.model_calls,
                prompt_tokens=loop_result.usage_totals.get("prompt_tokens"),
                completion_tokens=loop_result.usage_totals.get("completion_tokens"),
                total_tokens=loop_result.usage_totals.get("total_tokens"),
            )

            if loop_result.final_status == FinalStatus.ERROR and loop_result.error:
                # Still attempt verify if sandbox is usable
                pass

            t1 = time.monotonic()
            verify_result = await Verifier(mgr).verify(
                sandbox_handle, task.verifier
            )
            recorder.set_metrics(verify_s=time.monotonic() - t1)

            reward = verify_result.reward
            if verify_result.success:
                final_status = FinalStatus.SUCCESS
            elif loop_result.stop_reason == "timeout":
                final_status = FinalStatus.TIMEOUT
            elif loop_result.stop_reason == "max_steps":
                final_status = FinalStatus.MAX_STEPS
            elif loop_result.stop_reason == "error":
                final_status = FinalStatus.ERROR
            else:
                final_status = FinalStatus.FAILED

            traj = recorder.finalize(
                reward=reward,
                final_status=final_status,
                error=loop_result.error,
                metadata={
                    "verify_exit_code": verify_result.exit_code,
                    "verify_command": verify_result.command,
                    "stop_reason": loop_result.stop_reason,
                },
            )
            traj.metrics.duration_s = time.monotonic() - t_start
            return traj
        except Exception as exc:
            logger.exception("rollout failed task_id=%s", task.task_id)
            recorder.set_metrics(duration_s=time.monotonic() - t_start)
            return recorder.finalize(
                reward=0.0,
                final_status=FinalStatus.ERROR,
                error=str(exc),
            )
        finally:
            if sandbox_handle is not None and not sandbox_cfg.keep_on_failure:
                await mgr.destroy(sandbox_handle)
