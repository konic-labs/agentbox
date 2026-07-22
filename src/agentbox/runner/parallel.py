"""Parallel rollout runner with concurrency control."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agentbox.agent.agent import Agent
from agentbox.config import AgentConfig, ModelConfig, RolloutConfig, SandboxConfig
from agentbox.model.base import ModelClient
from agentbox.runner.rollout import Rollout
from agentbox.sandbox.manager import SandboxManager
from agentbox.tasks.schema import Task
from agentbox.trajectory.schema import Trajectory, TrajectoryMetrics
from agentbox.types import FinalStatus

logger = logging.getLogger("agentbox.parallel")


class ParallelResult(BaseModel):
    trajectories: list[Trajectory] = Field(default_factory=list)
    succeeded: int = 0
    failed: int = 0
    errors: int = 0

    @property
    def success_rate(self) -> float:
        if not self.trajectories:
            return 0.0
        return self.succeeded / len(self.trajectories)


class ParallelRunner:
    """Run many rollouts concurrently with failure isolation."""

    def __init__(
        self,
        *,
        concurrency: int = 8,
        config: RolloutConfig | None = None,
        model: ModelClient | ModelConfig | str | None = None,
        agent: Agent | AgentConfig | None = None,
        sandbox: SandboxConfig | None = None,
        manager: SandboxManager | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.concurrency = concurrency
        self.config = config
        self.model = model
        self.agent = agent
        self.sandbox = sandbox
        self.manager = manager

    async def run_tasks(
        self,
        tasks: Sequence[Task],
        *,
        n_per_task: int = 1,
        progress: bool = True,
        on_trajectory: Callable[[Trajectory], Any] | None = None,
        setup_checks: Sequence[Any] | None = None,
    ) -> list[Trajectory]:
        """Expand tasks × n_per_task into jobs and run with a semaphore."""
        if n_per_task < 1:
            raise ValueError("n_per_task must be >= 1")
        jobs: list[Task] = []
        for task in tasks:
            for _ in range(n_per_task):
                jobs.append(task)

        result = await self._run_jobs(
            jobs,
            progress=progress,
            on_trajectory=on_trajectory,
            setup_checks=list(setup_checks) if setup_checks else None,
        )
        return result.trajectories

    async def run_groups(
        self,
        tasks: Sequence[Task],
        *,
        group_size: int = 4,
        progress: bool = True,
    ) -> list[list[Trajectory]]:
        """GRPO-style groups: group_size rollouts per task."""
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        all_trajs = await self.run_tasks(tasks, n_per_task=group_size, progress=progress)
        groups: list[list[Trajectory]] = []
        idx = 0
        for _ in tasks:
            groups.append(all_trajs[idx : idx + group_size])
            idx += group_size
        return groups

    async def run(self, tasks: Sequence[Task], **kwargs: Any) -> ParallelResult:
        trajs = await self.run_tasks(tasks, **kwargs)
        return self._summarize(trajs)

    async def _run_jobs(
        self,
        jobs: Sequence[Task],
        *,
        progress: bool,
        on_trajectory: Callable[[Trajectory], Any] | None,
        setup_checks: list[Any] | None = None,
    ) -> ParallelResult:
        sem = asyncio.Semaphore(self.concurrency)
        results: list[Trajectory | None] = [None] * len(jobs)
        completed = 0
        total = len(jobs)
        pbar = None
        if progress:
            try:
                from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

                pbar_cm = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                )
                pbar_cm.__enter__()
                pbar = pbar_cm
                task_id = pbar.add_task("rollouts", total=total)
            except ImportError:
                pbar = None
                task_id = None
                logger.info("parallel.start jobs=%d concurrency=%d", total, self.concurrency)

        async def _one(i: int, task: Task) -> None:
            nonlocal completed
            async with sem:
                try:
                    traj = await Rollout.run(
                        task,
                        model=self.model,
                        agent=self.agent,
                        sandbox=self.sandbox,
                        config=self.config,
                        manager=None,  # isolate managers per job for thread-safety
                        setup_checks=setup_checks,
                    )
                except Exception as exc:
                    logger.exception("parallel job failed task_id=%s", task.task_id)
                    traj = Trajectory(
                        task_id=task.task_id,
                        run_id=str(uuid.uuid4()),
                        messages=[],
                        reward=0.0,
                        final_status=FinalStatus.ERROR,
                        metrics=TrajectoryMetrics(),
                        error=str(exc),
                        created_at=datetime.now(timezone.utc),
                        finished_at=datetime.now(timezone.utc),
                    )
                results[i] = traj
                if on_trajectory is not None:
                    maybe = on_trajectory(traj)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                completed += 1
                if pbar is not None and task_id is not None:
                    pbar.update(task_id, completed=completed)
                elif progress and completed % max(1, total // 10) == 0:
                    logger.info("parallel.progress %d/%d", completed, total)

        await asyncio.gather(*(_one(i, t) for i, t in enumerate(jobs)))

        if pbar is not None:
            pbar.__exit__(None, None, None)

        trajs = [r for r in results if r is not None]
        return self._summarize(trajs)

    @staticmethod
    def _summarize(trajs: list[Trajectory]) -> ParallelResult:
        succeeded = sum(1 for t in trajs if t.final_status == FinalStatus.SUCCESS)
        errors = sum(1 for t in trajs if t.final_status == FinalStatus.ERROR)
        failed = len(trajs) - succeeded - errors
        return ParallelResult(
            trajectories=trajs,
            succeeded=succeeded,
            failed=failed,
            errors=errors,
        )
