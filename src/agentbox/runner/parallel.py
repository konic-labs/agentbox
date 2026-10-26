"""Parallel rollout runner: bounded worker pool, streaming results, infra retries.

* ``concurrency`` workers pull jobs from a **lazy** iterator — no task per job,
  no list of all trajectories (GH #39);
* results are yielded as they finish (:meth:`ParallelRunner.stream`) or written
  to a sink (:meth:`ParallelRunner.run_to_sink`) with online metrics;
* infrastructure failures are retried and, if persistent, marked
  ``is_trainable=False`` instead of silently scoring 0 (GH #40);
* one model client per run, optionally rate-limited (GH #41, #46).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agentbox.agent.agent import Agent
from agentbox.agent.loop import AgentHooks
from agentbox.config import AgentConfig, ModelConfig, RolloutConfig, SandboxConfig
from agentbox.metrics.aggregate import AggregateMetrics, MetricsAccumulator
from agentbox.model.base import ModelClient, ModelResponse
from agentbox.runner.rollout import Rollout, resolve_model
from agentbox.sandbox.manager import SandboxManager
from agentbox.tasks.schema import Task
from agentbox.trajectory.io import TrajectorySink
from agentbox.trajectory.schema import Trajectory, TrajectoryMetrics
from agentbox.types import FailureKind, FinalStatus

logger = logging.getLogger("agentbox.parallel")

DEFAULT_RETRY_ON = frozenset(
    {
        FailureKind.SANDBOX,
        FailureKind.SEED,
        FailureKind.SETUP_CHECK,
        FailureKind.MODEL_API,
        FailureKind.VERIFIER,
        FailureKind.INTERNAL,
    }
)


@dataclass(slots=True)
class Job:
    task: Task
    sample_index: int
    group_id: str


class ParallelResult(BaseModel):
    trajectories: list[Trajectory] = Field(default_factory=list)
    succeeded: int = 0
    failed: int = 0
    errors: int = 0
    infra_failures: int = 0

    @property
    def success_rate(self) -> float:
        if not self.trajectories:
            return 0.0
        return self.succeeded / len(self.trajectories)


class RunSummary(BaseModel):
    run_id: str
    n: int
    skipped: int = 0
    wall_clock_s: float
    rollouts_per_min: float
    metrics: AggregateMetrics


class _LimitedModel:
    """Bounds in-flight requests to a model endpoint (separate from sandbox concurrency)."""

    def __init__(self, inner: ModelClient, limit: int) -> None:
        self.inner = inner
        self.model = inner.model
        self._sem = asyncio.Semaphore(limit)

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        async with self._sem:
            return await self.inner.complete(messages, **kwargs)


class _Progress:
    def __init__(self, total: int | None, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled
        self.done = 0
        self.ok = 0
        self._bar: Any = None
        self._task: Any = None

    def __enter__(self) -> _Progress:
        if not self.enabled:
            return self
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
        except ImportError:
            return self
        self._bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("ok={task.fields[ok]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self._bar.__enter__()
        self._task = self._bar.add_task("rollouts", total=self.total, ok=0)
        return self

    def advance(self, traj: Trajectory) -> None:
        self.done += 1
        self.ok += traj.final_status == FinalStatus.SUCCESS
        if self._bar is not None:
            self._bar.update(self._task, advance=1, ok=self.ok)
        elif self.enabled and self.total and self.done % max(1, self.total // 10) == 0:
            logger.info("parallel.progress %d/%d ok=%d", self.done, self.total, self.ok)

    def __exit__(self, *exc: object) -> None:
        if self._bar is not None:
            self._bar.__exit__(None, None, None)
            self._bar = None


def _error_trajectory(job: Job, run_id: str, exc: BaseException) -> Trajectory:
    now = datetime.now(timezone.utc)
    return Trajectory(
        task_id=job.task.task_id,
        run_id=run_id,
        rollout_id=uuid.uuid4().hex,
        group_id=job.group_id,
        sample_index=job.sample_index,
        messages=[],
        final_status=FinalStatus.ERROR,
        failure_kind=FailureKind.INTERNAL,
        metrics=TrajectoryMetrics(),
        error=f"{type(exc).__name__}: {exc}",
        created_at=now,
        finished_at=now,
    )


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
        max_retries: int = 1,
        retry_backoff_s: float = 2.0,
        retry_on: Iterable[FailureKind] = DEFAULT_RETRY_ON,
        model_concurrency: int | None = None,
        run_id: str | None = None,
        hooks: AgentHooks | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.concurrency = concurrency
        self.config = config
        self.model = model
        self.agent = agent
        self.sandbox = sandbox
        self.manager = manager
        self.max_retries = max(0, max_retries)
        self.retry_backoff_s = retry_backoff_s
        self.retry_on = frozenset(retry_on)
        self.model_concurrency = model_concurrency
        self.run_id = run_id or (config.run_id if config and config.run_id else uuid.uuid4().hex)
        self.hooks = hooks

    # ------------------------------------------------------------------ jobs
    def _jobs(
        self,
        tasks: Iterable[Task],
        n_per_task: int,
        skip: set[tuple[str, int]] | None,
    ) -> Iterator[Job]:
        for task in tasks:
            group_id = f"{task.task_id}@{self.run_id[:8]}"
            for sample_index in range(n_per_task):
                if skip and (task.task_id, sample_index) in skip:
                    continue
                yield Job(task=task, sample_index=sample_index, group_id=group_id)

    def _model_client(self) -> tuple[ModelClient, bool]:
        client, owned = resolve_model(self.model, self.agent, self.config)
        if self.model_concurrency:
            return _LimitedModel(client, self.model_concurrency), owned
        return client, owned

    async def _run_job(
        self, job: Job, model_client: ModelClient, setup_checks: list[Any] | None
    ) -> Trajectory:
        attempt = 0
        while True:
            attempt += 1
            try:
                traj = await Rollout.run(
                    job.task,
                    model=model_client,
                    agent=self.agent,
                    sandbox=self.sandbox,
                    config=self.config,
                    manager=self.manager,
                    setup_checks=setup_checks,
                    run_id=self.run_id,
                    group_id=job.group_id,
                    sample_index=job.sample_index,
                    hooks=self.hooks,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # Rollout.run is defensive; this is a last resort
                logger.exception("rollout crashed task_id=%s", job.task.task_id)
                traj = _error_trajectory(job, self.run_id, exc)
            traj.metrics.attempts = attempt
            kind = FailureKind(traj.failure_kind)
            if kind not in self.retry_on or attempt > self.max_retries:
                return traj
            logger.warning(
                "retrying %s#%d after %s failure (attempt %d): %s",
                job.task.task_id,
                job.sample_index,
                kind.value,
                attempt,
                (traj.error or "")[:200],
            )
            await asyncio.sleep(self.retry_backoff_s * attempt)

    # ------------------------------------------------------------- streaming
    async def stream(
        self,
        tasks: Iterable[Task],
        *,
        n_per_task: int = 1,
        setup_checks: Sequence[Any] | None = None,
        skip: set[tuple[str, int]] | None = None,
        progress: bool = False,
    ) -> AsyncIterator[Trajectory]:
        """Yield trajectories as rollouts finish (completion order)."""
        if n_per_task < 1:
            raise ValueError("n_per_task must be >= 1")
        total = None
        if isinstance(tasks, Sequence):
            total = len(tasks) * n_per_task - (len(skip) if skip else 0)
        jobs = self._jobs(tasks, n_per_task, skip)
        checks = list(setup_checks) if setup_checks else None
        queue: asyncio.Queue[Trajectory | None] = asyncio.Queue(maxsize=self.concurrency * 2)
        resolved: dict[str, Any] = {}  # model client resolved lazily (empty runs need none)
        failure: list[BaseException] = []

        def model_client() -> ModelClient:
            if "client" not in resolved:
                resolved["client"], resolved["owned"] = self._model_client()
            client: ModelClient = resolved["client"]
            return client

        async def worker() -> None:
            for job in jobs:  # shared iterator: next() never awaits, so no lock needed
                await queue.put(await self._run_job(job, model_client(), checks))

        async def supervise() -> None:
            try:
                results = await asyncio.gather(
                    *(worker() for _ in range(self.concurrency)), return_exceptions=True
                )
                failure.extend(
                    r for r in results if isinstance(r, BaseException)
                    and not isinstance(r, asyncio.CancelledError)
                )
            finally:
                await queue.put(None)

        supervisor = asyncio.create_task(supervise())
        try:
            with _Progress(total, progress) as bar:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    bar.advance(item)
                    yield item
            if failure:
                raise failure[0]
        finally:
            if not supervisor.done():
                supervisor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await supervisor
            if resolved.get("owned"):
                client = resolved["client"]
                close = getattr(getattr(client, "inner", client), "aclose", None)
                if close is not None:
                    await close()

    async def stream_groups(
        self,
        tasks: Iterable[Task],
        *,
        group_size: int = 4,
        setup_checks: Sequence[Any] | None = None,
        progress: bool = False,
    ) -> AsyncIterator[list[Trajectory]]:
        """Yield each GRPO group (sorted by sample index) as soon as it is complete."""
        pending: dict[str, list[Trajectory]] = {}
        async for traj in self.stream(
            tasks, n_per_task=group_size, setup_checks=setup_checks, progress=progress
        ):
            key = traj.group_id or traj.task_id
            bucket = pending.setdefault(key, [])
            bucket.append(traj)
            if len(bucket) == group_size:
                del pending[key]
                yield sorted(bucket, key=lambda t: t.sample_index)
        for bucket in pending.values():  # only reachable if a job vanished
            yield sorted(bucket, key=lambda t: t.sample_index)

    async def run_to_sink(
        self,
        tasks: Iterable[Task],
        sink: TrajectorySink,
        *,
        n_per_task: int = 1,
        setup_checks: Sequence[Any] | None = None,
        resume: bool = False,
        progress: bool = False,
        on_trajectory: Callable[[Trajectory], Any] | None = None,
    ) -> RunSummary:
        """Stream every trajectory into ``sink``; returns online-aggregated metrics."""
        skip: set[tuple[str, int]] | None = None
        if resume:
            completed = getattr(sink, "completed", None)
            skip = completed() if callable(completed) else None
        acc = MetricsAccumulator()
        t0 = time.monotonic()
        try:
            async for traj in self.stream(
                tasks,
                n_per_task=n_per_task,
                setup_checks=setup_checks,
                skip=skip,
                progress=progress,
            ):
                await sink.write(traj)
                acc.add(traj)
                if on_trajectory is not None:
                    maybe = on_trajectory(traj)
                    if asyncio.iscoroutine(maybe):
                        await maybe
        finally:
            await sink.close()
        wall = time.monotonic() - t0
        return RunSummary(
            run_id=self.run_id,
            n=acc.n,
            skipped=len(skip or ()),
            wall_clock_s=wall,
            rollouts_per_min=(acc.n / wall * 60.0) if wall > 0 else 0.0,
            metrics=acc.result(),
        )

    # ---------------------------------------------------------- list APIs
    async def run_tasks(
        self,
        tasks: Sequence[Task],
        *,
        n_per_task: int = 1,
        progress: bool = False,
        on_trajectory: Callable[[Trajectory], Any] | None = None,
        setup_checks: Sequence[Any] | None = None,
    ) -> list[Trajectory]:
        """Collect all trajectories, ordered by task then sample index."""
        order = {t.task_id: i for i, t in enumerate(tasks)}
        out: list[Trajectory] = []
        async for traj in self.stream(
            tasks, n_per_task=n_per_task, setup_checks=setup_checks, progress=progress
        ):
            out.append(traj)
            if on_trajectory is not None:
                maybe = on_trajectory(traj)
                if asyncio.iscoroutine(maybe):
                    await maybe
        out.sort(key=lambda t: (order.get(t.task_id, len(order)), t.sample_index))
        return out

    async def run_groups(
        self,
        tasks: Sequence[Task],
        *,
        group_size: int = 4,
        progress: bool = False,
    ) -> list[list[Trajectory]]:
        """GRPO-style groups: ``group_size`` rollouts per task, in task order."""
        if group_size < 1:
            raise ValueError("group_size must be >= 1")
        trajs = await self.run_tasks(tasks, n_per_task=group_size, progress=progress)
        groups: dict[str, list[Trajectory]] = {}
        for t in trajs:
            groups.setdefault(t.task_id, []).append(t)
        return [groups.get(t.task_id, []) for t in tasks]

    async def run(self, tasks: Sequence[Task], **kwargs: Any) -> ParallelResult:
        return self._summarize(await self.run_tasks(tasks, **kwargs))

    @staticmethod
    def _summarize(trajs: list[Trajectory]) -> ParallelResult:
        succeeded = sum(1 for t in trajs if t.final_status == FinalStatus.SUCCESS)
        errors = sum(1 for t in trajs if t.final_status == FinalStatus.ERROR)
        return ParallelResult(
            trajectories=trajs,
            succeeded=succeeded,
            failed=len(trajs) - succeeded - errors,
            errors=errors,
            infra_failures=sum(1 for t in trajs if not t.is_trainable),
        )
