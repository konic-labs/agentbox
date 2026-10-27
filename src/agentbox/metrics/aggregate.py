"""Aggregate metrics across trajectories (streaming-friendly)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from agentbox.rl.stats import pass_at_k
from agentbox.trajectory.schema import Trajectory
from agentbox.types import FinalStatus


class AggregateMetrics(BaseModel):
    n: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    reward_std: float = 0.0
    mean_steps: float = 0.0
    mean_tool_calls: float = 0.0
    mean_duration_s: float = 0.0
    p50_duration_s: float = 0.0
    p90_duration_s: float = 0.0
    mean_total_tokens: float | None = None
    tool_error_rate: float = 0.0
    infra_failures: int = 0
    salvaged_tool_calls: int = 0
    by_status: dict[str, int] = Field(default_factory=dict)
    by_failure_kind: dict[str, int] = Field(default_factory=dict)
    by_task: dict[str, dict[str, Any]] = Field(default_factory=dict)


class _TaskAcc:
    __slots__ = ("n", "successes", "reward_sum", "steps_sum", "duration_sum")

    def __init__(self) -> None:
        self.n = 0
        self.successes = 0
        self.reward_sum = 0.0
        self.steps_sum = 0
        self.duration_sum = 0.0


class MetricsAccumulator:
    """Online aggregation: O(tasks) memory instead of keeping trajectories."""

    def __init__(self) -> None:
        self.n = 0
        self.successes = 0
        self._reward_sum = 0.0
        self._reward_sq = 0.0
        self._steps = 0
        self._tools = 0
        self._tool_errors = 0
        self._tokens = 0
        self._tokens_n = 0
        self._durations: list[float] = []
        self.infra_failures = 0
        self.salvaged_tool_calls = 0
        self.by_status: dict[str, int] = defaultdict(int)
        self.by_failure_kind: dict[str, int] = defaultdict(int)
        self._tasks: dict[str, _TaskAcc] = {}

    def add(self, t: Trajectory) -> None:
        self.n += 1
        ok = t.final_status == FinalStatus.SUCCESS
        self.successes += ok
        self._reward_sum += t.reward
        self._reward_sq += t.reward * t.reward
        self._steps += t.metrics.steps
        self._tools += t.metrics.tool_calls
        self._tool_errors += sum(1 for r in t.tool_call_records if r.is_error)
        self.salvaged_tool_calls += t.metrics.salvaged_tool_calls
        if t.metrics.total_tokens is not None:
            self._tokens += t.metrics.total_tokens
            self._tokens_n += 1
        self._durations.append(t.metrics.duration_s)
        self.by_status[t.final_status.value] += 1
        self.by_failure_kind[str(getattr(t.failure_kind, "value", t.failure_kind))] += 1
        if not t.is_trainable:
            self.infra_failures += 1
        acc = self._tasks.get(t.task_id)
        if acc is None:
            acc = self._tasks[t.task_id] = _TaskAcc()
        acc.n += 1
        acc.successes += ok
        acc.reward_sum += t.reward
        acc.steps_sum += t.metrics.steps
        acc.duration_sum += t.metrics.duration_s

    def task_counts(self) -> dict[str, tuple[int, int]]:
        """task_id -> (n, successes)."""
        return {tid: (a.n, a.successes) for tid, a in self._tasks.items()}

    def result(self) -> AggregateMetrics:
        n = self.n
        if n == 0:
            return AggregateMetrics()
        mean_r = self._reward_sum / n
        var = max(0.0, self._reward_sq / n - mean_r * mean_r)
        durs = sorted(self._durations)

        def pct(q: float) -> float:
            return durs[min(len(durs) - 1, int(math.ceil(q * len(durs))) - 1)] if durs else 0.0

        by_task = {
            tid: {
                "n": a.n,
                "successes": a.successes,
                "success_rate": a.successes / a.n,
                "pass_at_1": pass_at_k(a.n, a.successes, 1),
                "mean_reward": a.reward_sum / a.n,
                "mean_steps": a.steps_sum / a.n,
                "mean_duration_s": a.duration_sum / a.n,
            }
            for tid, a in self._tasks.items()
        }
        return AggregateMetrics(
            n=n,
            success_rate=self.successes / n,
            mean_reward=mean_r,
            reward_std=math.sqrt(var),
            mean_steps=self._steps / n,
            mean_tool_calls=self._tools / n,
            mean_duration_s=sum(durs) / n,
            p50_duration_s=pct(0.5),
            p90_duration_s=pct(0.9),
            mean_total_tokens=(self._tokens / self._tokens_n) if self._tokens_n else None,
            tool_error_rate=(self._tool_errors / self._tools) if self._tools else 0.0,
            infra_failures=self.infra_failures,
            salvaged_tool_calls=self.salvaged_tool_calls,
            by_status=dict(self.by_status),
            by_failure_kind=dict(self.by_failure_kind),
            by_task=by_task,
        )


def aggregate_trajectories(trajs: Iterable[Trajectory]) -> AggregateMetrics:
    acc = MetricsAccumulator()
    for t in trajs:
        acc.add(t)
    return acc.result()
