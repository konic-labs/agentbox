"""Aggregate metrics across trajectories."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from agentbox.trajectory.schema import Trajectory
from agentbox.types import FinalStatus


class AggregateMetrics(BaseModel):
    n: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    mean_steps: float = 0.0
    mean_tool_calls: float = 0.0
    mean_duration_s: float = 0.0
    by_status: dict[str, int] = Field(default_factory=dict)
    by_task: dict[str, dict[str, Any]] = Field(default_factory=dict)


def aggregate_trajectories(trajs: Iterable[Trajectory]) -> AggregateMetrics:
    items = list(trajs)
    n = len(items)
    if n == 0:
        return AggregateMetrics()

    succeeded = sum(1 for t in items if t.final_status == FinalStatus.SUCCESS)
    rewards = [t.reward for t in items]
    steps = [t.metrics.steps for t in items]
    tools = [t.metrics.tool_calls for t in items]
    durs = [t.metrics.duration_s for t in items]

    by_status: dict[str, int] = defaultdict(int)
    by_task_raw: dict[str, list[Trajectory]] = defaultdict(list)
    for t in items:
        by_status[t.final_status.value] += 1
        by_task_raw[t.task_id].append(t)

    by_task: dict[str, dict[str, Any]] = {}
    for task_id, group in by_task_raw.items():
        g_ok = sum(1 for x in group if x.final_status == FinalStatus.SUCCESS)
        by_task[task_id] = {
            "n": len(group),
            "success_rate": g_ok / len(group),
            "mean_reward": sum(x.reward for x in group) / len(group),
            "mean_steps": sum(x.metrics.steps for x in group) / len(group),
        }

    return AggregateMetrics(
        n=n,
        success_rate=succeeded / n,
        mean_reward=sum(rewards) / n,
        mean_steps=sum(steps) / n,
        mean_tool_calls=sum(tools) / n,
        mean_duration_s=sum(durs) / n,
        by_status=dict(by_status),
        by_task=by_task,
    )
