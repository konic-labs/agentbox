"""Build Trajectory objects during a rollout."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agentbox.tasks.schema import Task
from agentbox.trajectory.schema import (
    Message,
    ToolCallRecord,
    Trajectory,
    TrajectoryMetrics,
)
from agentbox.types import FinalStatus


class TrajectoryRecorder:
    def __init__(
        self,
        task: Task,
        run_id: str,
        *,
        model: str | None = None,
        tool_mode: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.task = task
        self.run_id = run_id
        self.model = model
        self.tool_mode = tool_mode
        self.tools = tools
        self.messages: list[Message] = []
        self.tool_call_records: list[ToolCallRecord] = []
        self.metrics = TrajectoryMetrics()
        self.created_at = datetime.now(timezone.utc)

    def set_messages(self, messages: list[Message]) -> None:
        self.messages = list(messages)

    def set_tool_records(self, records: list[ToolCallRecord]) -> None:
        self.tool_call_records = list(records)

    def set_metrics(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

    def finalize(
        self,
        *,
        reward: float,
        final_status: FinalStatus,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Trajectory:
        finished = datetime.now(timezone.utc)
        if not self.metrics.duration_s:
            self.metrics.duration_s = (finished - self.created_at).total_seconds()
        meta = {"task_metadata": self.task.metadata, **(metadata or {})}
        return Trajectory(
            task_id=self.task.task_id,
            run_id=self.run_id,
            messages=self.messages,
            tool_call_records=self.tool_call_records,
            reward=reward,
            final_status=final_status,
            metrics=self.metrics,
            metadata=meta,
            error=error,
            model=self.model,
            tool_mode=self.tool_mode,
            tools=self.tools,
            created_at=self.created_at,
            finished_at=finished,
        )
