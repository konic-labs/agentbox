"""OpenPipe ART adapter (GH #75, #67).

AgentBox no longer depends on ART; this module converts trajectories into ART's
``Trajectory`` / ``TrajectoryGroup`` shapes when ``openpipe-art`` is installed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from agentbox.trajectory.schema import Trajectory
from agentbox.types import FinalStatus

_SCALARS = (str, int, float, bool, type(None))


def _scalar(value: Any) -> Any:
    """ART releases before the trajectories refactor accept only scalar metadata."""
    if isinstance(value, _SCALARS):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def to_art_dict(
    traj: Trajectory,
    *,
    tools: list[dict[str, Any]] | None = None,
    scalar_metadata: bool = True,
) -> dict[str, Any]:
    """Portable dict in ART ``Trajectory`` shape."""
    messages_and_choices = [m.to_openai_dict(include_reasoning=True) for m in traj.messages]
    if traj.verify is not None:
        correct = 1.0 if traj.verify.success else 0.0
    else:
        correct = 1.0 if traj.final_status == FinalStatus.SUCCESS else 0.0
    metrics: dict[str, float | int | bool] = {
        "duration": traj.metrics.duration_s,
        "steps": traj.metrics.steps,
        "tool_calls": traj.metrics.tool_calls,
        "tool_errors": traj.metrics.tool_errors,
        "correct": correct,
    }
    if traj.metrics.model_calls:
        metrics["model_calls"] = traj.metrics.model_calls
    if traj.metrics.total_tokens is not None:
        metrics["total_tokens"] = traj.metrics.total_tokens
    for name, value in traj.reward_breakdown.items():
        metrics[f"reward/{name}"] = value

    metadata: dict[str, Any] = {
        "task_id": traj.task_id,
        "run_id": traj.run_id,
        "rollout_id": traj.rollout_id,
        "final_status": traj.final_status.value,
        "failure_kind": str(getattr(traj.failure_kind, "value", traj.failure_kind)),
        **traj.metadata,
    }
    if traj.group_id is not None:
        metadata["group_id"] = traj.group_id
    if traj.stop_reason is not None:
        metadata["stop_reason"] = traj.stop_reason
    if traj.verify is not None:
        metadata["verify_exit_code"] = traj.verify.exit_code
        metadata["verify_success"] = traj.verify.success
    if traj.model:
        metadata["model"] = traj.model
    if traj.error:
        metadata["error"] = traj.error
    if scalar_metadata:
        metadata = {k: _scalar(v) for k, v in metadata.items()}

    out: dict[str, Any] = {
        "messages_and_choices": messages_and_choices,
        "reward": traj.reward,
        "metrics": metrics,
        "metadata": metadata,
    }
    tool_schemas = tools if tools is not None else traj.tools
    if tool_schemas:
        out["tools"] = tool_schemas
    return out


def _art() -> Any:
    try:
        import art
    except ImportError as exc:
        raise ImportError("openpipe-art is required: pip install 'agentbox-rl[art]'") from exc
    return art


def to_art_trajectory(traj: Trajectory) -> Any:
    """A live ``art.Trajectory``."""
    art = _art()
    data = to_art_dict(traj)
    return art.Trajectory(
        messages_and_choices=data["messages_and_choices"],
        tools=data.get("tools"),
        reward=data["reward"],
        metrics=data["metrics"],
        metadata=data["metadata"],
    )


def to_art_groups(trajectories: Iterable[Trajectory], *, trainable_only: bool = True) -> list[Any]:
    """``art.TrajectoryGroup`` per ``group_id`` (infra failures dropped by default)."""
    art = _art()
    groups: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        if trainable_only and not t.is_trainable:
            continue
        groups.setdefault(t.group_id or t.task_id, []).append(t)
    return [art.TrajectoryGroup([to_art_trajectory(t) for t in g]) for g in groups.values()]
