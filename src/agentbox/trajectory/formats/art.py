"""ART-compatible trajectory export."""

from __future__ import annotations

from typing import Any

from agentbox.trajectory.schema import Trajectory
from agentbox.types import FinalStatus


def to_art_dict(
    traj: Trajectory, *, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Export a portable dict matching OpenPipe ART Trajectory shape."""
    messages_and_choices = [m.to_openai_dict(include_reasoning=True) for m in traj.messages]
    if traj.verify is not None:
        correct = 1.0 if traj.verify.success else 0.0
    else:
        correct = 1.0 if traj.final_status == FinalStatus.SUCCESS else 0.0
    metrics: dict[str, float | int | bool] = {
        "duration": traj.metrics.duration_s,
        "steps": traj.metrics.steps,
        "tool_calls": traj.metrics.tool_calls,
        "correct": correct,
    }
    if traj.metrics.model_calls:
        metrics["model_calls"] = traj.metrics.model_calls
    if traj.metrics.total_tokens is not None:
        metrics["total_tokens"] = traj.metrics.total_tokens

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


def to_art(traj: Trajectory) -> Any:
    """Return a live art.Trajectory if openpipe-art is installed."""
    try:
        import art  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "openpipe-art is required for to_art(). "
            "Install with: pip install agentbox[art]"
        ) from exc

    data = to_art_dict(traj)
    return art.Trajectory(
        messages_and_choices=data["messages_and_choices"],
        tools=data.get("tools"),
        reward=data["reward"],
        metrics=data["metrics"],
        metadata=data["metadata"],
    )
