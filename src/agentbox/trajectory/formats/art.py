"""ART-compatible trajectory export."""

from __future__ import annotations

from typing import Any

from agentbox.trajectory.schema import Trajectory
from agentbox.types import FinalStatus


def to_art_dict(
    traj: Trajectory, *, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Export a portable dict matching OpenPipe ART Trajectory shape."""
    messages_and_choices = [m.to_openai_dict() for m in traj.messages]
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
        "final_status": traj.final_status.value,
        **traj.metadata,
    }
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
