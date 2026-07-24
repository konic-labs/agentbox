"""ART group / GRPO offline contract tests."""

from __future__ import annotations

from agentbox.trajectory.schema import Message, Trajectory, TrajectoryMetrics
from agentbox.types import FinalStatus, MessageRole


def _traj(task_id: str, reward: float, run_id: str) -> Trajectory:
    return Trajectory(
        task_id=task_id,
        run_id=run_id,
        messages=[
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="do it"),
            Message(role=MessageRole.ASSISTANT, content="done"),
        ],
        reward=reward,
        final_status=FinalStatus.SUCCESS if reward > 0 else FinalStatus.FAILED,
        metrics=TrajectoryMetrics(steps=3, tool_calls=1, duration_s=1.0),
        tools=[{"type": "function", "function": {"name": "write_file"}}],
    )


def test_art_group_relative_advantages() -> None:
    trajs = [
        _traj("t1", 1.0, "r1"),
        _traj("t1", 0.0, "r2"),
        _traj("t1", 1.0, "r3"),
        _traj("t1", 0.0, "r4"),
    ]
    group = [t.to_art_dict() for t in trajs]
    for art in group:
        assert "messages_and_choices" in art
        assert "reward" in art
        assert art["metrics"]["correct"] in (0.0, 1.0)
        assert art["metadata"]["task_id"] == "t1"
    rewards = [float(x["reward"]) for x in group]
    mean = sum(rewards) / len(rewards)
    advantages = [r - mean for r in rewards]
    assert abs(sum(advantages)) < 1e-9
    assert max(advantages) > 0
    assert min(advantages) < 0


def test_art_dict_tools_preserved() -> None:
    art = _traj("t2", 1.0, "rx").to_art_dict()
    assert art["tools"]
    assert art["tools"][0]["function"]["name"] == "write_file"
