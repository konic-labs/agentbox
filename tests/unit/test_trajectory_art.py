"""ART export shape tests."""

from agentbox.trajectory.schema import Message, Trajectory, TrajectoryMetrics
from agentbox.types import FinalStatus, MessageRole


def test_to_art_dict_shape() -> None:
    traj = Trajectory(
        task_id="t1",
        run_id="r1",
        messages=[
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="user"),
            Message(role=MessageRole.ASSISTANT, content="ok"),
        ],
        reward=1.0,
        final_status=FinalStatus.SUCCESS,
        metrics=TrajectoryMetrics(steps=1, tool_calls=0, duration_s=1.5),
        tools=[{"type": "function", "function": {"name": "read_file"}}],
    )
    art = traj.to_art_dict()
    assert "messages_and_choices" in art
    assert art["reward"] == 1.0
    assert art["metrics"]["correct"] == 1.0
    assert art["metadata"]["task_id"] == "t1"
    assert art["tools"]
