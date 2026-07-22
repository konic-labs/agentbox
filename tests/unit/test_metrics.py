from agentbox.metrics import aggregate_trajectories
from agentbox.trajectory.schema import Trajectory, TrajectoryMetrics
from agentbox.types import FinalStatus


def test_aggregate() -> None:
    trajs = [
        Trajectory(
            task_id="a",
            run_id="1",
            messages=[],
            reward=1.0,
            final_status=FinalStatus.SUCCESS,
            metrics=TrajectoryMetrics(steps=3, tool_calls=2, duration_s=1.0),
        ),
        Trajectory(
            task_id="a",
            run_id="2",
            messages=[],
            reward=0.0,
            final_status=FinalStatus.FAILED,
            metrics=TrajectoryMetrics(steps=5, tool_calls=4, duration_s=2.0),
        ),
    ]
    m = aggregate_trajectories(trajs)
    assert m.n == 2
    assert m.success_rate == 0.5
    assert m.mean_reward == 0.5
    assert m.by_task["a"]["n"] == 2
