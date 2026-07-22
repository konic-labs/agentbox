from agentbox.benchmark.report import build_leaderboard, build_model_report, redact_model_config
from agentbox.trajectory.schema import Trajectory, TrajectoryMetrics
from agentbox.types import FinalStatus


def test_redact_api_key() -> None:
    assert redact_model_config({"model": "m", "api_key": "secret"})["api_key"] == "***"


def test_leaderboard_order() -> None:
    def traj(tid: str, ok: bool) -> Trajectory:
        return Trajectory(
            task_id=tid,
            run_id="r",
            messages=[],
            reward=1.0 if ok else 0.0,
            final_status=FinalStatus.SUCCESS if ok else FinalStatus.FAILED,
            metrics=TrajectoryMetrics(steps=2, duration_s=1.0),
        )

    m1 = build_model_report("a", {"model": "a"}, [traj("t", True), traj("t", False)], wall_clock_s=1)
    m2 = build_model_report("b", {"model": "b"}, [traj("t", True), traj("t", True)], wall_clock_s=1)
    board = build_leaderboard([m1, m2])
    assert board[0]["model_id"] == "b"
    assert board[0]["success_rate"] == 1.0
