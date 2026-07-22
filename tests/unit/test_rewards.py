from agentbox.tasks.rewards import shaped_reward
from agentbox.tasks.verifier import VerifyResult


def test_shaped_reward_step_penalty() -> None:
    v = VerifyResult(
        reward=1.0,
        success=True,
        exit_code=0,
        stdout="",
        stderr="",
        duration_s=0.1,
        command="true",
    )
    r = shaped_reward(v, steps=10, step_penalty=0.01)
    assert abs(r - 0.9) < 1e-9
