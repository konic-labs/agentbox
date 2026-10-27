"""ART adapter: shapes, verifier-derived correctness, scalar metadata, Choices (GH #67, #75)."""

from __future__ import annotations

import json

import pytest

from agentbox.integrations.art import art_choice, to_art_dict
from agentbox.trajectory.schema import (
    FunctionCall,
    Message,
    ToolCall,
    Trajectory,
    TrajectoryMetrics,
    VerifyInfo,
)
from agentbox.trajectory.tokens import TurnTokens
from agentbox.types import FinalStatus, MessageRole


def _traj(**kw: object) -> Trajectory:
    data: dict[str, object] = {
        "task_id": "t1",
        "run_id": "r1",
        "group_id": "g1",
        "messages": [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="fix it"),
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=FunctionCall(name="read_file", arguments='{"path": "a.py"}'),
                    )
                ],
                finish_reason="tool_calls",
                tokens=TurnTokens(
                    completion_ids=[11, 12, 13], logprobs=[-0.5, -0.25, 0.0], prompt_ids=[1, 2, 3]
                ),
            ),
            Message(role=MessageRole.TOOL, content="x = 1", tool_call_id="c1", name="read_file"),
            Message(
                role=MessageRole.ASSISTANT,
                content="done",
                # reuses the whole previous sequence, then the tool-result delta
                tokens=TurnTokens(
                    completion_ids=[21], logprobs=[-0.1], prompt_reuse=6, prompt_ids=[30, 31]
                ),
            ),
            Message(role=MessageRole.ASSISTANT, content="bye"),  # no token data
        ],
        "reward": 1.0,
        "reward_breakdown": {"verifier": 1.0, "step_penalty": -0.1},
        "final_status": FinalStatus.SUCCESS,
        "metrics": TrajectoryMetrics(steps=2, tool_calls=1, duration_s=1.5, model_calls=2),
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "metadata": {"difficulty": "easy", "nested": {"a": [1, 2]}},
    }
    data.update(kw)
    return Trajectory(**data)  # type: ignore[arg-type]


def test_shape_metrics_and_scalar_metadata() -> None:
    art = to_art_dict(_traj())
    assert art["reward"] == 1.0 and art["tools"][0]["function"]["name"] == "read_file"
    assert art["metrics"]["correct"] == 1.0
    assert art["metrics"]["reward/step_penalty"] == -0.1 and art["metrics"]["model_calls"] == 2
    meta = art["metadata"]
    assert meta["task_id"] == "t1" and meta["group_id"] == "g1"
    assert all(isinstance(v, (str, int, float, bool, type(None))) for v in meta.values())
    assert json.loads(meta["nested"]) == {"a": [1, 2]}
    assert to_art_dict(_traj(), scalar_metadata=False)["metadata"]["nested"] == {"a": [1, 2]}


def test_correct_comes_from_the_verifier() -> None:
    # the loop "succeeded" but the verifier failed: not correct
    traj = _traj(verify=VerifyInfo(success=False, reward=0.0, exit_code=1))
    assert to_art_dict(traj)["metrics"]["correct"] == 0.0
    failed_loop = _traj(
        final_status=FinalStatus.MAX_STEPS, verify=VerifyInfo(success=True, reward=1.0)
    )
    assert to_art_dict(failed_loop)["metrics"]["correct"] == 1.0


def test_assistant_turns_with_token_data_become_choices() -> None:
    items = to_art_dict(_traj())["messages_and_choices"]
    roles = [i.get("role") for i in items]
    assert roles == ["system", "user", None, "tool", None, "assistant"]
    first, second = items[2], items[4]
    assert first["finish_reason"] == "tool_calls"
    assert first["message"]["tool_calls"][0]["function"]["name"] == "read_file"
    tokens = first["logprobs"]["content"]
    assert [t["token"] for t in tokens] == ["token_id:11", "token_id:12", "token_id:13"]
    assert [t["logprob"] for t in tokens] == [-0.5, -0.25, 0.0]
    # vLLM fields ART assembles sequences from; the second prompt is rebuilt
    assert first["prompt_token_ids"] == [1, 2, 3] and first["token_ids"] == [11, 12, 13]
    assert second["prompt_token_ids"] == [1, 2, 3, 11, 12, 13, 30, 31]
    assert second["token_ids"] == [21] and second["finish_reason"] == "stop"
    # opt out: plain messages only
    plain = to_art_dict(_traj(), choices=False)["messages_and_choices"]
    assert all("role" in i for i in plain)


def test_incomplete_token_data_stays_a_message() -> None:
    msg = Message(
        role=MessageRole.ASSISTANT,
        content="hi",
        tokens=TurnTokens(completion_ids=[1, 2], logprobs=None),
    )
    assert art_choice(msg, [7]) is None
    msg.tokens = TurnTokens(completion_ids=[1, 2], logprobs=[-0.1])  # length mismatch
    assert art_choice(msg, [7]) is None
    msg.tokens = TurnTokens(completion_ids=[1, 2], logprobs=[-0.1, -0.2])
    assert art_choice(msg, None) is None  # prompt unknown: ART could not place it
    assert art_choice(msg, [7]) is not None


def test_deprecated_trajectory_methods_warn_and_delegate() -> None:
    traj = _traj()
    with pytest.warns(DeprecationWarning, match="integrations.art"):
        legacy = traj.to_art_dict()
    assert legacy == to_art_dict(traj)


def test_live_art_objects_when_installed() -> None:
    pytest.importorskip("art")
    from agentbox.integrations.art import to_art_groups

    trajs = [_traj(rollout_id=f"x{i}", reward=float(i % 2)) for i in range(4)]
    groups = to_art_groups(trajs)
    assert len(groups) == 1 and len(groups[0].trajectories) == 4
