"""End-to-end mock rollout with Docker."""

from __future__ import annotations

import json

import pytest

from agentbox import Rollout, Task
from agentbox.config import ResourceLimits, SandboxConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.tasks.schema import VerifierSpec
from agentbox.trajectory.schema import FunctionCall, ToolCall
from agentbox.types import FinalStatus, VerifierType

pytestmark = pytest.mark.docker


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_mock_fizzbuzz_rollout() -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")

    task = Task(
        task_id="fix_fizzbuzz_test",
        description="Fix fizzbuzz",
        starter_files={
            "fizzbuzz.py": "def fizzbuzz(n):\n    return str(n)\n",
            "test_fizzbuzz.py": (
                "from fizzbuzz import fizzbuzz\n\n"
                "def test_3():\n    assert fizzbuzz(3) == 'Fizz'\n"
            ),
        },
        setup_commands=["pip install -q pytest"],
        verifier=VerifierSpec(
            type=VerifierType.PYTEST,
            command="python -m pytest -q",
            timeout_s=60,
        ),
    )
    fixed = (
        "def fizzbuzz(n):\n"
        "    if n % 3 == 0:\n"
        "        return 'Fizz'\n"
        "    return str(n)\n"
    )
    mock = MockModelClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(
                            name="write_file",
                            arguments=json.dumps(
                                {"path": "fizzbuzz.py", "content": fixed}
                            ),
                        ),
                    )
                ],
            ),
            ModelResponse(content="done", tool_calls=[]),
        ]
    )
    sandbox = SandboxConfig(
        image="python:3.12-slim-bookworm",
        ensure_pytest=False,
        limits=ResourceLimits(network_disabled=False, memory_mb=512),
    )
    traj = await Rollout.run(task=task, model=mock, sandbox=sandbox)
    assert traj.final_status == FinalStatus.SUCCESS
    assert traj.reward == 1.0
    assert traj.metrics.steps >= 1
    # Official verifier ground truth (P0.2)
    assert traj.metadata.get("verify_success") is True
    assert traj.metadata.get("verify_exit_code") == 0
    assert "pytest" in (traj.metadata.get("verify_command") or "")
    assert "verify_stdout" in traj.metadata
    assert "verify_stderr" in traj.metadata
