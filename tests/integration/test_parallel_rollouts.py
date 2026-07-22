"""Parallel rollouts with Docker + mock models."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentbox import Rollout, Task
from agentbox.config import ResourceLimits, SandboxConfig
from agentbox.metrics import aggregate_trajectories
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


def _mock(fixed: str) -> MockModelClient:
    return MockModelClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
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


@pytest.mark.asyncio
async def test_three_parallel_mock_rollouts() -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")

    task = Task(
        task_id="parallel_fizz",
        description="fix",
        starter_files={
            "fizzbuzz.py": "def fizzbuzz(n):\n    return str(n)\n",
            "test_fizzbuzz.py": (
                "from fizzbuzz import fizzbuzz\n\n"
                "def test_3():\n    assert fizzbuzz(3) == 'Fizz'\n"
            ),
        },
        setup_commands=["pip install -q pytest"],
        verifier=VerifierSpec(
            type=VerifierType.PYTEST, command="python -m pytest -q", timeout_s=60
        ),
    )
    fixed = (
        "def fizzbuzz(n):\n"
        "    if n % 3 == 0:\n"
        "        return 'Fizz'\n"
        "    return str(n)\n"
    )
    sandbox = SandboxConfig(
        limits=ResourceLimits(network_disabled=False, memory_mb=512),
        ensure_pytest=False,
    )

    async def one():
        return await Rollout.run(task=task, model=_mock(fixed), sandbox=sandbox)

    trajs = await asyncio.gather(one(), one(), one())
    assert all(t.final_status == FinalStatus.SUCCESS for t in trajs)
    stats = aggregate_trajectories(trajs)
    assert stats.success_rate == 1.0
