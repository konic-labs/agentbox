"""Docker integration: benchmark suite with mock model."""

from __future__ import annotations

import json
import uuid

import pytest

from agentbox.benchmark.loader import create_suite_from_tasks
from agentbox.benchmark.runner import BenchmarkRunner
from agentbox.benchmark.schema import BenchmarkRunConfig, ModelUnderTest
from agentbox.config import ModelConfig, ResourceLimits, SandboxConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.trajectory.schema import FunctionCall, ToolCall
from agentbox.types import FinalStatus, VerifierType

pytestmark = pytest.mark.docker


def _docker_ok() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_benchmark_mock_success(tmp_path) -> None:
    if not _docker_ok():
        pytest.skip("Docker not available")

    tasks_src = tmp_path / "tasks"
    tdir = tasks_src / "fix_one"
    tdir.mkdir(parents=True)
    Task(
        task_id="fix_one",
        description="fix fizz",
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
    ).save_json(tdir / "task.json")

    suite_dir = tmp_path / "suite"
    suite = create_suite_from_tasks(
        tasks_src,
        suite_dir,
        suite_id="mini",
        name="Mini",
        freeze=True,
        sandbox=SandboxConfig(
            limits=ResourceLimits(network_disabled=False, memory_mb=512),
            ensure_pytest=False,
        ),
    )
    # inject python setup check that passes
    from agentbox.benchmark.schema import SetupCheckSpec

    suite.manifest.scoring.setup_checks = [
        SetupCheckSpec(name="py", command="python -c 'print(1)'"),
    ]
    suite = suite.freeze()

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
        * 3
    )

    # Run via ParallelRunner path used by benchmark but inject mock
    from agentbox.benchmark.report import build_leaderboard, build_model_report
    from agentbox.benchmark.schema import BenchmarkReport
    from agentbox.config import AgentConfig, RolloutConfig
    from agentbox.runner.parallel import ParallelRunner
    from datetime import datetime, timezone
    import time

    agent = suite.manifest.agent.model_copy(update={"custom_tools": []})
    runner = ParallelRunner(
        concurrency=1,
        config=RolloutConfig(
            sandbox=suite.manifest.sandbox,
            model=ModelConfig(model="mock"),
            agent=agent,
        ),
        model=mock,
        agent=agent,
        sandbox=suite.manifest.sandbox,
    )
    t0 = time.monotonic()
    trajs = await runner.run_tasks(
        suite.tasks,
        n_per_task=1,
        progress=False,
        setup_checks=list(suite.manifest.scoring.setup_checks),
    )
    wall = time.monotonic() - t0
    assert trajs[0].final_status == FinalStatus.SUCCESS
    mr = build_model_report("mock", {"model": "mock"}, trajs, wall_clock_s=wall)
    assert mr.aggregate.success_rate == 1.0
