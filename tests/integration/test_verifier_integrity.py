"""Reward integrity, partial credit, setup cache and per-task envs (Docker)."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from agentbox import Rollout, Task
from agentbox.config import AgentConfig, ModelConfig, ResourceLimits, RolloutConfig, SandboxConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.rewards import RewardSpec
from agentbox.tasks.schema import PYTEST_TAMPER_FILES, EnvironmentSpec, VerifierSpec
from agentbox.trajectory.schema import FunctionCall, ToolCall
from agentbox.types import FailureKind, FinalStatus, VerifierType

pytestmark = pytest.mark.docker

BROKEN = "def add(a, b):\n    return a - b\n"
TESTS = (
    "from calc import add\n\n"
    "def test_one():\n    assert add(1, 1) == 2\n\n"
    "def test_two():\n    assert add(2, 2) == 4\n\n"
    "def test_zero():\n    assert add(0, 0) == 0\n"
)


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _need_docker() -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")


def _sandbox(**kw: Any) -> SandboxConfig:
    data: dict[str, Any] = {"ensure_pytest": True, "limits": ResourceLimits(network_disabled=True)}
    data.update(kw)
    return SandboxConfig(**data)


def _writes(*files: tuple[str, str]) -> MockModelClient:
    calls = [
        ToolCall(
            id=f"c{i}",
            function=FunctionCall(
                name="write_file", arguments=json.dumps({"path": p, "content": c})
            ),
        )
        for i, (p, c) in enumerate(files)
    ]
    return MockModelClient([ModelResponse(tool_calls=calls), ModelResponse(content="done")])


def _task(**verifier: Any) -> Task:
    return Task(
        task_id="calc",
        description="Fix add() in calc.py so the tests pass.",
        starter_files={"calc.py": BROKEN, "test_calc.py": TESTS},
        verifier=VerifierSpec(type=VerifierType.PYTEST, command="python -m pytest -q", **verifier),
    )


async def test_rewriting_protected_tests_is_caught() -> None:
    hack = _writes(("test_calc.py", "def test_ok():\n    assert True\n"))
    traj = await Rollout.run(_task(protected_paths=["test_*.py"]), model=hack, sandbox=_sandbox())
    assert traj.final_status != FinalStatus.SUCCESS
    assert traj.reward == 0.0
    assert traj.verify is not None and traj.verify.tampered == ["test_calc.py"]
    # without protection the same hack "passes" (documents why protection matters)
    naive = await Rollout.run(_task(), model=hack, sandbox=_sandbox())
    assert naive.final_status == FinalStatus.SUCCESS


async def test_planted_conftest_is_removed_and_flagged() -> None:
    conftest = (
        "import pytest\n"
        "def pytest_collection_modifyitems(items):\n"
        "    for it in items:\n"
        "        it.add_marker(pytest.mark.skip(reason='hacked'))\n"
    )
    hack = _writes(("conftest.py", conftest))
    traj = await Rollout.run(
        _task(forbid_new_files=list(PYTEST_TAMPER_FILES)), model=hack, sandbox=_sandbox()
    )
    assert traj.verify is not None and "conftest.py" in traj.verify.tampered
    assert traj.final_status != FinalStatus.SUCCESS


async def test_hidden_tests_and_honest_fix() -> None:
    fix = _writes(("calc.py", "def add(a, b):\n    return a + b\n"))
    task = Task(
        task_id="hidden",
        description="Implement add() in calc.py.",
        starter_files={"calc.py": BROKEN},
        verifier=VerifierSpec(
            type=VerifierType.PYTEST,
            command="python -m pytest -q test_hidden.py",
            hidden_files={"test_hidden.py": TESTS},
        ),
    )
    traj = await Rollout.run(task, model=fix, sandbox=_sandbox())
    assert traj.final_status == FinalStatus.SUCCESS, traj.verify
    assert traj.verify is not None and traj.verify.tests_passed == 3


async def test_pytest_fraction_partial_credit() -> None:
    # add(a, b) = a * b passes add(2,2)==4 and add(0,0)==0 but not add(1,1)==2
    partial = _writes(("calc.py", "def add(a, b):\n    return a * b\n"))
    traj = await Rollout.run(
        _task(reward_mode="pytest_fraction"), model=partial, sandbox=_sandbox()
    )
    assert traj.verify is not None
    assert traj.verify.tests_total == 3 and traj.verify.tests_passed == 2
    assert abs(traj.reward - 2 / 3) < 1e-6
    assert traj.final_status != FinalStatus.SUCCESS


async def test_stdout_reward_and_reward_breakdown() -> None:
    task = Task(
        task_id="score",
        description="noop",
        verifier=VerifierSpec(
            type=VerifierType.COMMAND,
            command='echo progress; echo \'{"reward": 0.25, "note": "x"}\'',
            reward_mode="stdout",
        ),
    )
    cfg = RolloutConfig(
        model=ModelConfig(model="mock"),
        sandbox=_sandbox(ensure_pytest=False),
        rewards=[
            RewardSpec(name="verifier"),
            RewardSpec(name="step_penalty", params={"per_step": 0.1}),
        ],
    )
    traj = await Rollout.run(
        task, model=MockModelClient([ModelResponse(content="done")]), config=cfg
    )
    assert traj.reward_breakdown == {"verifier": 0.25, "step_penalty": -0.0}
    assert abs(traj.reward - 0.25) < 1e-9


async def test_setup_snapshot_cache_and_task_environment() -> None:
    task = Task(
        task_id="cached-setup",
        description="check env",
        starter_files={"a.txt": "a"},
        binary_files={"blob.bin": base64.b64encode(bytes(range(256))).decode()},
        setup_commands=["echo cached > marker.txt"],
        environment=EnvironmentSpec(env={"TASK_VAR": "hello"}),
        verifier=VerifierSpec(
            type=VerifierType.COMMAND,
            command=(
                'test "$(cat marker.txt)" = cached && test "$TASK_VAR" = hello '
                "&& python -c \"import sys; d=open('blob.bin','rb').read(); sys.exit(0 if d==bytes(range(256)) else 1)\""
            ),
        ),
    )
    noop = MockModelClient([ModelResponse(content="done")])
    first = await Rollout.run(task, model=noop, sandbox=_sandbox(ensure_pytest=False))
    second = await Rollout.run(task, model=noop, sandbox=_sandbox(ensure_pytest=False))
    for traj in (first, second):
        assert traj.final_status == FinalStatus.SUCCESS, (traj.error, traj.verify)
    assert second.metrics.seed_s is not None and second.metrics.seed_s < 2.0


async def test_failing_setup_is_seed_failure() -> None:
    task = Task(
        task_id="bad-setup",
        description="x",
        setup_commands=["exit 3"],
        verifier=VerifierSpec(type=VerifierType.COMMAND, command="true"),
    )
    traj = await Rollout.run(
        task,
        model=MockModelClient([ModelResponse(content="done")]),
        sandbox=_sandbox(ensure_pytest=False),
    )
    assert traj.failure_kind == FailureKind.SEED and not traj.is_trainable


async def test_agent_config_penalties_still_apply() -> None:
    cfg = RolloutConfig(
        model=ModelConfig(model="mock"),
        sandbox=_sandbox(),
        agent=AgentConfig(step_penalty=0.1),
    )
    fix = _writes(("calc.py", "def add(a, b):\n    return a + b\n"))
    traj = await Rollout.run(_task(), model=fix, config=cfg)
    assert traj.final_status == FinalStatus.SUCCESS
    assert abs(traj.reward - 0.9) < 1e-9
