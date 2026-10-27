"""Runner behaviour in real sandboxes: kept sandboxes and concurrency stress (GH #21, #94)."""

from __future__ import annotations

import json
import time

import pytest

from agentbox import Rollout, Task
from agentbox.config import ResourceLimits, SandboxConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.runner.parallel import ParallelRunner
from agentbox.sandbox.backend import get_backend
from agentbox.tasks.schema import VerifierSpec
from agentbox.trajectory.io import MemorySink
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


@pytest.fixture(autouse=True)
def _need_docker() -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")


def _cfg(**kw: object) -> SandboxConfig:
    data: dict[str, object] = {
        "ensure_pytest": False,
        "limits": ResourceLimits(network_disabled=True, memory_mb=256),
    }
    data.update(kw)
    return SandboxConfig(**data)  # type: ignore[arg-type]


def _task(task_id: str, command: str) -> Task:
    return Task(
        task_id=task_id,
        description="write out.txt",
        verifier=VerifierSpec(type=VerifierType.COMMAND, command=command),
    )


def _writer() -> MockModelClient:
    call = ToolCall(
        id="w",
        function=FunctionCall(
            name="write_file", arguments=json.dumps({"path": "out.txt", "content": "hello\n"})
        ),
    )
    return MockModelClient([ModelResponse(tool_calls=[call]), ModelResponse(content="done")])


async def test_keep_on_failure_keeps_only_failed_sandboxes() -> None:
    backend = get_backend()
    cfg = _cfg(keep_on_failure=True)
    ok = await Rollout.run(_task("keep-ok", "grep -q hello out.txt"), model=_writer(), sandbox=cfg)
    bad = await Rollout.run(_task("keep-bad", "grep -q nope out.txt"), model=_writer(), sandbox=cfg)
    assert ok.final_status == FinalStatus.SUCCESS and "kept_sandbox" not in ok.metadata
    assert bad.final_status == FinalStatus.FAILED
    name = bad.metadata["kept_sandbox"]
    try:
        info = backend.client.api.inspect_container(name)
        assert info["State"]["Running"]
        assert info["Id"] not in backend.live_containers()  # survives exit cleanup
        assert not backend.list_containers_sync([f"agentbox.rollout_id={ok.rollout_id}"])
    finally:
        backend.client.api.remove_container(name, force=True, v=True)


async def test_32_concurrent_rollouts_complete_and_clean_up() -> None:
    backend = get_backend()
    before = set(backend.live_containers())
    tasks = [_task(f"stress-{i}", "grep -q hello out.txt") for i in range(32)]
    runner = ParallelRunner(model=_writer(), sandbox=_cfg(), concurrency=32)
    sink = MemorySink()
    t0 = time.monotonic()
    summary = await runner.run_to_sink(tasks, sink)
    elapsed = time.monotonic() - t0
    assert summary.n == 32 and summary.metrics.success_rate == 1.0, summary
    assert len({t.rollout_id for t in sink.trajectories}) == 32
    assert set(backend.live_containers()) == before  # every sandbox removed
    assert elapsed < 120, f"32 concurrent rollouts took {elapsed:.1f}s"
