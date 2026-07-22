"""Docker integration tests for SandboxManager."""

from __future__ import annotations

import uuid

import pytest

from agentbox.config import ResourceLimits, SandboxConfig
from agentbox.sandbox.manager import SandboxManager

pytestmark = pytest.mark.docker


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture
async def manager() -> SandboxManager:
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    cfg = SandboxConfig(
        image="python:3.12-slim-bookworm",
        auto_pull=True,
        ensure_pytest=False,
        limits=ResourceLimits(network_disabled=True, memory_mb=256, cpu_count=0.5),
    )
    return SandboxManager(cfg)


@pytest.mark.asyncio
async def test_create_write_exec_destroy(manager: SandboxManager) -> None:
    run_id = str(uuid.uuid4())
    sandbox = await manager.create(task_id="test-task", run_id=run_id)
    try:
        await manager.write_file(sandbox, "hello.txt", "world\n")
        content = await manager.read_file(sandbox, "hello.txt")
        assert content == "world\n"

        result = await manager.exec(sandbox, "python -c 'print(1+1)'")
        assert result.exit_code == 0
        assert "2" in result.stdout

        await manager.write_files(
            sandbox,
            {"src/a.py": "x=1\n", "src/b.py": "y=2\n"},
        )
        listed = await manager.list_dir(sandbox, "src")
        assert "a.py" in listed
        assert "b.py" in listed
    finally:
        await manager.destroy(sandbox)
