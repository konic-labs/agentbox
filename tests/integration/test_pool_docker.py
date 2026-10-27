"""Warm sandbox pool against a real Docker daemon (GH #23)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from agentbox.config import ResourceLimits, SandboxConfig
from agentbox.sandbox.backend import DockerBackend
from agentbox.sandbox.manager import SandboxManager
from agentbox.sandbox.pool import POOL_LABEL, pool_for

pytestmark = pytest.mark.docker


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture
async def backend() -> AsyncIterator[DockerBackend]:
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    b = DockerBackend()
    try:
        yield b
    finally:
        pool_for(b).close_sync()
        b.close()


def _cfg(**kw: object) -> SandboxConfig:
    data: dict[str, object] = {
        "ensure_pytest": False,
        "limits": ResourceLimits(network_disabled=True),
        "workspace_dir": "/workspace/project",  # pooled containers pre-create it
        "warm_pool": 2,
    }
    data.update(kw)
    return SandboxConfig(**data)  # type: ignore[arg-type]


async def _wait_ready(backend: DockerBackend, n: int, timeout: float = 30.0) -> None:
    pool = pool_for(backend)
    deadline = time.monotonic() + timeout
    while pool.ready_count() < n:
        if time.monotonic() > deadline:
            raise AssertionError(f"pool never reached {n} ready (stats={pool.stats})")
        await asyncio.sleep(0.05)


async def test_pool_hands_out_fresh_containers_once(backend: DockerBackend) -> None:
    mgr = SandboxManager(_cfg(), backend=backend)
    first = await mgr.create(task_id="pool-a", run_id="r1")  # miss; warms the shape
    await _wait_ready(backend, 2)
    pooled = await mgr.create(task_id="pool-b", run_id="r1")
    pool = pool_for(backend)
    assert pool.stats.hits == 1 and pool.stats.misses == 1
    assert pooled.name.startswith("agentbox-pool-b-")
    info = backend.client.api.inspect_container(pooled.id)
    assert info["Name"].lstrip("/") == pooled.name
    assert info["Config"]["Labels"][POOL_LABEL] == "1"
    assert "/workspace/project" in (await mgr.exec(pooled, "pwd; ls -d /workspace/project")).stdout

    await mgr.write_file(pooled, "secret.txt", "from rollout b")
    await mgr.destroy(pooled)
    await _wait_ready(backend, 2)
    nxt = await mgr.create(task_id="pool-c", run_id="r1")
    assert nxt.id not in {first.id, pooled.id}
    probe = await mgr.exec(nxt, "test -e /workspace/project/secret.txt && echo LEAK || echo clean")
    assert probe.stdout.strip() == "clean"
    for sb in (first, nxt):
        await mgr.destroy(sb)


async def test_pool_shapes_are_isolated(backend: DockerBackend) -> None:
    a = SandboxManager(_cfg(env={"MODE": "a"}), backend=backend)
    b = SandboxManager(_cfg(env={"MODE": "b"}), backend=backend)
    await a.warm(1)
    await _wait_ready(backend, 1)
    sb = await b.create(task_id="other-shape", run_id="r2")
    assert pool_for(backend).stats.hits == 0  # MODE=a containers never serve MODE=b
    assert (await b.exec(sb, "echo $MODE")).stdout.strip() == "b"
    await b.destroy(sb)


async def test_pool_expires_idle_and_close_removes(backend: DockerBackend) -> None:
    mgr = SandboxManager(_cfg(warm_pool_ttl_s=0.5), backend=backend)
    await mgr.warm(2)
    await _wait_ready(backend, 2)
    pool = pool_for(backend)
    idle = [cid for s in pool._shapes.values() for cid, _ in s.ready]
    await asyncio.sleep(0.8)
    sb = await mgr.create(task_id="after-ttl", run_id="r3")
    assert pool.stats.expired == 2 and pool.stats.hits == 0
    await mgr.destroy(sb)

    deadline = time.monotonic() + 15
    while any(cid in backend.live_containers() for cid in idle):
        assert time.monotonic() < deadline, "expired containers were not removed"
        await asyncio.sleep(0.05)
    await _wait_ready(backend, 2)
    assert pool.close_sync() == 2
    assert pool.ready_count() == 0
    assert not backend.list_containers_sync([f"{POOL_LABEL}=1"])
