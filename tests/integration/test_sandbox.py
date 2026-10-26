"""Docker integration tests for SandboxManager / DockerBackend."""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from agentbox.config import ResourceLimits, SandboxConfig
from agentbox.errors import PathEscapeError
from agentbox.sandbox.backend import get_backend
from agentbox.sandbox.manager import SandboxManager
from agentbox.sandbox.prune import prune_agentbox_containers

pytestmark = pytest.mark.docker

COUNT_PROCS = (
    "n=0; for p in /proc/[0-9]*; do "
    "tr '\\0' ' ' < $p/cmdline 2>/dev/null | grep -q '^sleep {secs}' && n=$((n+1)); "
    "done; echo $n"
)


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def _config(**overrides: object) -> SandboxConfig:
    data: dict[str, object] = {
        "image": "python:3.12-slim-bookworm",
        "auto_pull": True,
        "ensure_pytest": False,
        "limits": ResourceLimits(network_disabled=True, memory_mb=256, cpu_count=0.5),
    }
    data.update(overrides)
    return SandboxConfig(**data)  # type: ignore[arg-type]


@pytest.fixture
async def manager() -> SandboxManager:
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    return SandboxManager(_config())


@pytest.fixture
async def sandbox(manager: SandboxManager):  # type: ignore[no-untyped-def]
    sb = await manager.create(task_id="test-task", run_id=str(uuid.uuid4()))
    try:
        yield sb
    finally:
        await manager.destroy(sb)


async def test_create_write_exec_destroy(manager: SandboxManager) -> None:
    sb = await manager.create(task_id="test-task", run_id=str(uuid.uuid4()))
    try:
        await manager.write_file(sb, "hello.txt", "world\n")
        assert await manager.read_file(sb, "hello.txt") == "world\n"

        result = await manager.exec(sb, "python -c 'print(1+1)'")
        assert result.exit_code == 0
        assert "2" in result.stdout

        await manager.write_files(sb, {"src/a.py": "x=1\n", "src/b.py": "y=2\n"})
        listed = await manager.list_dir(sb, "src")
        assert "a.py" in listed and "b.py" in listed
        tree = await manager.list_dir(sb, ".", recursive=True)
        assert "src/" in tree and "src/a.py" in tree
    finally:
        t0 = time.monotonic()
        await manager.destroy(sb)
        assert time.monotonic() - t0 < 2.0, "destroy must not wait for a stop timeout"


async def test_timeout_kills_process(manager: SandboxManager, sandbox) -> None:  # type: ignore[no-untyped-def]
    t0 = time.monotonic()
    res = await manager.exec(sandbox, "sleep 31; echo done > /tmp/marker", timeout_s=1.5)
    assert res.timed_out and res.exit_code == -1
    assert time.monotonic() - t0 < 6.0
    alive = await manager.exec(sandbox, COUNT_PROCS.format(secs=31), timeout_s=10)
    assert alive.stdout.strip() == "0", "timed-out process must be killed"


async def test_concurrent_timeouts_do_not_starve(manager: SandboxManager) -> None:
    sandboxes = await asyncio.gather(
        *(manager.create(task_id=f"t{i}", run_id="starve") for i in range(4))
    )
    try:
        await asyncio.gather(
            *(manager.exec(sb, "sleep 600", timeout_s=1.0) for sb in sandboxes for _ in range(6))
        )
        t0 = time.monotonic()
        res = await manager.exec(sandboxes[0], "echo hi", timeout_s=30)
        assert res.stdout.strip() == "hi"
        assert time.monotonic() - t0 < 3.0
    finally:
        await asyncio.gather(*(manager.destroy(sb) for sb in sandboxes))


async def test_output_is_bounded(manager: SandboxManager, sandbox) -> None:  # type: ignore[no-untyped-def]
    t0 = time.monotonic()
    res = await manager.exec(
        sandbox,
        "python -c \"import sys; sys.stdout.write('a'*20_000_000 + 'END')\"",
        timeout_s=60,
        max_output_bytes=4096,
    )
    assert res.exit_code == 0
    assert res.stdout_truncated
    assert res.stdout.endswith("END")
    assert len(res.stdout) < 4096 + 100
    assert res.output_bytes >= 20_000_000
    assert time.monotonic() - t0 < 30


async def test_output_overflow_kills(sandbox) -> None:  # type: ignore[no-untyped-def]
    mgr = SandboxManager(_config(kill_output_bytes=2_000_000))
    res = await mgr.backend.exec(
        sandbox.id,
        ["/bin/sh", "-c", "yes"],
        timeout_s=30,
        max_output_bytes=1024,
        kill_output_bytes=2_000_000,
    )
    assert res.killed
    assert not res.timed_out


async def test_hostile_paths_and_listing_errors(manager: SandboxManager, sandbox) -> None:  # type: ignore[no-untyped-def]
    weird = "dir with space/a'b$(id)`x`.txt"
    await manager.write_file(sandbox, weird, "ok")
    assert await manager.read_file(sandbox, weird) == "ok"
    listed = await manager.list_dir(sandbox, "dir with space")
    assert "a'b$(id)`x`.txt" in listed
    with pytest.raises(FileNotFoundError):
        await manager.list_dir(sandbox, "does-not-exist; touch /tmp/pwned")
    probe = await manager.exec(sandbox, "test -e /tmp/pwned && echo yes || echo no")
    assert probe.stdout.strip() == "no"


async def test_read_limits_dirs_and_symlinks(manager: SandboxManager, sandbox) -> None:  # type: ignore[no-untyped-def]
    await manager.write_file(sandbox, "big.txt", "x" * 10_000)
    data = await manager.read_bytes(sandbox, "big.txt", max_bytes=100)
    assert len(data) == 100
    await manager.exec(sandbox, "mkdir -p d && ln -s big.txt link.txt && ln -s /etc/passwd out")
    with pytest.raises(IsADirectoryError):
        await manager.read_file(sandbox, "d")
    assert (await manager.read_file(sandbox, "link.txt")).startswith("xxx")
    with pytest.raises(PathEscapeError):
        await manager.read_file(sandbox, "out")
    with pytest.raises(FileNotFoundError):
        await manager.read_file(sandbox, "missing.txt")


async def test_security_defaults_and_labels(manager: SandboxManager, sandbox) -> None:  # type: ignore[no-untyped-def]
    info = manager.backend.client.api.inspect_container(sandbox.id)
    host = info["HostConfig"]
    assert host["CapDrop"] == ["ALL"]
    assert "no-new-privileges:true" in (host["SecurityOpt"] or [])
    assert host["Init"] is True
    assert host["NetworkMode"] == "none"
    labels = info["Config"]["Labels"]
    assert labels["agentbox"] == "1" and labels["agentbox.owner"]
    # pip/apt-style operations still work with the reduced capability set
    res = await manager.exec(sandbox, "touch /usr/local/x && chown nobody /usr/local/x && echo ok")
    assert res.stdout.strip() == "ok", res.stderr


async def test_prune_orphans_spares_live_sandboxes(sandbox) -> None:  # type: ignore[no-untyped-def]
    removed = prune_agentbox_containers(scope="orphans", dry_run=True)
    assert sandbox.id not in removed


async def test_ensure_pytest_builds_cached_image_once() -> None:
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    mgr = SandboxManager(_config(ensure_pytest=True))
    sb = await mgr.create(task_id="pytest-check", run_id="p")
    try:
        assert sb.image.startswith("agentbox-cache:")
        res = await mgr.exec(sb, "python -m pytest --version", timeout_s=60)
        assert res.exit_code == 0, res.stderr
    finally:
        await mgr.destroy(sb)
    t0 = time.monotonic()
    sb2 = await mgr.create(task_id="pytest-check", run_id="p")
    await mgr.destroy(sb2)
    assert time.monotonic() - t0 < 5.0, "second create must reuse the cached image"
    assert sb2.image == sb.image


async def test_backend_tracks_live_containers(manager: SandboxManager) -> None:
    backend = get_backend()
    sb = await manager.create(task_id="track", run_id="r")
    assert sb.id in backend.live_containers()
    await manager.destroy(sb)
    assert sb.id not in backend.live_containers()
