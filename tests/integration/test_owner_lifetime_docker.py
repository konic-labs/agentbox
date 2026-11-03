"""Sandboxes die with the process that created them, even by SIGKILL (GH #111)."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
import uuid

import pytest

pytestmark = pytest.mark.docker

CHILD = textwrap.dedent(
    """
    import asyncio, sys, time
    from agentbox.config import SandboxConfig
    from agentbox.sandbox.manager import SandboxManager

    async def main(run, bound, keep, pool):
        cfg = SandboxConfig(ensure_pytest=False, die_with_owner=bound, warm_pool=pool)
        mgr = SandboxManager(cfg)
        boxes = [await mgr.create(task_id="owner-test", run_id=run) for _ in range(2)]
        if keep:
            mgr.keep(boxes[0])
        await asyncio.sleep(3 if pool else 0.2)  # let the warm pool refill
        print(" ".join(b.id for b in boxes), flush=True)
        time.sleep(300)

    run, bound, keep, pool = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1", int(sys.argv[4])
    asyncio.run(main(run, bound, keep, pool))
    """
)


def _docker() -> object:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return client
    except Exception:
        pytest.skip("Docker is not available")


def _owned_by(client: object, pid: int) -> list[dict]:
    owner = f"agentbox.owner={socket.gethostname()}:{pid}"
    return list(client.api.containers(all=True, filters={"label": [owner]}))  # type: ignore[attr-defined]


def _child(*, bound: bool, keep: bool = False, pool: int = 0) -> tuple[subprocess.Popen, list[str]]:
    run = f"owner-{uuid.uuid4().hex[:8]}"
    env = {**os.environ, "AGENTBOX_REAP_ORPHANS": "0"}
    args = [sys.executable, "-c", CHILD, run, str(int(bound)), str(int(keep)), str(pool)]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, text=True, env=env)
    assert proc.stdout is not None
    ids = proc.stdout.readline().split()
    assert len(ids) == 2, "the child did not create its sandboxes"
    return proc, ids


def _sigkill(proc: subprocess.Popen) -> None:
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(10)


def _wait_until(check, timeout: float = 15.0) -> bool:  # type: ignore[no-untyped-def]
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.1)
    return False


def test_sandboxes_and_warm_pool_die_with_a_killed_owner() -> None:
    client = _docker()
    proc, _ = _child(bound=True, pool=2)
    try:
        assert len(_owned_by(client, proc.pid)) >= 3  # two sandboxes and the refilled pool
        t0 = time.monotonic()
        _sigkill(proc)
        assert _wait_until(lambda: not _owned_by(client, proc.pid)), "bound sandboxes outlived"
        assert time.monotonic() - t0 < 15
    finally:
        if proc.poll() is None:
            proc.kill()
        for c in _owned_by(client, proc.pid):
            client.api.remove_container(c["Id"], force=True)  # type: ignore[attr-defined]


def test_a_kept_sandbox_outlives_its_owner() -> None:
    client = _docker()
    proc, ids = _child(bound=True, keep=True)
    kept, other = ids
    try:
        _sigkill(proc)
        assert _wait_until(lambda: [c["Id"] for c in _owned_by(client, proc.pid)] == [kept])
        time.sleep(1)
        state = client.api.inspect_container(kept)["State"]  # type: ignore[attr-defined]
        assert state["Running"], "a kept sandbox must stay up"
    finally:
        if proc.poll() is None:
            proc.kill()
        for c in _owned_by(client, proc.pid):
            client.api.remove_container(c["Id"], force=True)  # type: ignore[attr-defined]


def test_unbound_sandboxes_wait_for_a_prune() -> None:
    client = _docker()
    proc, ids = _child(bound=False)
    try:
        _sigkill(proc)
        time.sleep(2)
        assert {c["Id"] for c in _owned_by(client, proc.pid)} == set(ids)
    finally:
        if proc.poll() is None:
            proc.kill()
        from agentbox.sandbox.prune import prune_agentbox_containers

        removed = prune_agentbox_containers(scope="orphans")
        assert set(ids) <= set(removed)
