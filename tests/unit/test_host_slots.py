"""Host-wide sandbox slots: cap, crash safety, FIFO waiting, provisioning (GH #98)."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentbox.config import SandboxConfig
from agentbox.runner import provision as provision_mod
from agentbox.runner.provision import ProvisionError, provision, release
from agentbox.sandbox.slots import HostSlots, host_slots
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import FailureKind


@pytest.fixture(autouse=True)
def _slots_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENTBOX_SLOTS_DIR", str(tmp_path / "slots"))
    return tmp_path / "slots"


def test_cap_and_release(tmp_path: Path) -> None:
    slots = HostSlots(2, directory=tmp_path / "s")
    a, b = slots.try_acquire(), slots.try_acquire()
    assert a is not None and b is not None and {a.index, b.index} == {0, 1}
    assert slots.try_acquire() is None
    assert slots.in_use() == 2
    a.release()
    a.release()  # idempotent
    assert slots.in_use() == 1
    c = slots.try_acquire()
    assert c is not None and c.index == a.index
    b.release()
    c.release()
    assert slots.in_use() == 0


def test_a_killed_process_frees_its_slot(tmp_path: Path) -> None:
    """flock dies with its process: SIGKILLed trainer workers never keep slots."""
    directory = tmp_path / "s"
    code = textwrap.dedent(
        f"""
        import sys, time
        from pathlib import Path
        from agentbox.sandbox.slots import HostSlots
        lease = HostSlots(1, directory=Path({str(directory)!r})).try_acquire()
        print("held" if lease else "busy", flush=True)
        time.sleep(60)
        """
    )
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "held"
        slots = HostSlots(1, directory=directory)
        assert slots.try_acquire() is None and slots.in_use() == 1
        os.kill(child.pid, signal.SIGKILL)
        child.wait(10)
        lease = slots.try_acquire()
        assert lease is not None
        lease.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(10)


async def test_waiters_are_served_in_order_and_time_out(tmp_path: Path) -> None:
    slots = HostSlots(1, directory=tmp_path / "s")
    held = await slots.acquire()
    order: list[str] = []

    async def wait(name: str, timeout: float | None) -> None:
        try:
            lease = await slots.acquire(timeout)
        except TimeoutError:
            order.append(f"{name}:timeout")
            return
        order.append(name)
        await asyncio.sleep(0.02)
        lease.release()

    tasks = [
        asyncio.create_task(wait("first", None)),
        asyncio.create_task(wait("impatient", 0.05)),
        asyncio.create_task(wait("second", None)),
    ]
    await asyncio.sleep(0.2)  # the impatient waiter gives up meanwhile
    held.release()
    await asyncio.wait_for(asyncio.gather(*tasks), 5)
    assert order == ["impatient:timeout", "first", "second"]
    assert slots.in_use() == 0
    assert not slots._queues  # the poller dropped its queue when it drained


async def test_cancelled_waiter_leaks_nothing(tmp_path: Path) -> None:
    slots = HostSlots(1, directory=tmp_path / "s")
    held = await slots.acquire()
    waiter = asyncio.create_task(slots.acquire())
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    held.release()
    lease = await asyncio.wait_for(slots.acquire(), 2)
    assert slots.in_use() == 1
    lease.release()


def test_host_slots_is_shared_per_process(_slots_dir: Path) -> None:
    assert host_slots(3) is host_slots(3)
    assert host_slots(3).directory.parent == _slots_dir


# ---------------------------------------------------------------- provisioning
class _Manager:
    def __init__(self, cfg: SandboxConfig, *, fail: bool = False) -> None:
        self.config = cfg
        self.fail = fail
        self.destroyed: list[str] = []
        self.backend = SimpleNamespace()

    async def create(self, **kw: Any) -> Any:
        if self.fail:
            raise RuntimeError("daemon said no")
        return SimpleNamespace(id="c1", name="agentbox-t-1", task_id=kw["task_id"])

    async def destroy(self, sandbox: Any) -> None:
        self.destroyed.append(sandbox.id)


def _task() -> Task:
    return Task(task_id="t", description="d", verifier=VerifierSpec(type="command", command="true"))


class _Seeder:
    def __init__(self, manager: Any) -> None:
        pass

    async def seed(self, sandbox: Any, task: Any) -> Any:
        return SimpleNamespace(ok=True, duration_s=0.0, error=None)


async def test_provision_holds_a_slot_until_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provision_mod, "TaskSeeder", _Seeder)
    cfg = SandboxConfig(host_max_sandboxes=1, host_slot_timeout_s=0.2)
    task = _task()
    mgr = _Manager(cfg)
    prov = await provision(task, manager=mgr, run_id="r", rollout_id="a")
    assert prov.lease is not None and prov.wait_s is not None
    with pytest.raises(ProvisionError) as info:  # the only slot is taken
        await provision(task, manager=mgr, run_id="r", rollout_id="b")
    assert info.value.kind is FailureKind.SANDBOX and "host_max_sandboxes=1" in str(info.value)
    await release(prov, keep=False)
    assert mgr.destroyed == ["c1"] and not prov.lease.held
    again = await provision(task, manager=mgr, run_id="r", rollout_id="c")
    await release(again, keep=False)


async def test_failed_create_frees_the_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provision_mod, "TaskSeeder", _Seeder)
    cfg = SandboxConfig(host_max_sandboxes=1, host_slot_timeout_s=0.2)
    task = _task()
    with pytest.raises(ProvisionError):
        await provision(task, manager=_Manager(cfg, fail=True), run_id="r", rollout_id="a")
    assert host_slots(1).in_use() == 0


async def test_no_cap_means_no_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provision_mod, "TaskSeeder", _Seeder)
    mgr = _Manager(SandboxConfig())
    prov = await provision(_task(), manager=mgr, run_id="r", rollout_id="a")
    assert prov.lease is None and prov.wait_s is None
    await release(prov, keep=False)
