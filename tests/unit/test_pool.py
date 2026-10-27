"""Warm pool bookkeeping with a fake backend (no Docker)."""

from __future__ import annotations

import dataclasses
import itertools
import threading
import time
from concurrent.futures import Future
from typing import Any

from agentbox.sandbox.backend import ContainerSpec
from agentbox.sandbox.pool import POOL_LABEL, WarmPool, shape_key


class _Api:
    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend
        self.renamed: dict[str, str] = {}

    def inspect_container(self, cid: str) -> dict[str, Any]:
        return {"State": {"Running": cid not in self.backend.dead}}

    def rename(self, cid: str, name: str) -> None:
        self.renamed[cid] = name


class FakeBackend:
    """Runs background work inline when ``flush()`` is called."""

    def __init__(self, fail_creates: int = 0) -> None:
        self.ids = (f"c{i}" for i in itertools.count())
        self.created: list[ContainerSpec] = []
        self.removed: list[str] = []
        self.dead: set[str] = set()
        self.queue: list[tuple[Any, tuple[Any, ...]]] = []
        self.fail_creates = fail_creates
        self.client = type("C", (), {})()
        self.client.api = _Api(self)

    def create_container_sync(self, spec: ContainerSpec) -> str:
        if self.fail_creates:
            self.fail_creates -= 1
            raise RuntimeError("daemon hiccup")
        self.created.append(spec)
        return next(self.ids)

    def remove_container_sync(self, cid: str) -> None:
        self.removed.append(cid)

    def forget_container(self, cid: str) -> None:
        pass

    def submit_ctl(self, fn: Any, *args: Any) -> Future[Any]:
        self.queue.append((fn, args))
        return Future()

    def flush(self, *, once: bool = False) -> None:
        """Run queued work; ``once`` skips work queued while flushing."""
        while self.queue:
            batch, self.queue = self.queue, []
            for fn, args in batch:
                fn(*args)
            if once:
                return


def _spec(**kw: Any) -> ContainerSpec:
    base = ContainerSpec(
        image="img:1",
        name="agentbox-task-abc",
        command=["sleep", "infinity"],
        working_dir="/workspace",
        labels={"agentbox": "1", "agentbox.task_id": "t1", "agentbox.run_id": "r1", "team": "x"},
    )
    return dataclasses.replace(base, **kw)


def _pool(backend: FakeBackend, **kw: Any) -> WarmPool:
    return WarmPool(backend, **kw)  # type: ignore[arg-type]


def test_shape_key_ignores_rollout_identity_only() -> None:
    a = _spec()
    b = _spec(
        name="other",
        labels={**a.labels, "agentbox.task_id": "t2", "agentbox.group_id": "g9"},
    )
    assert shape_key(a) == shape_key(b)
    assert shape_key(a) != shape_key(_spec(env={"A": "1"}))
    assert shape_key(a) != shape_key(_spec(memory_mb=512))
    assert shape_key(a) != shape_key(_spec(labels={**a.labels, "team": "y"}))
    assert shape_key(a) != shape_key(a, extra="/workspace/other")


def test_miss_warms_then_hit_claims_and_renames() -> None:
    be = FakeBackend()
    pool = _pool(be)
    assert pool.acquire_sync(_spec(), size=2) is None
    be.flush()
    assert pool.ready_count() == 2
    pooled = be.created[0]
    assert pooled.labels[POOL_LABEL] == "1"
    assert "agentbox.task_id" not in pooled.labels and pooled.name.startswith("agentbox-pool-")

    cid = pool.acquire_sync(_spec(name="agentbox-mine-1"), size=2)
    assert cid == "c0" and be.client.api.renamed[cid] == "agentbox-mine-1"
    assert pool.stats.hits == 1 and pool.stats.misses == 1
    be.flush()  # topped back up
    assert pool.ready_count() == 2


def test_dead_container_is_not_handed_out() -> None:
    be = FakeBackend()
    pool = _pool(be)
    pool.prefill(_spec(), 1)
    be.flush()
    be.dead.add("c0")
    assert pool.acquire_sync(_spec(), size=1) is None
    assert "c0" in be.removed


def test_pending_creations_are_bounded() -> None:
    be = FakeBackend()
    pool = _pool(be, max_pending=3)
    pool.prefill(_spec(), 10)
    assert len(be.queue) == 3
    be.flush()
    assert pool.ready_count() == 3


def test_lru_shapes_are_evicted_and_removed() -> None:
    be = FakeBackend()
    pool = _pool(be, max_shapes=2)
    for i in range(3):
        pool.prefill(_spec(image=f"img:{i}"), 1)
        be.flush()
    # img:0 was least recently used when img:2 arrived
    assert be.removed == ["c0"]
    assert pool.ready_count() == 2


def test_failed_fill_is_counted_and_retried_on_demand() -> None:
    be = FakeBackend(fail_creates=1)
    pool = _pool(be)
    pool.prefill(_spec(), 1)
    be.flush()
    assert pool.stats.failed == 1 and pool.ready_count() == 0
    assert pool.acquire_sync(_spec(), size=1) is None
    be.flush()
    assert pool.ready_count() == 1


def test_close_removes_idle_and_drops_late_arrivals() -> None:
    be = FakeBackend()
    pool = _pool(be)
    pool.prefill(_spec(), 2)
    be.flush()
    pool.prefill(_spec(image="img:2"), 1)  # still queued when the pool closes
    assert pool.close_sync() == 2
    be.flush()
    assert sorted(be.removed) == ["c0", "c1", "c2"]
    assert pool.acquire_sync(_spec(), size=2) is None
    assert not be.queue


def test_burst_miss_waits_for_in_flight_creation() -> None:
    be = FakeBackend()
    pool = _pool(be)
    pool.prefill(_spec(), 1)  # one creation in flight
    got: list[str | None] = []
    waiter = threading.Thread(target=lambda: got.append(pool.acquire_sync(_spec(), size=1)))
    waiter.start()
    deadline = time.monotonic() + 5
    while not pool._shapes or next(iter(pool._shapes.values())).waiters == 0:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    # a second caller cannot wait on the same creation: it misses right away
    assert pool.acquire_sync(_spec(), size=1) is None
    be.flush(once=True)
    waiter.join(5)
    assert got == ["c0"]
    assert pool.stats.waited == 1 and pool.stats.hits == 1 and pool.stats.misses == 1
    assert len(be.created) == 1  # the burst did not start a second create


def test_wait_is_bounded() -> None:
    be = FakeBackend()
    pool = _pool(be, wait_s=0.1)
    pool.prefill(_spec(), 1)  # never completes
    t0 = time.monotonic()
    assert pool.acquire_sync(_spec(), size=1) is None
    assert 0.1 <= time.monotonic() - t0 < 2
