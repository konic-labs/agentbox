"""Warm pool of pre-started sandbox containers (GH #23).

Creating and starting a container sits on every rollout's critical path. The
pool moves that work off it: containers are created ahead of demand for each
container *shape* (image, limits, security, env, ...) and handed out **at most
once**, so a rollout never sees a container another rollout touched and the
clean-state guarantee is unchanged.

Refills are demand-driven: a shape is only warmed after a rollout asked for it
(or :meth:`WarmPool.prefill` was called), idle containers expire after
``ttl_s``, and the least recently used shapes are dropped beyond
``max_shapes``. That bounds the waste when every task has its own
setup-snapshot image. A rollout that misses while a container of its shape is
already being created waits for that one instead of starting another create, so
a burst of misses does not double the load on the Docker daemon.

Pooled containers are created before their rollout exists, so they carry no
``agentbox.task_id`` / ``run_id`` / ``rollout_id`` labels (Docker labels are
immutable); they are renamed to the usual ``agentbox-<task>-<hex>`` name when
claimed and keep the owner label, so orphan pruning still applies (pruning by
``--run-id`` does not see them).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
import time
import uuid
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable

from docker.errors import APIError, DockerException, NotFound

from agentbox.sandbox.backend import ContainerSpec, DockerBackend

logger = logging.getLogger("agentbox.sandbox")

# labels that identify one rollout (a pooled container is created before it exists)
ROLLOUT_LABELS = frozenset(
    {
        "agentbox.task_id",
        "agentbox.run_id",
        "agentbox.rollout_id",
        "agentbox.group_id",
        "agentbox.created",
    }
)
POOL_LABEL = "agentbox.pool"

Prepare = Callable[[str], None]


def shape_key(spec: ContainerSpec, extra: str = "") -> str:
    """Hash of everything that makes two containers interchangeable."""
    data = dataclasses.asdict(spec)
    data.pop("name")
    data["labels"] = {k: v for k, v in sorted(spec.labels.items()) if k not in ROLLOUT_LABELS}
    data["extra"] = extra
    blob = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclasses.dataclass
class PoolStats:
    hits: int = 0  # includes ``waited``
    misses: int = 0
    waited: int = 0  # hits that waited for an in-flight creation
    created: int = 0
    expired: int = 0
    failed: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class _Shape:
    __slots__ = ("pending", "prepare", "ready", "spec", "waiters")

    def __init__(self, spec: ContainerSpec, prepare: Prepare | None) -> None:
        self.spec = spec
        self.prepare = prepare
        self.ready: deque[tuple[str, float]] = deque()  # (container id, ready at)
        self.pending = 0
        self.waiters = 0


class WarmPool:
    """Per-shape queues of started, never-used containers."""

    def __init__(
        self,
        backend: DockerBackend,
        *,
        ttl_s: float = 300.0,
        max_shapes: int = 8,
        max_pending: int = 4,
        wait_s: float = 10.0,
    ) -> None:
        self.backend = backend
        self.ttl_s = ttl_s
        self.max_shapes = max_shapes
        self.max_pending = max_pending  # background creations in flight, all shapes
        self.wait_s = wait_s  # upper bound on waiting for an in-flight creation
        self.stats = PoolStats()
        self.closed = False
        self._shapes: OrderedDict[str, _Shape] = OrderedDict()
        self._pending_total = 0
        self._lock = threading.Lock()
        self._arrived = threading.Condition(self._lock)

    # ------------------------------------------------------------------ api
    def acquire_sync(
        self,
        spec: ContainerSpec,
        *,
        size: int,
        prepare: Prepare | None = None,
        extra: str = "",
    ) -> str | None:
        """Claim a ready container for ``spec``'s shape, renamed to ``spec.name``.

        Returns ``None`` on a miss (the caller creates its own container). When
        nothing is ready but more creations of this shape are in flight than
        callers already waiting for them, waits (at most ``wait_s``) for one.
        Either way the shape is topped up to ``size`` ready-or-pending
        containers in the background. Blocks: call it from a worker thread.
        """
        key = shape_key(spec, extra)
        stale: list[str] = []
        cid: str | None = None
        waited = False
        with self._lock:
            if self.closed:
                return None
            shape = self._touch_locked(key, spec, prepare, stale)
            if shape.ready:
                cid = shape.ready.popleft()[0]
            elif shape.pending > shape.waiters:
                waited = True
                shape.waiters += 1
                deadline = time.monotonic() + self.wait_s
                try:
                    while not shape.ready and shape.pending and not self.closed:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._arrived.wait(remaining)
                    if shape.ready:
                        cid = shape.ready.popleft()[0]
                finally:
                    shape.waiters -= 1
        self._discard(stale)
        self._refill(key, size)
        hit = cid is not None and self._claim(cid, spec.name)
        with self._lock:
            if hit:
                self.stats.hits += 1
                self.stats.waited += waited
            else:
                self.stats.misses += 1
        return cid if hit else None

    def prefill(
        self,
        spec: ContainerSpec,
        n: int,
        *,
        prepare: Prepare | None = None,
        extra: str = "",
    ) -> None:
        """Start warming ``n`` containers of ``spec``'s shape (non-blocking)."""
        key = shape_key(spec, extra)
        stale: list[str] = []
        with self._lock:
            if self.closed:
                return
            self._touch_locked(key, spec, prepare, stale)
        self._discard(stale)
        self._refill(key, n)

    def ready_count(self) -> int:
        with self._lock:
            return sum(len(s.ready) for s in self._shapes.values())

    def close_sync(self) -> int:
        """Remove every idle pooled container; in-flight creations are dropped on arrival."""
        with self._lock:
            self.closed = True
            ids = [cid for s in self._shapes.values() for cid, _ in s.ready]
            for s in self._shapes.values():
                s.ready.clear()
            self._arrived.notify_all()
        self._discard(ids, sync=True)
        return len(ids)

    async def aclose(self) -> int:
        return await self.backend.run_ctl(self.close_sync)

    # ------------------------------------------------------------- internals
    def _touch_locked(
        self, key: str, spec: ContainerSpec, prepare: Prepare | None, stale: list[str]
    ) -> _Shape:
        """Mark ``key`` most recently used; expire idle containers; evict LRU shapes."""
        shape = self._shapes.get(key)
        if shape is None:
            shape = self._shapes[key] = _Shape(_template(spec), prepare)
        elif prepare is not None:
            shape.prepare = prepare
        self._shapes.move_to_end(key)
        now = time.monotonic()
        for s in self._shapes.values():
            while s.ready and now - s.ready[0][1] > self.ttl_s:
                stale.append(s.ready.popleft()[0])
                self.stats.expired += 1
        while len(self._shapes) > self.max_shapes:
            _, old = self._shapes.popitem(last=False)
            stale.extend(cid for cid, _ in old.ready)
            old.ready.clear()
        return shape

    def _refill(self, key: str, size: int) -> None:
        with self._lock:
            shape = self._shapes.get(key)
            if shape is None or self.closed:
                return
            want = size - len(shape.ready) - shape.pending
            n = max(0, min(want, self.max_pending - self._pending_total))
            shape.pending += n
            self._pending_total += n
        for _ in range(n):
            try:
                self.backend.submit_ctl(self._fill_one, shape)
            except RuntimeError:  # backend shut down
                with self._lock:
                    shape.pending -= 1
                    self._pending_total -= 1

    def _fill_one(self, shape: _Shape) -> None:
        cid: str | None = None
        try:
            spec = dataclasses.replace(
                shape.spec,
                name=f"agentbox-pool-{uuid.uuid4().hex[:12]}",
                labels={**shape.spec.labels, "agentbox.created": str(int(time.time()))},
            )
            cid = self.backend.create_container_sync(spec)
            if shape.prepare is not None:
                shape.prepare(cid)
        except Exception as exc:  # the rollout path reports real failures
            logger.warning("pool.fill failed: %s", exc)
            if cid is not None:
                self.backend.remove_container_sync(cid)
                cid = None
            with self._lock:
                self.stats.failed += 1
        finally:
            with self._lock:
                shape.pending -= 1
                self._pending_total -= 1
                if cid is not None:
                    self.stats.created += 1
                live = not self.closed and any(s is shape for s in self._shapes.values())
                if cid is not None and live:
                    shape.ready.append((cid, time.monotonic()))
                    cid = None
                self._arrived.notify_all()  # waiters re-check (arrival or failure)
            if cid is not None:  # shape evicted or pool closed meanwhile
                self.backend.remove_container_sync(cid)

    def _claim(self, cid: str, name: str) -> bool:
        api = self.backend.client.api
        try:
            if not api.inspect_container(cid)["State"]["Running"]:
                self.backend.remove_container_sync(cid)
                return False
        except NotFound:
            self.backend.forget_container(cid)
            return False
        except DockerException as exc:
            logger.warning("pool.claim inspect %s failed: %s", cid[:12], exc)
            self.backend.remove_container_sync(cid)
            return False
        try:
            api.rename(cid, name)
        except APIError as exc:  # cosmetic: keep the pool name
            logger.debug("pool.claim rename %s failed: %s", cid[:12], exc)
        return True

    def _discard(self, ids: list[str], *, sync: bool = False) -> None:
        for cid in ids:
            if sync:
                self.backend.remove_container_sync(cid)
                continue
            try:
                self.backend.submit_ctl(self.backend.remove_container_sync, cid)
            except RuntimeError:
                self.backend.remove_container_sync(cid)


def _template(spec: ContainerSpec) -> ContainerSpec:
    labels = {k: v for k, v in spec.labels.items() if k not in ROLLOUT_LABELS}
    labels[POOL_LABEL] = "1"
    return dataclasses.replace(spec, name="", labels=labels)


_POOLS: weakref.WeakKeyDictionary[DockerBackend, WarmPool] = weakref.WeakKeyDictionary()
_POOLS_LOCK = threading.Lock()


def pool_for(backend: DockerBackend, *, ttl_s: float | None = None) -> WarmPool:
    """The warm pool attached to ``backend`` (created on first use)."""
    with _POOLS_LOCK:
        pool = _POOLS.get(backend)
        if pool is None or pool.closed:
            pool = _POOLS[backend] = WarmPool(backend, ttl_s=ttl_s or 300.0)
        elif ttl_s is not None:
            pool.ttl_s = ttl_s
        return pool
