"""Host-wide cap on live sandboxes, shared by every AgentBox process on the host (GH #98).

Distributed trainers run one AgentBox process per worker: verl agent-loop
workers, OpenRLHF vLLM engines, TRL ranks. A per-process limit multiplies with
the number of workers, so this cap lives on the host instead. It is ``limit``
lock files in a directory, and each sandbox holds an exclusive ``flock`` on one
of them for as long as it lives. The kernel drops a lock when its process dies,
so a crashed or SIGKILLed worker never keeps a slot.

Waiting is FIFO per event loop: one poller task per loop scans the slot files
with a short backoff and hands each free slot to the oldest waiter, so a
thousand waiting rollouts cost one scan per loop, not a thousand.
"""

from __future__ import annotations

import asyncio
import errno
import os
import random
import re
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

SLOTS_DIR_ENV = "AGENTBOX_SLOTS_DIR"
_POLL_MIN_S = 0.005
_POLL_MAX_S = 0.2


def default_slots_dir() -> Path:
    """``$AGENTBOX_SLOTS_DIR``, else a per-user directory in /dev/shm (or the temp dir)."""
    override = os.environ.get(SLOTS_DIR_ENV)
    if override:
        return Path(override)
    shm = Path("/dev/shm")
    base = shm if shm.is_dir() and os.access(shm, os.W_OK) else Path(tempfile.gettempdir())
    return base / f"agentbox-slots-{os.getuid()}"


def daemon_key() -> str:
    """One slot namespace per Docker daemon (``$DOCKER_HOST``, else the local one)."""
    host = os.environ.get("DOCKER_HOST") or "local"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", host)[:64]


class SlotLease:
    """A held slot. :meth:`release` frees it and may be called more than once."""

    __slots__ = ("_fd", "index")

    def __init__(self, fd: int, index: int) -> None:
        self._fd: int | None = fd
        self.index = index

    @property
    def held(self) -> bool:
        return self._fd is not None

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)  # closing the descriptor drops the flock


@dataclass
class _Waiter:
    future: asyncio.Future[SlotLease]
    deadline: float | None


@dataclass
class _LoopQueue:
    loop: asyncio.AbstractEventLoop  # strong ref: the id(loop) key stays unique while queued
    waiters: deque[_Waiter] = field(default_factory=deque)
    poller: asyncio.Task[None] | None = None


class HostSlots:
    """``limit`` slots in ``directory``, shared with every process that uses it."""

    def __init__(self, limit: int, *, directory: Path | None = None) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if fcntl is None:  # pragma: no cover - Windows
            raise RuntimeError("host_max_sandboxes needs flock (Linux or macOS)")
        self.limit = limit
        self.directory = directory or default_slots_dir() / daemon_key()
        self.directory.mkdir(parents=True, exist_ok=True)
        # keyed by id(loop): some loops (uvloop) cannot be weakly referenced; an entry
        # lives only while its loop has waiters
        self._queues: dict[int, _LoopQueue] = {}
        self._guard = threading.Lock()

    def _path(self, index: int) -> Path:
        return self.directory / f"slot-{index}.lock"

    # ------------------------------------------------------------------ sync
    def try_acquire(self) -> SlotLease | None:
        """Take a free slot without waiting, or return ``None``."""
        start = random.randrange(self.limit)
        for k in range(self.limit):
            index = (start + k) % self.limit
            fd = os.open(self._path(index), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                    continue
                raise
            return SlotLease(fd, index)
        return None

    def in_use(self) -> int:
        """Slots held by any process right now (a racy snapshot, for diagnostics)."""
        busy = 0
        for index in range(self.limit):
            try:
                fd = os.open(self._path(index), os.O_RDWR)
            except FileNotFoundError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                busy += 1
            finally:
                os.close(fd)
        return busy

    # ----------------------------------------------------------------- async
    def _queue(self, loop: asyncio.AbstractEventLoop) -> _LoopQueue:
        with self._guard:
            q = self._queues.get(id(loop))
            if q is None or q.loop is not loop:
                q = self._queues[id(loop)] = _LoopQueue(loop)
            return q

    def _drop_queue(self, q: _LoopQueue) -> None:
        with self._guard:
            if self._queues.get(id(q.loop)) is q:
                del self._queues[id(q.loop)]

    async def acquire(self, timeout_s: float | None = None) -> SlotLease:
        """Wait (FIFO within this event loop) for a free slot.

        Raises :class:`TimeoutError` when none frees up within ``timeout_s``.
        """
        loop = asyncio.get_running_loop()
        with self._guard:
            ahead = self._queues.get(id(loop))
            queued = ahead is not None and ahead.loop is loop and bool(ahead.waiters)
        if not queued:  # nobody waits ahead of us in this loop: try right away
            lease = self.try_acquire()
            if lease is not None:
                return lease
        q = self._queue(loop)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        waiter = _Waiter(loop.create_future(), deadline)
        q.waiters.append(waiter)
        if q.poller is None or q.poller.done():
            q.poller = loop.create_task(self._poll(q), name="agentbox-host-slots")
        try:
            return await waiter.future
        except asyncio.CancelledError:
            fut = waiter.future
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                fut.result().release()  # handed over just as we were cancelled
            raise

    async def _poll(self, q: _LoopQueue) -> None:
        """Hand free slots to the oldest waiters; fail the ones past their deadline.

        Only this task completes waiter futures, in synchronous blocks, so a
        lease is never lost between being taken and being handed over.
        """
        delay = _POLL_MIN_S
        while True:
            now = time.monotonic()
            while q.waiters:
                head = q.waiters[0]
                if head.future.done():  # cancelled by its caller
                    q.waiters.popleft()
                elif head.deadline is not None and now >= head.deadline:
                    q.waiters.popleft()
                    head.future.set_exception(
                        TimeoutError(f"all {self.limit} host sandbox slots stayed busy")
                    )
                else:
                    break
            for w in list(q.waiters)[1:]:  # deadlines behind the head expire too
                if not w.future.done() and w.deadline is not None and now >= w.deadline:
                    w.future.set_exception(
                        TimeoutError(f"all {self.limit} host sandbox slots stayed busy")
                    )
            if not q.waiters:
                q.poller = None
                self._drop_queue(q)
                return
            lease = self.try_acquire()
            if lease is not None:
                q.waiters.popleft().future.set_result(lease)
                delay = _POLL_MIN_S
                continue
            await asyncio.sleep(delay * (0.5 + random.random()))
            delay = min(delay * 2, _POLL_MAX_S)


_SLOTS: dict[tuple[int, str], HostSlots] = {}
_SLOTS_LOCK = threading.Lock()


def host_slots(limit: int) -> HostSlots:
    """The process-wide :class:`HostSlots` for ``limit`` slots on this host."""
    directory = default_slots_dir() / daemon_key()
    key = (limit, str(directory))
    with _SLOTS_LOCK:
        slots = _SLOTS.get(key)
        if slots is None:
            slots = _SLOTS[key] = HostSlots(limit, directory=directory)
        return slots
