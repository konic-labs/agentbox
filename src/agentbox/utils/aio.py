"""Run AgentBox coroutines from synchronous code (trainers, notebooks, threads)."""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class BackgroundLoop:
    """A dedicated event loop on a daemon thread.

    ``run(coro)`` blocks the caller until the coroutine finishes on the
    background loop, so it works from plain threads *and* from inside another
    running event loop (where ``asyncio.run`` would fail).
    """

    def __init__(self, name: str = "agentbox-loop") -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, name=name, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        if threading.current_thread() is self._thread:
            raise RuntimeError("BackgroundLoop.run() called from its own loop thread")
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def submit(self, coro: Coroutine[Any, Any, T]) -> asyncio.Future[T]:
        return asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self.loop))

    def close(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)


_shared: BackgroundLoop | None = None
_shared_lock = threading.Lock()


def shared_loop() -> BackgroundLoop:
    """Process-wide background loop used by sync facades."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = BackgroundLoop()
            atexit.register(_shared.close)
        return _shared


def run_sync(coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
    return shared_loop().run(coro, timeout)
