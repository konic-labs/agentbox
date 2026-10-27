"""Docker access layer shared by every sandbox in the process.

Design (see GH #13, #14, #15, #16, #17):

* **One client** with a connection pool sized for the configured concurrency.
* **Two bounded thread pools**: ``io`` for long-running exec streams and image
  builds, ``ctl`` for short control calls (create/remove/kill/archive). A burst of
  slow commands can never starve container teardown or kills.
* **Streaming exec** into bounded head+tail buffers (docker-py's ``demux=True``
  concatenates bytes quadratically and keeps everything in memory).
* **Real timeouts**: every exec carries a unique ``AGENTBOX_EXEC_ID`` env marker;
  on timeout (or output overflow, or cancellation) every process carrying the
  marker is SIGKILLed via a ``/proc/*/environ`` scan, which needs only ``/bin/sh``
  and ``grep`` (``ps``/``pkill`` are missing from slim images).
"""

from __future__ import annotations

import asyncio
import atexit
import functools
import hashlib
import io
import json
import logging
import os
import queue
import socket
import tarfile
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, TypeVar

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound

from agentbox.errors import DockerNotAvailableError, ImageNotFoundError, SandboxError
from agentbox.sandbox.types import ExecResult, ImageProfile

logger = logging.getLogger("agentbox.sandbox")

T = TypeVar("T")

EXEC_MARKER_ENV = "AGENTBOX_EXEC_ID"
CACHE_REPOSITORY = "agentbox-cache"
OWNER = f"{socket.gethostname()}:{os.getpid()}"

# Go os.FileMode bits reported by the archive stat header.
_GO_MODE_DIR = 1 << 31
_GO_MODE_SYMLINK = 1 << 27

_KILL_SCRIPT = (
    "for d in /proc/[0-9]*; do "
    'if grep -qs "{marker}" "$d/environ" 2>/dev/null; '
    'then kill -9 "${{d#/proc/}}" 2>/dev/null; fi; '
    "done; true"
)

_PROBE_SCRIPT = (
    "command -v bash >/dev/null 2>&1 && echo has_bash; "
    "command -v python >/dev/null 2>&1 && echo has_python; "
    "python -m pytest --version >/dev/null 2>&1 && echo has_pytest; true"
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


class _DaemonPool(Executor):
    """Bounded executor with *daemon* workers.

    ``ThreadPoolExecutor`` joins its workers at interpreter exit; a worker blocked
    on a container stream would then hang shutdown before our atexit cleanup can
    remove the container that unblocks it.
    """

    def __init__(self, max_workers: int, name: str) -> None:
        self._max = max(1, max_workers)
        self._name = name
        self._queue: queue.SimpleQueue[tuple[Future[Any], Callable[..., Any]] | None] = (
            queue.SimpleQueue()
        )
        self._idle = threading.Semaphore(0)
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        fut: Future[Any] = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError(f"{self._name} pool is shut down")
            self._queue.put((fut, functools.partial(fn, *args, **kwargs)))
            if not self._idle.acquire(timeout=0) and len(self._threads) < self._max:
                t = threading.Thread(
                    target=self._worker,
                    name=f"{self._name}-{len(self._threads)}",
                    daemon=True,
                )
                self._threads.append(t)
                t.start()
        return fut

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fut, call = item
            if fut.set_running_or_notify_cancel():
                try:
                    fut.set_result(call())
                except BaseException as exc:  # noqa: BLE001 - forwarded to the future
                    fut.set_exception(exc)
            self._idle.release()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._lock:
            self._shutdown = True
            if cancel_futures:
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is not None:
                        item[0].cancel()
            for _ in self._threads:
                self._queue.put(None)
        if wait:
            for t in self._threads:
                t.join()


class OutputCapture:
    """Thread-safe bounded capture keeping the head and tail of a byte stream."""

    __slots__ = ("_head_budget", "_tail_budget", "_head", "_tail", "_lock", "total")

    def __init__(self, limit: int) -> None:
        limit = max(0, limit)
        self._head_budget = limit // 2
        self._tail_budget = limit - self._head_budget
        self._head = bytearray()
        self._tail = bytearray()
        self._lock = threading.Lock()
        self.total = 0

    def feed(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            self.total += len(data)
            room = self._head_budget - len(self._head)
            if room > 0:
                self._head += data[:room]
                data = data[room:]
            if data and self._tail_budget > 0:
                self._tail += data
                # Amortised trim: let the buffer grow to 2x before compacting.
                if len(self._tail) > 2 * self._tail_budget:
                    del self._tail[: len(self._tail) - self._tail_budget]

    def value(self) -> tuple[bytes, bool]:
        """Return ``(captured_bytes, truncated)``."""
        with self._lock:
            tail = bytes(self._tail[-self._tail_budget :]) if self._tail_budget else b""
            kept = len(self._head) + len(tail)
            if self.total <= kept:
                return bytes(self._head) + tail, False
            dropped = self.total - kept
            marker = f"\n...[truncated {dropped} bytes]...\n".encode()
            return bytes(self._head) + marker + tail, True

    def text(self) -> tuple[str, bool]:
        data, truncated = self.value()
        return data.decode("utf-8", errors="replace"), truncated


@dataclass(slots=True)
class ContainerSpec:
    """Everything needed to create one sandbox container."""

    image: str
    name: str
    command: list[str]
    working_dir: str
    labels: dict[str, str]
    env: dict[str, str] = field(default_factory=dict)
    network_disabled: bool = True
    cpu_count: float | None = None
    memory_mb: int | None = None
    memswap_mb: int | None = None
    pids_limit: int | None = None
    shm_size_mb: int | None = None
    nofile: int | None = None
    read_only_rootfs: bool = False
    tmpfs: dict[str, str] = field(default_factory=dict)
    cap_drop: list[str] = field(default_factory=list)
    cap_add: list[str] = field(default_factory=list)
    no_new_privileges: bool = True
    init: bool = True
    user: str | None = None
    runtime: str | None = None


@dataclass(slots=True)
class ReadResult:
    data: bytes
    size: int
    truncated: bool
    link_target: str | None = None


class _ChunkReader(io.RawIOBase):
    """File-like adapter over an iterator of byte chunks (for streaming tarfile)."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self._buf = b""

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        while not self._buf:
            try:
                self._buf = next(self._chunks)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


def build_tar(
    files: Mapping[str, bytes],
    *,
    modes: Mapping[str, int] | None = None,
    uid: int = 0,
    gid: int = 0,
) -> bytes:
    """Tar ``files`` (relative POSIX paths) including parent directory entries.

    Directory entries let ``put_archive`` create missing parents, so no separate
    ``mkdir -p`` exec is needed (GH #16).
    """
    now = time.time()
    buf = io.BytesIO()
    seen_dirs: set[str] = set()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel, data in files.items():
            rel = rel.lstrip("/")
            parts = PurePosixPath(rel).parts
            for i in range(1, len(parts)):
                d = "/".join(parts[:i])
                if d in seen_dirs:
                    continue
                seen_dirs.add(d)
                dinfo = tarfile.TarInfo(d)
                dinfo.type = tarfile.DIRTYPE
                dinfo.mode = 0o755
                dinfo.mtime = now
                dinfo.uid, dinfo.gid = uid, gid
                tar.addfile(dinfo)
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mode = (modes or {}).get(rel, 0o644)
            info.mtime = now
            info.uid, info.gid = uid, gid
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def container_name(task_id: str, prefix: str = "agentbox") -> str:
    """DNS-safe, collision-resistant container name (48 random bits, GH #26)."""
    slug = "".join(c if c.isalnum() or c in "-_." else "-" for c in task_id.lower())
    slug = slug.strip("-.")[:32] or "task"
    return f"{prefix}-{slug}-{uuid.uuid4().hex[:12]}"[:63]


class DockerBackend:
    """Thread-safe Docker operations with bounded pools and live-container tracking."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        io_threads: int | None = None,
        ctl_threads: int | None = None,
        timeout_s: float = 300.0,
        kill_grace_s: float = 5.0,
    ) -> None:
        self.io_threads = io_threads or _env_int("AGENTBOX_DOCKER_IO_THREADS", 128)
        self.ctl_threads = ctl_threads or _env_int("AGENTBOX_DOCKER_CTL_THREADS", 16)
        self.kill_grace_s = kill_grace_s
        self._external_client = client is not None
        if client is None:
            try:
                client = docker.from_env(
                    timeout=int(timeout_s),
                    max_pool_size=self.io_threads + self.ctl_threads,
                )
                client.ping()
            except DockerException as exc:
                raise DockerNotAvailableError(
                    "Docker daemon is not available. Is Docker running?"
                ) from exc
        self.client = client
        self._io = _DaemonPool(self.io_threads, "agentbox-io")
        self._ctl = _DaemonPool(self.ctl_threads, "agentbox-ctl")
        self._images: set[str] = set()
        self._profiles: dict[str, ImageProfile] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._live: set[str] = set()
        self._live_guard = threading.Lock()
        self._cache_used: dict[str, float] = {}
        self.closed = False
        _BACKENDS.add(self)

    # ------------------------------------------------------------------ pools
    async def run_ctl(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._ctl, functools.partial(fn, *args, **kwargs))

    def submit_ctl(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        """Run ``fn`` on the control pool without awaiting it (background work)."""
        return self._ctl.submit(fn, *args, **kwargs)

    async def run_ctl_owned(
        self, fn: Callable[..., str | None], *args: Any, **kwargs: Any
    ) -> str | None:
        """Run a control call that yields a new container id, owning the result.

        The call cannot be interrupted once it runs in its thread; if the awaiting
        task is cancelled meanwhile, the container it produces is removed on
        arrival instead of running until process exit.
        """
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(self._ctl, functools.partial(fn, *args, **kwargs))
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:

            def reap(done: asyncio.Future[str | None]) -> None:
                if done.cancelled() or done.exception() is not None:
                    return
                cid = done.result()
                if cid:
                    try:
                        self.submit_ctl(self.remove_container_sync, cid)
                    except RuntimeError:  # shut down: exit cleanup removes it
                        pass

            fut.add_done_callback(reap)
            raise

    async def run_io(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._io, functools.partial(fn, *args, **kwargs))

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = threading.Lock()
            return lock

    # ----------------------------------------------------------------- images
    def ensure_image_sync(self, image: str, *, auto_pull: bool = True) -> None:
        if image in self._images:
            return
        with self._lock_for(f"image:{image}"):  # single-flight pulls
            if image in self._images:
                return
            api = self.client.api
            try:
                api.inspect_image(image)
            except ImageNotFound:
                if not auto_pull:
                    raise ImageNotFoundError(f"Image not found: {image}") from None
                logger.info("pulling image %s", image)
                try:
                    self.client.images.pull(image)
                except DockerException as exc:
                    raise ImageNotFoundError(f"Failed to pull image {image}: {exc}") from exc
            self._images.add(image)

    async def ensure_image(self, image: str, *, auto_pull: bool = True) -> None:
        if image in self._images:
            return
        await self.run_ctl(self.ensure_image_sync, image, auto_pull=auto_pull)

    def image_exists_sync(self, image: str) -> bool:
        if image in self._images:
            return True
        try:
            self.client.api.inspect_image(image)
        except ImageNotFound:
            return False
        self._images.add(image)
        return True

    def image_profile_sync(self, image: str) -> ImageProfile:
        cached = self._profiles.get(image)
        if cached is not None:
            return cached
        with self._lock_for(f"profile:{image}"):
            cached = self._profiles.get(image)
            if cached is not None:
                return cached
            image_id = str(self.client.api.inspect_image(image).get("Id", image))
            spec = ContainerSpec(
                image=image,
                name=container_name("probe", prefix="agentbox-probe"),
                command=["sleep", "infinity"],
                working_dir="/",
                labels=self._labels({"agentbox.role": "probe"}),
                network_disabled=True,
                memory_mb=256,
            )
            out: list[str] = []
            try:
                cid = self.create_container_sync(spec)
                try:
                    res = self.exec_sync(cid, ["/bin/sh", "-c", _PROBE_SCRIPT], timeout_s=60)
                    out = res.stdout.split()
                finally:
                    self.remove_container_sync(cid)
            except SandboxError as exc:  # e.g. distroless image without sleep/sh
                logger.warning("could not probe image %s: %s", image, exc)
            profile = ImageProfile(
                image=image,
                image_id=image_id,
                has_bash="has_bash" in out,
                has_python="has_python" in out,
                has_pytest="has_pytest" in out,
            )
            self._profiles[image] = profile
            return profile

    async def image_profile(self, image: str) -> ImageProfile:
        cached = self._profiles.get(image)
        if cached is not None:
            return cached
        return await self.run_io(self.image_profile_sync, image)

    def derive_image_sync(
        self,
        base: str,
        *,
        steps: list[str],
        files: Mapping[str, bytes] | None = None,
        workdir: str = "/workspace",
        env: Mapping[str, str] | None = None,
        network: bool = True,
        shell: list[str] | None = None,
        step_timeout_s: float = 900.0,
        key_extra: Mapping[str, Any] | None = None,
        max_images: int = 64,
    ) -> str:
        """Build (once) and return a cached image = ``base`` + files + ``steps``.

        Keyed by the base image id and every input, so changing anything yields a
        new image. Concurrent callers for the same key wait for one build.
        """
        base_id = str(self.client.api.inspect_image(base).get("Id", base))
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "base": base_id,
                    "steps": steps,
                    "workdir": workdir,
                    "env": dict(env or {}),
                    "network": network,
                    "extra": dict(key_extra or {}),
                },
                sort_keys=True,
            ).encode()
        )
        for path in sorted(files or {}):
            digest.update(path.encode())
            digest.update(hashlib.sha256((files or {})[path]).digest())
        key = digest.hexdigest()[:24]
        tag = f"{CACHE_REPOSITORY}:{key}"
        with self._lock_for(f"derive:{tag}"):
            if self.image_exists_sync(tag):
                self._cache_used[tag] = time.time()
                return tag
            spec = ContainerSpec(
                image=base,
                name=container_name(key, prefix="agentbox-build"),
                command=["sleep", "infinity"],
                working_dir=workdir,
                labels=self._labels({"agentbox.role": "builder", "agentbox.cache_key": key}),
                env=dict(env or {}),
                network_disabled=not network,
                memory_mb=2048,
                pids_limit=1024,
            )
            t0 = time.monotonic()
            cid = self.create_container_sync(spec)
            try:
                if files:
                    self.put_files_sync(cid, workdir, dict(files))
                prefix = shell or ["/bin/sh", "-c"]
                for step in steps:
                    res = self.exec_sync(
                        cid, [*prefix, step], timeout_s=step_timeout_s, workdir=workdir
                    )
                    if not res.ok:
                        detail = (res.stderr or res.stdout).strip()[-2000:]
                        raise SandboxError(
                            f"image build step failed ({step!r}, exit={res.exit_code}"
                            f"{', timed out' if res.timed_out else ''}): {detail}"
                        )
                self.client.api.commit(
                    cid,
                    repository=CACHE_REPOSITORY,
                    tag=key,
                    conf={"Labels": {"agentbox.cache": "1", "agentbox.cache_key": key}},
                )
            finally:
                self.remove_container_sync(cid)
            self._images.add(tag)
            self._cache_used[tag] = time.time()
            logger.info("cache.built %s from %s in %.1fs", tag, base, time.monotonic() - t0)
        self._evict_cache_sync(max_images, keep=tag)
        return tag

    async def derive_image(self, base: str, **kwargs: Any) -> str:
        return await self.run_io(self.derive_image_sync, base, **kwargs)

    def list_cache_images_sync(self) -> list[dict[str, Any]]:
        images = self.client.api.images(name=CACHE_REPOSITORY)
        return list(images or [])

    def _evict_cache_sync(self, max_images: int, *, keep: str) -> None:
        if max_images <= 0:
            return
        try:
            images = self.list_cache_images_sync()
        except DockerException:
            return
        if len(images) <= max_images:
            return

        def last_used(img: dict[str, Any]) -> float:
            tags = img.get("RepoTags") or []
            return max(
                [self._cache_used.get(t, 0.0) for t in tags] + [float(img.get("Created", 0))]
            )

        for img in sorted(images, key=last_used)[: len(images) - max_images]:
            tags = img.get("RepoTags") or []
            if keep in tags:
                continue
            for t in tags:
                try:
                    self.client.api.remove_image(t)
                    self._images.discard(t)
                    logger.info("cache.evicted %s", t)
                except DockerException:
                    pass  # in use by a running sandbox; try again later

    # ------------------------------------------------------------- containers
    def _labels(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        labels = {
            "agentbox": "1",
            "agentbox.owner": OWNER,
            "agentbox.created": str(int(time.time())),
            "agentbox.role": "sandbox",
        }
        labels.update(extra or {})
        return labels

    def create_container_sync(self, spec: ContainerSpec) -> str:
        api = self.client.api
        host_kwargs: dict[str, Any] = {
            "init": spec.init,
            "network_mode": "none" if spec.network_disabled else "bridge",
            "read_only": spec.read_only_rootfs,
        }
        if spec.memory_mb is not None:
            host_kwargs["mem_limit"] = f"{spec.memory_mb}m"
            swap = spec.memswap_mb if spec.memswap_mb is not None else spec.memory_mb
            host_kwargs["memswap_limit"] = f"{swap}m"
        if spec.cpu_count is not None:
            host_kwargs["nano_cpus"] = int(spec.cpu_count * 1e9)
        if spec.pids_limit is not None:
            host_kwargs["pids_limit"] = spec.pids_limit
        if spec.shm_size_mb is not None:
            host_kwargs["shm_size"] = f"{spec.shm_size_mb}m"
        if spec.nofile is not None:
            host_kwargs["ulimits"] = [
                docker.types.Ulimit(name="nofile", soft=spec.nofile, hard=spec.nofile)
            ]
        if spec.tmpfs:
            host_kwargs["tmpfs"] = dict(spec.tmpfs)
        if spec.cap_drop:
            host_kwargs["cap_drop"] = list(spec.cap_drop)
        if spec.cap_add:
            host_kwargs["cap_add"] = list(spec.cap_add)
        if spec.no_new_privileges:
            host_kwargs["security_opt"] = ["no-new-privileges:true"]
        if spec.runtime:
            host_kwargs["runtime"] = spec.runtime
        try:
            host_config = api.create_host_config(**host_kwargs)
            resp = api.create_container(
                image=spec.image,
                command=spec.command,
                name=spec.name,
                working_dir=spec.working_dir,
                environment=spec.env or None,
                labels=spec.labels,
                host_config=host_config,
                user=spec.user or "",
                detach=True,
                hostname="sandbox",
            )
            cid = str(resp["Id"])
        except APIError as exc:
            raise SandboxError(f"Failed to create container: {exc}") from exc
        with self._live_guard:
            self._live.add(cid)
        try:
            api.start(cid)
        except APIError as exc:
            self.remove_container_sync(cid)
            raise SandboxError(f"Failed to start container: {exc}") from exc
        return cid

    async def create_container(self, spec: ContainerSpec) -> str:
        cid = await self.run_ctl_owned(self.create_container_sync, spec)
        assert cid is not None
        return cid

    def remove_container_sync(self, cid: str) -> None:
        """SIGKILL + remove (with anonymous volumes) in a single API call."""
        try:
            self.client.api.remove_container(cid, force=True, v=True)
        except NotFound:
            pass
        except APIError as exc:
            if "removal of container" in str(exc) and "in progress" in str(exc):
                pass
            else:
                logger.warning("remove_container %s failed: %s", cid[:12], exc)
        finally:
            with self._live_guard:
                self._live.discard(cid)

    async def remove_container(self, cid: str) -> None:
        await self.run_ctl(self.remove_container_sync, cid)

    def forget_container(self, cid: str) -> None:
        """Stop tracking ``cid`` so exit cleanup leaves it running (kept sandboxes)."""
        with self._live_guard:
            self._live.discard(cid)

    def keep_container_sync(self, cid: str, name: str) -> str:
        """Leave ``cid`` running after this process exits; returns its new name.

        Kept containers are renamed ``<name>-kept`` so orphan reaping and
        orphan pruning skip them.
        """
        self.forget_container(cid)
        new = name if name.endswith("-kept") else f"{name[:58]}-kept"
        try:
            self.client.api.rename(cid, new)
        except DockerException as exc:  # still kept, just not marked
            logger.warning("could not mark %s as kept: %s", cid[:12], exc)
            return name
        return new

    def list_containers_sync(self, labels: list[str] | None = None) -> list[dict[str, Any]]:
        filters = {"label": ["agentbox=1", *(labels or [])]}
        return list(self.client.api.containers(all=True, filters=filters) or [])

    def commit_sync(self, cid: str, *, repository: str, tag: str) -> str:
        self.client.api.commit(cid, repository=repository, tag=tag)
        return f"{repository}:{tag}"

    # ------------------------------------------------------------------- exec
    def kill_marker_sync(self, cid: str, marker: str) -> None:
        """Kill every process in ``cid`` whose environment carries ``marker``."""
        api = self.client.api
        try:
            ex = api.exec_create(
                cid,
                ["/bin/sh", "-c", _KILL_SCRIPT.format(marker=f"{EXEC_MARKER_ENV}={marker}")],
                stdout=True,
                stderr=True,
                user="0",
            )
            api.exec_start(ex["Id"])
        except (NotFound, APIError) as exc:
            logger.debug("kill_marker %s failed: %s", cid[:12], exc)

    def exec_sync(
        self,
        cid: str,
        cmd: list[str],
        *,
        timeout_s: float | None = 60.0,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
        user: str | None = None,
        max_output_bytes: int = 256 * 1024,
        kill_output_bytes: int | None = 64 * 1024 * 1024,
        marker: str | None = None,
        stdout: OutputCapture | None = None,
        stderr: OutputCapture | None = None,
    ) -> ExecResult:
        """Run ``cmd`` to completion (or deadline) streaming into bounded buffers."""
        marker = marker or uuid.uuid4().hex
        out = stdout or OutputCapture(max_output_bytes)
        err = stderr or OutputCapture(max_output_bytes)
        exec_env = dict(env or {})
        exec_env[EXEC_MARKER_ENV] = marker
        api = self.client.api
        t0 = time.monotonic()
        timed_out = threading.Event()
        killed = False

        def on_deadline() -> None:
            timed_out.set()
            self.kill_marker_sync(cid, marker)

        timer: threading.Timer | None = None
        try:
            ex = api.exec_create(
                cid,
                cmd,
                stdout=True,
                stderr=True,
                workdir=workdir,
                environment=exec_env,
                user=user or "",
            )
        except NotFound as exc:
            raise SandboxError(f"container {cid[:12]} not found") from exc
        except APIError as exc:
            raise SandboxError(f"exec_create failed: {exc}") from exc
        if timeout_s is not None:
            timer = threading.Timer(max(0.0, timeout_s), on_deadline)
            timer.daemon = True
            timer.start()
        try:
            stream = api.exec_start(ex["Id"], stream=True, demux=True)
            for chunk_out, chunk_err in stream:
                if chunk_out:
                    out.feed(chunk_out)
                if chunk_err:
                    err.feed(chunk_err)
                if (
                    kill_output_bytes is not None
                    and not killed
                    and out.total + err.total > kill_output_bytes
                ):
                    killed = True
                    # Never kill from the reading thread: the daemon blocks the
                    # kill exec while this stream is not being drained.
                    self._ctl.submit(self.kill_marker_sync, cid, marker)
            code = api.exec_inspect(ex["Id"]).get("ExitCode")
        except NotFound as exc:
            raise SandboxError(f"container {cid[:12]} vanished during exec") from exc
        finally:
            if timer is not None:
                timer.cancel()
        duration = time.monotonic() - t0
        stdout_s, out_trunc = out.text()
        stderr_s, err_trunc = err.text()
        if killed:
            note = f"\n[agentbox] output limit exceeded ({kill_output_bytes} bytes); process killed"
            stderr_s += note
        if timed_out.is_set():
            note = f"Command timed out after {timeout_s}s"
            stderr_s = f"{stderr_s}\n{note}" if stderr_s else note
        return ExecResult(
            exit_code=-1 if timed_out.is_set() else (int(code) if code is not None else -1),
            stdout=stdout_s,
            stderr=stderr_s,
            duration_s=duration,
            timed_out=timed_out.is_set(),
            stdout_truncated=out_trunc,
            stderr_truncated=err_trunc,
            output_bytes=out.total + err.total,
            killed=killed,
        )

    async def exec(
        self,
        cid: str,
        cmd: list[str],
        *,
        timeout_s: float | None = 60.0,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
        user: str | None = None,
        max_output_bytes: int = 256 * 1024,
        kill_output_bytes: int | None = 64 * 1024 * 1024,
    ) -> ExecResult:
        marker = uuid.uuid4().hex
        out = OutputCapture(max_output_bytes)
        err = OutputCapture(max_output_bytes)
        t0 = time.monotonic()
        fut = asyncio.ensure_future(
            self.run_io(
                self.exec_sync,
                cid,
                cmd,
                timeout_s=timeout_s,
                workdir=workdir,
                env=env,
                user=user,
                max_output_bytes=max_output_bytes,
                kill_output_bytes=kill_output_bytes,
                marker=marker,
                stdout=out,
                stderr=err,
            )
        )
        # Abandoned futures must not log "exception was never retrieved".
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())
        safety = None if timeout_s is None else timeout_s + self.kill_grace_s + 10.0
        try:
            return await asyncio.wait_for(asyncio.shield(fut), safety)
        except TimeoutError:
            # The kill did not end the stream (no /bin/sh+grep, or the process
            # escaped its marker). Abandon the stream; it ends when the
            # container is removed, which also releases the thread.
            logger.warning("exec in %s did not stop after kill; abandoning stream", cid[:12])
            stdout_s, out_trunc = out.text()
            stderr_s, err_trunc = err.text()
            return ExecResult(
                exit_code=-1,
                stdout=stdout_s,
                stderr=(stderr_s + f"\nCommand timed out after {timeout_s}s").lstrip("\n"),
                duration_s=time.monotonic() - t0,
                timed_out=True,
                stdout_truncated=out_trunc,
                stderr_truncated=err_trunc,
                output_bytes=out.total + err.total,
                killed=True,
            )
        except asyncio.CancelledError:
            # Caller gave up (episode deadline / shutdown): kill in the background.
            self._ctl.submit(self.kill_marker_sync, cid, marker)
            raise

    # --------------------------------------------------------------- archives
    def put_files_sync(
        self,
        cid: str,
        root: str,
        files: Mapping[str, bytes],
        *,
        modes: Mapping[str, int] | None = None,
        uid: int = 0,
        gid: int = 0,
    ) -> None:
        data = build_tar(files, modes=modes, uid=uid, gid=gid)
        try:
            ok = self.client.api.put_archive(cid, root, data)
        except NotFound as exc:
            raise SandboxError(f"put_archive: {root} not found in {cid[:12]}") from exc
        except APIError as exc:
            raise SandboxError(f"put_archive failed under {root}: {exc}") from exc
        if not ok:
            raise SandboxError(f"put_archive failed under {root}")

    async def put_files(self, cid: str, root: str, files: Mapping[str, bytes], **kw: Any) -> None:
        await self.run_ctl(self.put_files_sync, cid, root, files, **kw)

    def read_file_sync(self, cid: str, path: str, *, max_bytes: int | None = None) -> ReadResult:
        try:
            stream, stat = self.client.api.get_archive(cid, path)
        except NotFound as exc:
            raise FileNotFoundError(path) from exc
        except APIError as exc:
            raise SandboxError(f"get_archive failed for {path}: {exc}") from exc
        mode = int(stat.get("mode", 0))
        size = int(stat.get("size", 0))
        if mode & _GO_MODE_DIR:
            _close_quietly(stream)
            raise IsADirectoryError(path)
        if mode & _GO_MODE_SYMLINK:
            _close_quietly(stream)
            return ReadResult(b"", size, False, link_target=str(stat.get("linkTarget") or ""))
        try:
            with tarfile.open(fileobj=_ChunkReader(iter(stream)), mode="r|") as tar:
                member = tar.next()
                if member is None:
                    raise FileNotFoundError(path)
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise FileNotFoundError(path)
                if max_bytes is None:
                    data = extracted.read()
                    truncated = False
                else:
                    data = extracted.read(max_bytes + 1)
                    truncated = len(data) > max_bytes
                    data = data[:max_bytes]
        finally:
            _close_quietly(stream)
        return ReadResult(data, size or len(data), truncated)

    async def read_file(self, cid: str, path: str, *, max_bytes: int | None = None) -> ReadResult:
        return await self.run_ctl(self.read_file_sync, cid, path, max_bytes=max_bytes)

    # -------------------------------------------------------------- lifecycle
    def live_containers(self) -> list[str]:
        with self._live_guard:
            return list(self._live)

    def cleanup_live_sync(self) -> int:
        """Remove every container this process created and still tracks."""
        ids = self.live_containers()
        for cid in ids:
            self.remove_container_sync(cid)
        return len(ids)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.cleanup_live_sync()
        finally:
            self._io.shutdown(wait=False, cancel_futures=True)
            self._ctl.shutdown(wait=False, cancel_futures=True)
            if not self._external_client:
                try:
                    self.client.close()
                except Exception:  # pragma: no cover - best effort
                    pass


def _close_quietly(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if close is not None:
        try:
            close()
        except Exception:  # pragma: no cover - best effort
            pass


_BACKENDS: weakref.WeakSet[DockerBackend] = weakref.WeakSet()
_default_backend: DockerBackend | None = None
_default_lock = threading.Lock()
_reaped = False


def get_backend() -> DockerBackend:
    """Process-wide shared backend (created lazily, thread-safe).

    The first backend of a process also removes, in the background, sandboxes
    left behind by dead processes on this host (``AGENTBOX_REAP_ORPHANS=0``
    disables it).
    """
    global _default_backend, _reaped
    with _default_lock:
        if _default_backend is None or _default_backend.closed:
            _default_backend = DockerBackend()
            if not _reaped and os.environ.get("AGENTBOX_REAP_ORPHANS", "1") != "0":
                _reaped = True
                from agentbox.sandbox.prune import reap_orphans

                threading.Thread(
                    target=reap_orphans,
                    args=(_default_backend,),
                    name="agentbox-reaper",
                    daemon=True,
                ).start()
        return _default_backend


def get_client(docker_client: Any | None = None) -> Any:
    """Return ``docker_client`` or the shared backend's client (compat helper)."""
    if docker_client is not None:
        return docker_client
    return get_backend().client


@atexit.register
def _cleanup_at_exit() -> None:  # pragma: no cover - exercised at interpreter exit
    for backend in list(_BACKENDS):
        try:
            removed = backend.cleanup_live_sync()
            if removed:
                logger.info("removed %d sandbox(es) left running at exit", removed)
        except Exception:
            pass
