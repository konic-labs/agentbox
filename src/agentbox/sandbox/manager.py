"""SandboxManager — container lifecycle and I/O for rollouts.

Managers are cheap: they share the process-wide :class:`DockerBackend` (one
client, bounded pools), so creating one per rollout no longer opens a new Docker
connection (GH #17).
"""

from __future__ import annotations

import logging
import shlex
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from agentbox.config import SandboxConfig
from agentbox.errors import SandboxError
from agentbox.sandbox.backend import ContainerSpec, DockerBackend, container_name, get_backend
from agentbox.sandbox.paths import relative_to_workspace, resolve_workspace_path
from agentbox.sandbox.pool import pool_for
from agentbox.sandbox.types import ExecResult, Sandbox

logger = logging.getLogger("agentbox.sandbox")

_PRUNED_DIRS = (".git", "node_modules", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache")
_MAX_SYMLINK_HOPS = 8


class SandboxManager:
    """Creates sandboxes from a :class:`SandboxConfig` and performs their I/O."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        docker_client: Any | None = None,
        *,
        backend: DockerBackend | None = None,
    ) -> None:
        self.config = config or SandboxConfig()
        self._owns_backend = backend is None and docker_client is not None
        if backend is not None:
            self.backend = backend
        elif docker_client is not None:
            self.backend = DockerBackend(client=docker_client)
        else:
            self.backend = get_backend()
        self._shells: dict[str, list[str]] = {}

    # ---------------------------------------------------------------- context
    async def __aenter__(self) -> SandboxManager:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release the backend if this manager created it (shared ones stay open)."""
        if self._owns_backend:
            self.backend.close()

    # ---------------------------------------------------------------- images
    async def ensure_image(self, image: str | None = None) -> None:
        await self.backend.ensure_image(image or self.config.image, auto_pull=self.config.auto_pull)

    async def resolve_image(self, image: str | None = None) -> str:
        """Return the image to start sandboxes from (base, or base + pytest).

        ``ensure_pytest`` installs pytest **once per image** into a cached derived
        image instead of once per container (GH #16).
        """
        cfg = self.config
        image = image or cfg.image
        await self.backend.ensure_image(image, auto_pull=cfg.auto_pull)
        profile = await self.backend.image_profile(image)
        self._shells.setdefault(
            image, ["/bin/bash", "-c"] if profile.has_bash else ["/bin/sh", "-c"]
        )
        if not cfg.ensure_pytest or profile.has_pytest or not profile.has_python:
            return image
        if not cfg.prepare_network and cfg.limits.network_disabled:
            logger.warning(
                "pytest missing in %s and prepare_network=False; verifiers using pytest will fail",
                image,
            )
            return image
        try:
            derived = await self.backend.derive_image(
                image,
                steps=["python -m pip install -q --disable-pip-version-check pytest"],
                workdir=cfg.workspace_dir,
                network=cfg.prepare_network or not cfg.limits.network_disabled,
                key_extra={"purpose": "pytest"},
                max_images=cfg.cache_max_images,
            )
        except SandboxError as exc:
            logger.warning("could not prepare pytest image from %s: %s", image, exc)
            return image
        self._shells.setdefault(derived, self._shells[image])
        return derived

    def shell_for(self, image: str) -> list[str]:
        return list(self.config.shell or self._shells.get(image) or ["/bin/sh", "-c"])

    # ------------------------------------------------------------- lifecycle
    def container_spec(self, *, image: str, name: str, labels: Mapping[str, str]) -> ContainerSpec:
        cfg = self.config
        lim = cfg.limits
        sec = cfg.security
        return ContainerSpec(
            image=image,
            name=name,
            command=list(cfg.command),
            working_dir=cfg.working_dir,
            labels=dict(labels),
            env=dict(cfg.env),
            network_disabled=lim.network_disabled,
            cpu_count=lim.cpu_count,
            memory_mb=lim.memory_mb,
            memswap_mb=lim.memswap_mb,
            pids_limit=lim.pids_limit,
            shm_size_mb=lim.shm_size_mb,
            nofile=lim.nofile,
            read_only_rootfs=lim.read_only_rootfs,
            tmpfs=dict(lim.tmpfs),
            cap_drop=list(sec.cap_drop),
            cap_add=list(sec.cap_add),
            no_new_privileges=sec.no_new_privileges,
            init=sec.init,
            user=sec.user,
            runtime=sec.runtime,
        )

    async def create(
        self,
        *,
        task_id: str,
        run_id: str,
        rollout_id: str | None = None,
        image: str | None = None,
        labels: Mapping[str, str] | None = None,
        resolve: bool = True,
    ) -> Sandbox:
        """Create and start a sandbox.

        ``image`` overrides the configured image; with ``resolve=False`` it is used
        verbatim (e.g. a cached setup snapshot that already contains pytest).
        """
        cfg = self.config
        if resolve or image is None:
            image = await self.resolve_image(image)
        else:
            await self.backend.ensure_image(image, auto_pull=cfg.auto_pull)
        name = container_name(task_id)
        all_labels = self.backend._labels(
            {
                "agentbox.task_id": task_id[:256],
                "agentbox.run_id": run_id,
                "agentbox.rollout_id": rollout_id or "",
                **cfg.labels,
                **(labels or {}),
            }
        )
        t0 = time.monotonic()
        spec = self.container_spec(image=image, name=name, labels=all_labels)
        cid: str | None = None
        if cfg.warm_pool > 0:
            pool = pool_for(self.backend, ttl_s=cfg.warm_pool_ttl_s)
            cid = await self.backend.run_ctl_owned(
                pool.acquire_sync,
                spec,
                size=cfg.warm_pool,
                prepare=self._prepare_sync,
                extra=cfg.workspace_dir,
            )
        pooled = cid is not None
        if cid is None:
            cid = await self.backend.create_container(spec)
        sandbox = Sandbox(
            id=cid,
            name=name,
            task_id=task_id,
            run_id=run_id,
            rollout_id=rollout_id or "",
            workspace_dir=cfg.workspace_dir,
            status="running",
            image=image,
            created_at=time.time(),
            _manager=self,
        )
        if cfg.workspace_dir != cfg.working_dir and not pooled:
            # Docker only auto-creates working_dir.
            await self.exec(sandbox, ["mkdir", "-p", "--", cfg.workspace_dir], timeout_s=30.0)
        logger.debug(
            "sandbox.created id=%s name=%s image=%s pooled=%s duration_s=%.3f",
            cid[:12],
            name,
            image,
            pooled,
            time.monotonic() - t0,
        )
        return sandbox

    def _prepare_sync(self, cid: str) -> None:
        """Task-independent setup done ahead of time for pooled containers."""
        cfg = self.config
        if cfg.workspace_dir == cfg.working_dir:
            return
        res = self.backend.exec_sync(
            cid,
            ["mkdir", "-p", "--", cfg.workspace_dir],
            timeout_s=30.0,
            workdir=cfg.working_dir,
            user=cfg.security.user,
        )
        if res.exit_code != 0:
            raise SandboxError(f"mkdir {cfg.workspace_dir} failed: {res.stderr or res.stdout}")

    async def warm(self, n: int | None = None, *, image: str | None = None) -> int:
        """Pre-start ``n`` (default ``warm_pool``) containers of this config's default shape.

        Returns immediately; containers are created in the background. Rollouts
        whose task overrides the image or environment use their own shapes.
        """
        cfg = self.config
        count = cfg.warm_pool if n is None else n
        if count <= 0:
            return 0
        resolved = await self.resolve_image(image)
        spec = self.container_spec(
            image=resolved, name="", labels=self.backend._labels(dict(cfg.labels))
        )
        pool_for(self.backend, ttl_s=cfg.warm_pool_ttl_s).prefill(
            spec, count, prepare=self._prepare_sync, extra=cfg.workspace_dir
        )
        return count

    async def destroy(self, sandbox: Sandbox, *, force: bool = True) -> None:  # noqa: ARG002
        if sandbox.status == "destroyed":
            return
        await self.backend.remove_container(sandbox.id)
        sandbox.status = "destroyed"
        sandbox._manager = None
        logger.debug("sandbox.destroyed id=%s", sandbox.id[:12])

    def keep(self, sandbox: Sandbox) -> None:
        """Leave ``sandbox`` running after the process exits (debugging).

        The container is renamed ``<name>-kept`` (``sandbox.name`` follows) so
        automatic orphan reaping never removes it.
        """
        sandbox.name = self.backend.keep_container_sync(sandbox.id, sandbox.name)

    async def reset(self, sandbox: Sandbox) -> Sandbox:
        task_id, run_id, rollout_id = sandbox.task_id, sandbox.run_id, sandbox.rollout_id
        image = sandbox.image
        await self.destroy(sandbox)
        return await self.create(
            task_id=task_id, run_id=run_id, rollout_id=rollout_id, image=image, resolve=False
        )

    # ------------------------------------------------------------------- exec
    async def exec(
        self,
        sandbox: Sandbox,
        command: str | list[str],
        *,
        timeout_s: float | None = 60.0,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        max_output_bytes: int | None = None,
    ) -> ExecResult:
        if sandbox.status == "destroyed":
            raise SandboxError(f"Sandbox {sandbox.id[:12]} is destroyed")
        cmd = (
            [*self.shell_for(sandbox.image), command] if isinstance(command, str) else list(command)
        )
        cfg = self.config
        return await self.backend.exec(
            sandbox.id,
            cmd,
            timeout_s=timeout_s,
            workdir=workdir or cfg.working_dir,
            env=env,
            user=user or cfg.security.user,
            max_output_bytes=max_output_bytes or cfg.max_output_bytes,
            kill_output_bytes=cfg.kill_output_bytes,
        )

    async def healthcheck(self, sandbox: Sandbox) -> bool:
        result = await self.exec(sandbox, ["true"], timeout_s=10.0)
        return result.ok

    # ------------------------------------------------------------------- files
    def _owner(self) -> tuple[int, int]:
        user = self.config.security.user
        if not user:
            return 0, 0
        uid_s, _, gid_s = user.partition(":")
        try:
            uid = int(uid_s)
            return uid, int(gid_s) if gid_s else uid
        except ValueError:
            return 0, 0  # named users: let the daemon keep root ownership

    async def write_file(
        self,
        sandbox: Sandbox,
        path: str,
        content: str | bytes,
        *,
        mode: int = 0o644,
    ) -> None:
        abs_path = resolve_workspace_path(sandbox.workspace_dir, path)
        rel = relative_to_workspace(sandbox.workspace_dir, abs_path)
        if rel in ("", "."):
            raise IsADirectoryError(path)
        data = content.encode("utf-8") if isinstance(content, str) else content
        uid, gid = self._owner()
        await self.backend.put_files(
            sandbox.id, sandbox.workspace_dir, {rel: data}, modes={rel: mode}, uid=uid, gid=gid
        )

    async def write_files(self, sandbox: Sandbox, files: Mapping[str, str | bytes]) -> None:
        """Write many files with a single ``put_archive`` call."""
        if not files:
            return
        prepared: dict[str, bytes] = {}
        for path, content in files.items():
            abs_path = resolve_workspace_path(sandbox.workspace_dir, path)
            rel = relative_to_workspace(sandbox.workspace_dir, abs_path)
            if rel in ("", "."):
                raise IsADirectoryError(path)
            prepared[rel] = content.encode("utf-8") if isinstance(content, str) else content
        uid, gid = self._owner()
        await self.backend.put_files(sandbox.id, sandbox.workspace_dir, prepared, uid=uid, gid=gid)

    async def read_bytes(
        self, sandbox: Sandbox, path: str, *, max_bytes: int | None = None
    ) -> bytes:
        """Read a workspace file (following symlinks that stay inside the jail)."""
        abs_path = resolve_workspace_path(sandbox.workspace_dir, path)
        for _ in range(_MAX_SYMLINK_HOPS):
            res = await self.backend.read_file(sandbox.id, abs_path, max_bytes=max_bytes)
            if res.link_target is None:
                return res.data
            target = res.link_target
            if not target.startswith("/"):
                target = str(PurePosixPath(abs_path).parent / target)
            abs_path = resolve_workspace_path(sandbox.workspace_dir, target)
        raise SandboxError(f"too many levels of symbolic links: {path}")

    async def read_file(self, sandbox: Sandbox, path: str, *, max_bytes: int | None = None) -> str:
        try:
            data = await self.read_bytes(sandbox, path, max_bytes=max_bytes)
        except FileNotFoundError as exc:
            raise FileNotFoundError(path) from exc
        return data.decode("utf-8", errors="replace")

    async def list_dir(
        self,
        sandbox: Sandbox,
        path: str = ".",
        *,
        recursive: bool = False,
        max_entries: int = 2000,
    ) -> list[str]:
        """List entries (directories end with ``/``); errors on missing paths."""
        abs_path = resolve_workspace_path(sandbox.workspace_dir, path)
        q = shlex.quote(abs_path)
        if recursive:
            prune = " -o ".join(f"-name {shlex.quote(d)}" for d in _PRUNED_DIRS)
            script = (
                f"test -d {q} || {{ echo 'not a directory' >&2; exit 2; }}; "
                "if find / -maxdepth 0 -printf '' >/dev/null 2>&1; then "
                f"find {q} -mindepth 1 \\( {prune} \\) -prune -o "
                f"\\( -type d -printf '%P/\\n' -o -printf '%P\\n' \\); "
                f"else find {q} -mindepth 1 \\( {prune} \\) -prune -o -print; fi"
            )
        else:
            script = f"ls -1Ap -- {q}"
        result = await self.exec(sandbox, ["/bin/sh", "-c", script], timeout_s=30.0)
        if result.exit_code != 0:
            raise FileNotFoundError(
                (result.stderr or f"cannot list {path}").strip().splitlines()[-1]
            )
        prefix = abs_path.rstrip("/") + "/"
        entries: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(prefix):  # busybox fallback prints absolute paths
                line = line[len(prefix) :]
            elif line == abs_path:  # listing a file echoes its (absolute) path
                line = relative_to_workspace(sandbox.workspace_dir, abs_path)
            entries.append(line)
            if len(entries) >= max_entries:
                entries.append(f"... (truncated at {max_entries} entries)")
                break
        return entries

    async def copy_from(self, sandbox: Sandbox, path: str, local_path: Path) -> None:
        """Copy a workspace file to the host (binary-safe)."""
        data = await self.read_bytes(sandbox, path)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
