"""SandboxManager — Docker container lifecycle and I/O."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
from pathlib import Path, PurePosixPath
from typing import Any

from agentbox.config import SandboxConfig
from agentbox.errors import PathEscapeError, SandboxError
from agentbox.sandbox import docker_backend
from agentbox.sandbox.paths import relative_to_workspace, resolve_workspace_path
from agentbox.sandbox.types import ExecResult, Sandbox

logger = logging.getLogger("agentbox.sandbox")

_DNS_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _container_name(run_id: str, task_id: str) -> str:
    task_slug = _DNS_SAFE.sub("-", task_id.lower())[:24].strip("-") or "task"
    run_slug = run_id.replace("-", "")[:8]
    rand = secrets.token_hex(2)
    name = f"agentbox-{run_slug}-{task_slug}-{rand}"
    return name[:63]


class SandboxManager:
    """Owns Docker client and container lifecycle for rollouts."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        docker_client: Any | None = None,
    ) -> None:
        self.config = config or SandboxConfig()
        self._client = docker_backend.get_client(docker_client)
        self._containers: dict[str, Any] = {}

    async def ensure_image(self, image: str | None = None) -> None:
        img = image or self.config.image
        await asyncio.to_thread(
            docker_backend.ensure_image,
            self._client,
            img,
            auto_pull=self.config.auto_pull,
        )

    async def create(self, *, task_id: str, run_id: str) -> Sandbox:
        await self.ensure_image()
        name = _container_name(run_id, task_id)
        labels = {
            "agentbox": "1",
            "agentbox.task_id": task_id,
            "agentbox.run_id": run_id,
            **self.config.labels,
        }
        t0 = time.monotonic()
        container = await asyncio.to_thread(
            docker_backend.create_and_start,
            self._client,
            name=name,
            config=self.config,
            labels=labels,
        )
        create_s = time.monotonic() - t0
        sandbox = Sandbox(
            id=container.id,
            name=name,
            task_id=task_id,
            run_id=run_id,
            workspace_dir=self.config.workspace_dir,
            status="running",
            image=self.config.image,
            created_at=time.time(),
            _manager=self,
        )
        self._containers[sandbox.id] = container
        logger.info(
            "sandbox.created id=%s name=%s duration_s=%.3f",
            sandbox.id[:12],
            name,
            create_s,
        )

        if self.config.ensure_pytest:
            await self._maybe_install_pytest(sandbox)

        # Ensure workspace exists
        await self.exec(sandbox, f"mkdir -p {self.config.workspace_dir}", timeout_s=30.0)
        return sandbox

    async def _maybe_install_pytest(self, sandbox: Sandbox) -> None:
        check = await self.exec(
            sandbox, "python -m pytest --version", timeout_s=30.0
        )
        if check.exit_code == 0:
            return
        # Network may be disabled — only works if image has network or pytest baked in
        if self.config.limits.network_disabled:
            logger.warning(
                "pytest missing and network_disabled=True; "
                "use a baked image or set network_disabled=False / ensure_pytest=False"
            )
            return
        install = await self.exec(
            sandbox, "pip install -q pytest", timeout_s=180.0
        )
        if install.exit_code != 0:
            logger.warning("failed to install pytest: %s", install.stderr)

    async def destroy(self, sandbox: Sandbox, *, force: bool = True) -> None:
        container = self._containers.pop(sandbox.id, None)
        if container is None:
            try:
                container = await asyncio.to_thread(
                    self._client.containers.get, sandbox.id
                )
            except Exception:
                sandbox.status = "destroyed"
                return
        await asyncio.to_thread(docker_backend.destroy_container, container, force=force)
        sandbox.status = "destroyed"
        sandbox._manager = None
        logger.info("sandbox.destroyed id=%s", sandbox.id[:12])

    async def reset(self, sandbox: Sandbox) -> Sandbox:
        task_id, run_id = sandbox.task_id, sandbox.run_id
        await self.destroy(sandbox)
        return await self.create(task_id=task_id, run_id=run_id)

    async def exec(
        self,
        sandbox: Sandbox,
        command: str | list[str],
        *,
        timeout_s: float | None = 60.0,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,  # noqa: ARG002 — reserved
    ) -> ExecResult:
        container = self._require_container(sandbox)
        if isinstance(command, str):
            cmd = ["/bin/sh", "-lc", command]
        else:
            cmd = list(command)
        workdir = workdir or self.config.working_dir

        async def _run() -> ExecResult:
            code, out, err, dur = await asyncio.to_thread(
                docker_backend.exec_run,
                container,
                cmd,
                workdir=workdir,
                env=env,
            )
            return ExecResult(
                exit_code=code, stdout=out, stderr=err, duration_s=dur, timed_out=False
            )

        if timeout_s is None:
            return await _run()
        try:
            return await asyncio.wait_for(_run(), timeout=timeout_s)
        except TimeoutError:
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {timeout_s}s",
                duration_s=float(timeout_s),
                timed_out=True,
            )

    async def write_file(
        self,
        sandbox: Sandbox,
        path: str,
        content: str | bytes,
        *,
        mode: int = 0o644,  # noqa: ARG002
    ) -> None:
        abs_path = resolve_workspace_path(sandbox.workspace_dir, path)
        rel = relative_to_workspace(sandbox.workspace_dir, abs_path)
        data = content.encode("utf-8") if isinstance(content, str) else content
        # Ensure parent directory exists inside container
        parent = str(PurePosixPath(abs_path).parent)
        await self.exec(sandbox, f"mkdir -p {parent}", timeout_s=30.0)
        container = self._require_container(sandbox)
        await asyncio.to_thread(
            docker_backend.put_files,
            container,
            {rel: data},
            root=sandbox.workspace_dir,
        )

    async def write_files(
        self, sandbox: Sandbox, files: dict[str, str | bytes]
    ) -> None:
        if not files:
            return
        # Group by ensuring all parent dirs, then one archive
        prepared: dict[str, bytes] = {}
        parents: set[str] = set()
        for path, content in files.items():
            abs_path = resolve_workspace_path(sandbox.workspace_dir, path)
            rel = relative_to_workspace(sandbox.workspace_dir, abs_path)
            prepared[rel] = (
                content.encode("utf-8") if isinstance(content, str) else content
            )
            parents.add(str(PurePosixPath(abs_path).parent))
        if parents:
            mkdir_cmd = "mkdir -p " + " ".join(sorted(parents))
            await self.exec(sandbox, mkdir_cmd, timeout_s=60.0)
        container = self._require_container(sandbox)
        await asyncio.to_thread(
            docker_backend.put_files,
            container,
            prepared,
            root=sandbox.workspace_dir,
        )

    async def read_file(self, sandbox: Sandbox, path: str) -> str:
        abs_path = resolve_workspace_path(sandbox.workspace_dir, path)
        container = self._require_container(sandbox)
        try:
            data = await asyncio.to_thread(
                docker_backend.get_file_bytes, container, abs_path
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(path) from exc
        return data.decode("utf-8", errors="replace")

    async def list_dir(
        self, sandbox: Sandbox, path: str = ".", *, recursive: bool = False
    ) -> list[str]:
        abs_path = resolve_workspace_path(sandbox.workspace_dir, path)
        if recursive:
            cmd = f'find {abs_path} -mindepth 1 -printf "%P\\n" 2>/dev/null || find {abs_path} -mindepth 1 | sed "s|^{abs_path}/||"'
        else:
            cmd = f'ls -1A {abs_path} 2>/dev/null || true'
        result = await self.exec(sandbox, cmd, timeout_s=30.0)
        if result.exit_code != 0 and result.stderr:
            raise SandboxError(f"list_dir failed: {result.stderr}")
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        return lines

    async def copy_from(self, sandbox: Sandbox, path: str, local_path: Path) -> None:
        content = await self.read_file(sandbox, path)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(content, encoding="utf-8")

    async def healthcheck(self, sandbox: Sandbox) -> bool:
        result = await self.exec(sandbox, "true", timeout_s=10.0)
        return result.exit_code == 0 and not result.timed_out

    def _require_container(self, sandbox: Sandbox) -> Any:
        container = self._containers.get(sandbox.id)
        if container is None:
            raise SandboxError(f"Unknown sandbox {sandbox.id[:12]}")
        return container
