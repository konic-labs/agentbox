"""Sandbox data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agentbox.sandbox.manager import SandboxManager


@dataclass(slots=True)
class ExecResult:
    """Result of a command executed inside a container.

    ``stdout``/``stderr`` hold at most ``SandboxConfig.max_output_bytes`` per
    stream (head + tail); ``output_bytes`` is the total produced.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    output_bytes: int = 0
    killed: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass(slots=True)
class ImageProfile:
    """Capabilities of an image, probed once per image per process."""

    image: str
    image_id: str
    has_bash: bool = False
    has_python: bool = False
    has_pytest: bool = False


@dataclass
class Sandbox:
    """Handle for a running Docker container.

    Public async methods form the narrow surface passed to tools.
    """

    id: str
    name: str
    task_id: str
    run_id: str
    workspace_dir: str
    status: Literal["created", "running", "exited", "destroyed"]
    image: str
    created_at: float
    rollout_id: str = ""
    _manager: Any = field(default=None, repr=False, compare=False)

    def _require_manager(self) -> SandboxManager:
        if self._manager is None:
            raise RuntimeError("Sandbox is not bound to a SandboxManager")
        manager: SandboxManager = self._manager
        return manager

    async def exec(
        self,
        command: str | list[str],
        *,
        timeout_s: float | None = 60.0,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        max_output_bytes: int | None = None,
    ) -> ExecResult:
        return await self._require_manager().exec(
            self,
            command,
            timeout_s=timeout_s,
            workdir=workdir,
            env=env,
            max_output_bytes=max_output_bytes,
        )

    async def read_text(self, path: str, *, max_bytes: int | None = None) -> str:
        return await self._require_manager().read_file(self, path, max_bytes=max_bytes)

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        return await self._require_manager().read_bytes(self, path, max_bytes=max_bytes)

    async def write_text(self, path: str, content: str) -> None:
        await self._require_manager().write_file(self, path, content)

    async def write_bytes(self, path: str, content: bytes, *, mode: int = 0o644) -> None:
        await self._require_manager().write_file(self, path, content, mode=mode)

    async def write_files(self, files: dict[str, str | bytes]) -> None:
        await self._require_manager().write_files(self, files)

    async def list_dir(
        self, path: str = ".", *, recursive: bool = False, max_entries: int = 2000
    ) -> list[str]:
        return await self._require_manager().list_dir(
            self, path, recursive=recursive, max_entries=max_entries
        )
