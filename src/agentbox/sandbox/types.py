"""Sandbox data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agentbox.sandbox.manager import SandboxManager


@dataclass
class ExecResult:
    """Result of a command executed inside a container."""

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


@dataclass
class Sandbox:
    """Handle for a running (or created) Docker container.

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
    _manager: Any = field(default=None, repr=False, compare=False)

    async def exec(
        self,
        command: str | list[str],
        *,
        timeout_s: float | None = 60.0,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if self._manager is None:
            raise RuntimeError("Sandbox is not bound to a SandboxManager")
        return await self._manager.exec(
            self, command, timeout_s=timeout_s, workdir=workdir, env=env
        )

    async def read_text(self, path: str) -> str:
        if self._manager is None:
            raise RuntimeError("Sandbox is not bound to a SandboxManager")
        return await self._manager.read_file(self, path)

    async def write_text(self, path: str, content: str) -> None:
        if self._manager is None:
            raise RuntimeError("Sandbox is not bound to a SandboxManager")
        await self._manager.write_file(self, path, content)

    async def list_dir(self, path: str = ".", *, recursive: bool = False) -> list[str]:
        if self._manager is None:
            raise RuntimeError("Sandbox is not bound to a SandboxManager")
        return await self._manager.list_dir(self, path, recursive=recursive)
