"""Inject task files and setup_commands into a sandbox."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from agentbox.sandbox.types import ExecResult
from agentbox.tasks.schema import Task

if TYPE_CHECKING:
    from agentbox.sandbox.manager import SandboxManager
    from agentbox.sandbox.types import Sandbox


class SeedResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ok: bool
    files_written: int = 0
    setup_results: list[ExecResult] = Field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0


class TaskSeeder:
    """Materialize a Task into a clean sandbox before the agent runs."""

    def __init__(self, manager: SandboxManager) -> None:
        self.manager = manager

    async def seed(self, sandbox: Sandbox, task: Task) -> SeedResult:
        t0 = time.monotonic()
        setup_results: list[ExecResult] = []
        try:
            if task.starter_files:
                await self.manager.write_files(sandbox, task.starter_files)
            files_written = len(task.starter_files)

            for cmd in task.setup_commands:
                result = await self.manager.exec(sandbox, cmd, timeout_s=300.0)
                setup_results.append(result)
                if result.timed_out or result.exit_code != 0:
                    err = result.stderr or result.stdout or "setup command failed"
                    return SeedResult(
                        ok=False,
                        files_written=files_written,
                        setup_results=setup_results,
                        error=f"setup_commands failed ({cmd!r}): {err}",
                        duration_s=time.monotonic() - t0,
                    )

            healthy = await self.manager.healthcheck(sandbox)
            if not healthy:
                return SeedResult(
                    ok=False,
                    files_written=files_written,
                    setup_results=setup_results,
                    error="sandbox healthcheck failed after seed",
                    duration_s=time.monotonic() - t0,
                )

            return SeedResult(
                ok=True,
                files_written=files_written,
                setup_results=setup_results,
                duration_s=time.monotonic() - t0,
            )
        except Exception as exc:
            return SeedResult(
                ok=False,
                files_written=0,
                setup_results=setup_results,
                error=str(exc),
                duration_s=time.monotonic() - t0,
            )
