"""Post-seed environment health checks (not task success scoring)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from agentbox.benchmark.schema import SetupCheckSpec
from agentbox.sandbox.types import ExecResult

if TYPE_CHECKING:
    from agentbox.sandbox.manager import SandboxManager
    from agentbox.sandbox.types import Sandbox


class SetupCheckResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ok: bool
    results: list[ExecResult] = Field(default_factory=list)
    failed_check: str | None = None
    error: str | None = None


class SetupChecker:
    """Run suite-level commands after seed, before the agent starts."""

    def __init__(self, manager: SandboxManager) -> None:
        self.manager = manager

    async def run(
        self,
        sandbox: Sandbox,
        checks: list[SetupCheckSpec],
    ) -> SetupCheckResult:
        if not checks:
            return SetupCheckResult(ok=True)

        results: list[ExecResult] = []
        for check in checks:
            result = await self.manager.exec(
                sandbox,
                check.command,
                timeout_s=check.timeout_s,
            )
            results.append(result)
            failed = result.timed_out or result.exit_code != check.success_exit_code
            if failed and check.required:
                err = result.stderr or result.stdout or "setup check failed"
                return SetupCheckResult(
                    ok=False,
                    results=results,
                    failed_check=check.name,
                    error=f"setup check {check.name!r} failed: {err}",
                )
        return SetupCheckResult(ok=True, results=results)
