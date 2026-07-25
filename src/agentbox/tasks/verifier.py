"""Run task verifiers inside the sandbox."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

from agentbox.tasks.schema import VerifierSpec
from agentbox.types import VerifierType

if TYPE_CHECKING:
    from agentbox.sandbox.manager import SandboxManager
    from agentbox.sandbox.types import Sandbox


class VerifyResult(BaseModel):
    reward: float
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    command: str


class Verifier:
    def __init__(self, manager: SandboxManager) -> None:
        self.manager = manager

    def build_command(self, spec: VerifierSpec) -> str:
        if spec.type in (VerifierType.PYTEST, VerifierType.TEST):
            if spec.command:
                return spec.command
            path = spec.path or ""
            return f"python -m pytest -q {path}".strip()
        if spec.type in (VerifierType.COMMAND, VerifierType.SHELL):
            if not spec.command:
                raise ValueError("verifier.command required for type=command/shell")
            return spec.command
        if spec.command:
            return spec.command
        raise ValueError(f"Unsupported verifier type: {spec.type}")

    async def verify(self, sandbox: Sandbox, spec: VerifierSpec) -> VerifyResult:
        command = self.build_command(spec)
        t0 = time.monotonic()
        result = await self.manager.exec(
            sandbox, command, timeout_s=spec.timeout_s
        )
        duration = time.monotonic() - t0
        success = (
            not result.timed_out and result.exit_code == spec.success_exit_code
        )
        reward = spec.reward_success if success else spec.reward_failure
        return VerifyResult(
            reward=reward,
            success=success,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_s=duration,
            command=command,
        )
