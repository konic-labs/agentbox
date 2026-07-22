"""SetupChecker unit tests with fake manager."""

import pytest

from agentbox.benchmark.schema import SetupCheckSpec
from agentbox.benchmark.setup_check import SetupChecker
from agentbox.sandbox.types import ExecResult


class FakeManager:
    def __init__(self, exit_codes: list[int]) -> None:
        self.exit_codes = list(exit_codes)

    async def exec(self, sandbox, command, timeout_s=60.0):
        code = self.exit_codes.pop(0) if self.exit_codes else 0
        return ExecResult(
            exit_code=code,
            stdout="",
            stderr="fail" if code else "ok",
            duration_s=0.01,
            timed_out=False,
        )


@pytest.mark.asyncio
async def test_setup_check_fail_closed() -> None:
    checker = SetupChecker(FakeManager([1]))  # type: ignore[arg-type]
    res = await checker.run(
        None,  # type: ignore[arg-type]
        [SetupCheckSpec(name="bad", command="false")],
    )
    assert not res.ok
    assert res.failed_check == "bad"


@pytest.mark.asyncio
async def test_setup_check_ok() -> None:
    checker = SetupChecker(FakeManager([0, 0]))  # type: ignore[arg-type]
    res = await checker.run(
        None,  # type: ignore[arg-type]
        [
            SetupCheckSpec(name="a", command="true"),
            SetupCheckSpec(name="b", command="true"),
        ],
    )
    assert res.ok
