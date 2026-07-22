"""Parallel runner isolation tests (no Docker — inject errors via mock tasks)."""

from __future__ import annotations

import pytest

from agentbox.runner.parallel import ParallelRunner
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.types import FinalStatus, VerifierType


@pytest.mark.asyncio
async def test_parallel_result_summary_empty() -> None:
    runner = ParallelRunner(concurrency=2)
    # empty
    result = await runner.run([])
    assert result.succeeded == 0
    assert result.trajectories == []


# Full docker parallel covered in integration; unit checks API shape
def test_parallel_runner_init() -> None:
    with pytest.raises(ValueError):
        ParallelRunner(concurrency=0)
