"""Parallel mock rollouts demo."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentbox import Task
from agentbox.config import ResourceLimits, SandboxConfig
from agentbox.metrics import aggregate_trajectories
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.runner.parallel import ParallelRunner
from agentbox.trajectory.formats.jsonl import export_jsonl
from agentbox.trajectory.schema import FunctionCall, ToolCall


def _mock_that_writes(path: str, content: str) -> MockModelClient:
    return MockModelClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(
                            name="write_file",
                            arguments=json.dumps({"path": path, "content": content}),
                        ),
                    )
                ],
            ),
            ModelResponse(content="done", tool_calls=[]),
        ]
    )


async def main() -> None:
    task = Task.from_json(Path(__file__).parent / "tasks" / "fix_fizzbuzz" / "task.json")
    fixed = '''def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
'''
    # Note: each rollout needs its own mock scripted client — use a factory via
    # simple shared client that always returns same script (stateful idx resets per client).
    # For demo, run with one mock client that has enough scripted responses for 1 rollout;
    # ParallelRunner creates separate Rollout.run calls so we need fresh clients.
    # Workaround: pass model as a factory isn't supported — use script long enough or
    # run sequential for mock. Here we use n=1 task with concurrency and a custom approach.

    sandbox = SandboxConfig(
        limits=ResourceLimits(network_disabled=False, memory_mb=512),
        ensure_pytest=True,
    )

    async def one() -> None:
        mock = _mock_that_writes("fizzbuzz.py", fixed)
        from agentbox import Rollout

        return await Rollout.run(task=task, model=mock, sandbox=sandbox)

    trajs = await asyncio.gather(*[one() for _ in range(3)])
    out = Path("out/parallel")
    out.mkdir(parents=True, exist_ok=True)
    export_jsonl(trajs, out / "dataset.jsonl")
    stats = aggregate_trajectories(trajs)
    print(stats.model_dump())
    print("saved", out / "dataset.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
