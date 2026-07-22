"""Minimal mock rollout demo (no real LLM required)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentbox import Rollout, Task
from agentbox.config import SandboxConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.trajectory.schema import FunctionCall, ToolCall


async def main() -> None:
    task_path = Path(__file__).parent / "tasks" / "fix_fizzbuzz" / "task.json"
    task = Task.from_json(task_path)

    fixed = '''def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
'''
    mock = MockModelClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(
                            name="write_file",
                            arguments=json.dumps(
                                {"path": "fizzbuzz.py", "content": fixed}
                            ),
                        ),
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                content="Fixed fizzbuzz implementation.",
                tool_calls=[],
                finish_reason="stop",
            ),
        ]
    )

    # ensure_pytest: network disabled by default — install pytest needs image with it
    # or enable network. Use ensure_pytest=False and setup_commands if image has pytest.
    sandbox = SandboxConfig(
        image="python:3.12-slim-bookworm",
        ensure_pytest=True,
        # allow pip install pytest for first-time slim image
        limits=SandboxConfig().limits.model_copy(update={"network_disabled": False}),
    )

    traj = await Rollout.run(task=task, model=mock, sandbox=sandbox)
    print("status:", traj.final_status)
    print("reward:", traj.reward)
    print("steps:", traj.metrics.steps)
    out = Path("out/hello_traj.json")
    traj.save(out)
    print("saved:", out)
    print("art reward:", traj.to_art_dict()["reward"])


if __name__ == "__main__":
    asyncio.run(main())
