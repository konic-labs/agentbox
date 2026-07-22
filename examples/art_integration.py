"""Offline ART-style trajectory groups (no training server required).

Shows how AgentBox trajectories map into ART-compatible dicts for GRPO.
If openpipe-art is installed, also builds live art.Trajectory objects.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentbox import Rollout, Task
from agentbox.config import ResourceLimits, SandboxConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.runner.parallel import ParallelRunner
from agentbox.trajectory.schema import FunctionCall, ToolCall


FIXED = '''def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
'''


def make_mock() -> MockModelClient:
    return MockModelClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=FunctionCall(
                            name="write_file",
                            arguments=json.dumps(
                                {"path": "fizzbuzz.py", "content": FIXED}
                            ),
                        ),
                    )
                ],
            ),
            ModelResponse(content="Fixed.", tool_calls=[]),
        ]
    )


async def main() -> None:
    task = Task.from_json(
        Path(__file__).parent / "tasks" / "fix_fizzbuzz" / "task.json"
    )
    sandbox = SandboxConfig(
        limits=ResourceLimits(network_disabled=False),
        ensure_pytest=True,
    )

    # Collect a GRPO group of 4 rollouts (fresh mock each time)
    group = []
    for _ in range(4):
        traj = await Rollout.run(task=task, model=make_mock(), sandbox=sandbox)
        group.append(traj)

    art_group = [t.to_art_dict() for t in group]
    out = Path("out/art_group.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art_group, indent=2), encoding="utf-8")
    print(f"wrote {out} with {len(art_group)} trajectories")
    print("rewards:", [g["reward"] for g in art_group])
    print("correct:", [g["metrics"].get("correct") for g in art_group])

    try:
        live = [t.to_art() for t in group]
        print(f"built {len(live)} live art.Trajectory objects")
    except ImportError:
        print("openpipe-art not installed; skipped live art.Trajectory")


if __name__ == "__main__":
    asyncio.run(main())
