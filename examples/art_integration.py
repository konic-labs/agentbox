"""ART-shaped GRPO groups from AgentBox rollouts (no ART server required).

Collects a group of mock rollouts, converts it with the ART adapter, and writes
the portable dicts. With `pip install "agentbox-rl[art]"` it also builds live
``art.TrajectoryGroup`` objects (what ``backend.train(model, groups)`` takes).

    python examples/art_integration.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentbox import ParallelRunner, SandboxConfig, Task
from agentbox.integrations.art import to_art_dict, to_art_groups
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.rl import group_advantages
from agentbox.trajectory.schema import FunctionCall, ToolCall

FIXED = """def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
"""


def scripted_policy() -> MockModelClient:
    write = ToolCall(
        id="call_1",
        function=FunctionCall(
            name="write_file", arguments=json.dumps({"path": "fizzbuzz.py", "content": FIXED})
        ),
    )
    return MockModelClient([ModelResponse(tool_calls=[write]), ModelResponse(content="Fixed.")])


async def main() -> None:
    task = Task.from_json(Path(__file__).parent / "tasks" / "fix_fizzbuzz" / "task.json")
    runner = ParallelRunner(model=scripted_policy(), sandbox=SandboxConfig(), concurrency=4)
    [group] = await runner.run_groups([task], group_size=4)

    rewards = [t.reward for t in group]
    print("rewards:", rewards, "GRPO advantages:", group_advantages(rewards, method="grpo"))

    out = Path("out/art_group.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([to_art_dict(t) for t in group], indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(group)} ART trajectories)")

    try:
        groups = to_art_groups(group)
    except ImportError:
        print("openpipe-art not installed; skipped live art.TrajectoryGroup")
        return
    sizes = [len(g.trajectories) for g in groups]
    print(f"built {len(groups)} art.TrajectoryGroup(s) with {sizes} trajectories")


if __name__ == "__main__":
    asyncio.run(main())
