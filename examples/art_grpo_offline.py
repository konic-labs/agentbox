"""Offline GRPO-style group collection for ART.

Runs G mock rollouts of one task and writes an ART-compatible group JSON.
No GPU / ART server required.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentbox import ParallelRunner, Task
from agentbox.config import AgentConfig, ModelConfig, ResourceLimits, RolloutConfig, SandboxConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.trajectory.schema import FunctionCall, ToolCall

ROOT = Path(__file__).resolve().parent
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
                        id="c1",
                        function=FunctionCall(
                            name="write_file",
                            arguments=json.dumps(
                                {"path": "fizzbuzz.py", "content": FIXED}
                            ),
                        ),
                    )
                ],
            ),
            ModelResponse(content="done", tool_calls=[]),
        ]
    )


async def main() -> None:
    task = Task.from_json(ROOT / "tasks" / "fix_fizzbuzz" / "task.json")
    g = 4
    mock = make_mock()
    runner = ParallelRunner(
        concurrency=g,
        config=RolloutConfig(
            model=ModelConfig(model="mock"),
            agent=AgentConfig(max_steps=10),
            sandbox=SandboxConfig(
                limits=ResourceLimits(network_disabled=False),
                ensure_pytest=True,
            ),
        ),
        model=mock,
    )
    trajs = await runner.run_tasks([task], n_per_task=g, progress=False)
    group = [t.to_art_dict() for t in trajs]
    # Group-relative advantages (toy demo)
    rewards = [float(x["reward"]) for x in group]
    mean = sum(rewards) / max(len(rewards), 1)
    advantages = [r - mean for r in rewards]
    out = Path("out/art_grpo_group.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"group": group, "rewards": rewards, "advantages": advantages},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print("rewards:", rewards)
    print("advantages:", advantages)
    print("correct:", [g_["metrics"].get("correct") for g_ in group])


if __name__ == "__main__":
    asyncio.run(main())
