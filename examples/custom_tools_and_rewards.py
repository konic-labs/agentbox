"""Custom tools (@tool and BaseTool) and a custom reward component.

python examples/custom_tools_and_rewards.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agentbox import (
    AgentConfig,
    BaseTool,
    RewardSpec,
    Rollout,
    RolloutConfig,
    SandboxConfig,
    Task,
    VerifierSpec,
    register_reward,
    tool,
)
from agentbox.config import ModelConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.rewards import RewardContext
from agentbox.tools.base import ToolContext, ToolResult
from agentbox.trajectory.schema import FunctionCall, ToolCall


@tool(read_only=True)
async def count_lines(sandbox: Any, path: str, pattern: str | None = None) -> str:
    """Count lines in a workspace file.

    Args:
        path: File path relative to the workspace.
        pattern: Only count lines containing this text.
    """
    text = await sandbox.read_text(path)
    return str(sum(1 for line in text.splitlines() if pattern is None or pattern in line))


@tool(description="Report the step number and the time left in the episode")
def budget(sandbox: Any, ctx: ToolContext) -> str:
    left = ctx.remaining_s()
    return f"step {ctx.step}, {'no' if left is None else round(left)} seconds left"


class Grep(BaseTool):
    name = "grep"
    description = "Search the workspace with grep -rn."
    read_only = True

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"pattern": {"type": "string", "description": "Text to find"}},
            "required": ["pattern"],
        }

    async def execute(self, sandbox: Any, pattern: str, **_: Any) -> str | ToolResult:
        res = await sandbox.exec(["grep", "-rn", "--", pattern, "."], timeout_s=30)
        if res.exit_code > 1:
            return ToolResult(content=res.stderr, is_error=True)
        return res.stdout or "(no matches)"


@register_reward("few_tool_calls")
def few_tool_calls(budget: int = 3):
    def fn(ctx: RewardContext) -> float:
        calls = len(ctx.loop.tool_call_records) if ctx.loop else 0
        return 1.0 if calls <= budget else 0.0

    return fn


def call(i: int, name: str, **args: Any) -> ToolCall:
    return ToolCall(id=f"c{i}", function=FunctionCall(name=name, arguments=json.dumps(args)))


async def main() -> None:
    task = Task(
        task_id="todo_count",
        description="Write the number of TODO lines in notes.txt into answer.txt.",
        starter_files={"notes.txt": "TODO a\ndone b\nTODO c\n"},
        verifier=VerifierSpec(type="command", command='test "$(cat answer.txt)" = 2'),
    )
    policy = MockModelClient(
        [
            ModelResponse(tool_calls=[call(0, "count_lines", path="notes.txt", pattern="TODO")]),
            ModelResponse(tool_calls=[call(1, "grep", pattern="TODO"), call(2, "budget")]),
            ModelResponse(tool_calls=[call(3, "write_file", path="answer.txt", content="2")]),
            ModelResponse(content="answer.txt contains 2"),
        ]
    )
    agent = AgentConfig(tools="structured", custom_tools=[count_lines, budget, Grep()])
    config = RolloutConfig(
        model=ModelConfig(model="scripted"),
        agent=agent,
        sandbox=SandboxConfig(ensure_pytest=False),
        rewards=[
            RewardSpec(name="verifier"),
            RewardSpec(name="few_tool_calls", weight=0.1, params={"budget": 4}),
        ],
    )
    traj = await Rollout.run(task, model=policy, config=config)
    for rec in traj.tool_call_records:
        print(f"{rec.name}({rec.arguments}) -> {rec.result[:60]!r}")
    print("status:", traj.final_status, "reward:", traj.reward, traj.reward_breakdown)


if __name__ == "__main__":
    asyncio.run(main())
