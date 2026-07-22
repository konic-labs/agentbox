"""Agent loop tests with MockModelClient."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentbox.agent.loop import AgentLoop
from agentbox.config import AgentConfig
from agentbox.model.base import ModelResponse
from agentbox.model.mock import MockModelClient
from agentbox.tools.base import ToolContext
from agentbox.tools.executor import ToolExecutor
from agentbox.tools.registry import build_tool_registry
from agentbox.trajectory.schema import FunctionCall, ToolCall
from agentbox.types import FinalStatus, ToolMode


class FakeSandbox:
    def __init__(self) -> None:
        self.files: dict[str, str] = {"a.txt": "hello"}

    async def exec(self, command, timeout_s=60.0):
        return SimpleNamespace(
            exit_code=0, stdout="ok", stderr="", duration_s=0.0, timed_out=False
        )

    async def read_text(self, path: str) -> str:
        return self.files[path]

    async def write_text(self, path: str, content: str) -> None:
        self.files[path] = content

    async def list_dir(self, path: str = ".", recursive: bool = False):
        return list(self.files.keys())


@pytest.mark.asyncio
async def test_parallel_tool_calls_and_final() -> None:
    sandbox = FakeSandbox()
    registry = build_tool_registry(ToolMode.STRUCTURED)
    executor = ToolExecutor(registry)
    mock = MockModelClient(
        [
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=FunctionCall(
                            name="read_file",
                            arguments=json.dumps({"path": "a.txt"}),
                        ),
                    ),
                    ToolCall(
                        id="c2",
                        function=FunctionCall(
                            name="list_files",
                            arguments=json.dumps({"path": "."}),
                        ),
                    ),
                ],
            ),
            ModelResponse(content="done", tool_calls=[]),
        ]
    )
    loop = AgentLoop(mock, executor, AgentConfig(max_steps=5))
    ctx = ToolContext(
        sandbox=sandbox,
        manager=None,
        workspace_dir="/workspace",
        run_id="r",
        task_id="t",
    )
    result = await loop.run(
        system_prompt="sys",
        user_prompt="do stuff",
        ctx=ctx,
    )
    assert result.stop_reason == "final_answer"
    assert result.steps == 1
    # system, user, assistant(tools), tool, tool, assistant(final)
    roles = [m.role if isinstance(m.role, str) else m.role.value for m in result.messages]
    assert roles.count("tool") == 2
    tool_ids = [
        m.tool_call_id for m in result.messages if (m.role.value if hasattr(m.role, "value") else m.role) == "tool"
    ]
    assert tool_ids == ["c1", "c2"]


@pytest.mark.asyncio
async def test_max_steps() -> None:
    sandbox = FakeSandbox()
    registry = build_tool_registry(ToolMode.STRUCTURED)
    executor = ToolExecutor(registry)
    # always request tools
    responses = [
        ModelResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"c{i}",
                    function=FunctionCall(
                        name="list_files",
                        arguments="{}",
                    ),
                )
            ],
        )
        for i in range(5)
    ]
    mock = MockModelClient(responses)
    loop = AgentLoop(mock, executor, AgentConfig(max_steps=2))
    ctx = ToolContext(sandbox=sandbox, manager=None, workspace_dir="/workspace")
    result = await loop.run(system_prompt="s", user_prompt="u", ctx=ctx)
    assert result.final_status == FinalStatus.MAX_STEPS
    assert result.steps == 2
