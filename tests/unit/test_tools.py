"""Unit tests for tools and registry."""

from __future__ import annotations

import pytest

from agentbox.tools.base import BaseTool
from agentbox.tools.builtins import default_tools
from agentbox.tools.decorator import tool
from agentbox.tools.file_tools import EditFileTool
from agentbox.tools.registry import build_tool_registry
from agentbox.types import ToolMode


def test_default_tools_count() -> None:
    tools = default_tools()
    names = {t.name for t in tools}
    assert names == {
        "list_files",
        "read_file",
        "write_file",
        "edit_file",
        "run_command",
        "run_tests",
    }


def test_edit_file_schema_params() -> None:
    schema = EditFileTool().openai_schema()
    props = schema["function"]["parameters"]["properties"]
    assert "old_string" in props
    assert "new_string" in props


def test_tool_decorator() -> None:
    @tool(description="echo")
    async def my_echo(sandbox, text: str) -> str:
        return text

    assert isinstance(my_echo, BaseTool)
    assert my_echo.name == "my_echo"
    schema = my_echo.openai_schema()
    assert schema["function"]["parameters"]["required"] == ["text"]


def test_registry_override_and_custom_only() -> None:
    class FakeRead(BaseTool):
        name = "read_file"
        description = "fake"

        def parameters(self):
            return {"type": "object", "properties": {}}

        async def execute(self, sandbox, **kwargs):
            return "fake"

    reg = build_tool_registry(ToolMode.STRUCTURED, custom_tools=[FakeRead()])
    assert isinstance(reg.get("read_file"), FakeRead)

    reg2 = build_tool_registry(
        ToolMode.CUSTOM, custom_tools=[FakeRead()], include_builtins=False
    )
    assert reg2.names() == ["read_file"]


def test_shell_mode() -> None:
    reg = build_tool_registry(ToolMode.SHELL)
    assert reg.names() == ["run_command"]
