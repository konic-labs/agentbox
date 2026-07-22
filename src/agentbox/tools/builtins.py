"""Default builtin tool set."""

from __future__ import annotations

from agentbox.tools.base import BaseTool
from agentbox.tools.file_tools import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from agentbox.tools.shell import RunCommandTool
from agentbox.tools.tests_tool import RunTestsTool


def default_tools() -> list[BaseTool]:
    return [
        ListFilesTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        RunCommandTool(),
        RunTestsTool(),
    ]


def shell_tools() -> list[BaseTool]:
    return [RunCommandTool()]


BUILTIN_NAMES = {t.name for t in default_tools()}
