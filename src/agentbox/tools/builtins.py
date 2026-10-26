"""Default builtin tool sets."""

from __future__ import annotations

from agentbox.tools.base import BaseTool
from agentbox.tools.file_tools import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from agentbox.tools.patch import ApplyPatchTool
from agentbox.tools.search import SearchFilesTool
from agentbox.tools.shell import RunCommandTool
from agentbox.tools.submit import SubmitTool
from agentbox.tools.tests_tool import RunTestsTool


def default_tools() -> list[BaseTool]:
    """The stable ``structured`` set (unchanged so benchmarks stay comparable)."""
    return [
        ListFilesTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        RunCommandTool(),
        RunTestsTool(),
    ]


def extended_tools() -> list[BaseTool]:
    """``structured`` + search, patch and explicit submission."""
    return [*default_tools(), SearchFilesTool(), ApplyPatchTool(), SubmitTool()]


def shell_tools() -> list[BaseTool]:
    return [RunCommandTool()]


def all_builtins() -> dict[str, BaseTool]:
    return {t.name: t for t in extended_tools()}


BUILTIN_NAMES = set(all_builtins())
