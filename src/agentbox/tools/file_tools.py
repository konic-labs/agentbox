"""File system tools (list/read/write/edit)."""

from __future__ import annotations

from typing import Any

from agentbox.errors import PathEscapeError
from agentbox.tools.base import BaseTool, ToolResult

READ_CHAR_LIMIT = 200_000


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List files and directories under the workspace path."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory relative to /workspace",
                    "default": ".",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List recursively",
                    "default": False,
                },
            },
            "additionalProperties": False,
        }

    async def execute(
        self, sandbox: Any, path: str = ".", recursive: bool = False, **_: Any
    ) -> str | ToolResult:
        try:
            entries = await sandbox.list_dir(path, recursive=bool(recursive))
        except (PathEscapeError, FileNotFoundError, Exception) as exc:
            return ToolResult(content=f"ERROR: list_files failed: {exc}", is_error=True)
        if not entries:
            return "(empty)"
        return "\n".join(entries)


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the full content of a file under the workspace."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path under /workspace"},
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    async def execute(self, sandbox: Any, path: str, **_: Any) -> str | ToolResult:
        try:
            content = await sandbox.read_text(path)
        except FileNotFoundError:
            return ToolResult(
                content=f"ERROR: read_file failed: file not found: {path}",
                is_error=True,
            )
        except (PathEscapeError, Exception) as exc:
            return ToolResult(content=f"ERROR: read_file failed: {exc}", is_error=True)

        if len(content) > READ_CHAR_LIMIT:
            truncated = content[:READ_CHAR_LIMIT]
            return (
                truncated
                + f"\n\n...[truncated {len(content) - READ_CHAR_LIMIT} chars]"
            )
        return content


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create or overwrite a file under the workspace."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    async def execute(
        self, sandbox: Any, path: str, content: str, **_: Any
    ) -> str | ToolResult:
        try:
            await sandbox.write_text(path, content)
        except (PathEscapeError, Exception) as exc:
            return ToolResult(content=f"ERROR: write_file failed: {exc}", is_error=True)
        return f"Wrote {len(content)} chars to {path}"


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Replace exactly one occurrence of old_string with new_string in a workspace file."
    )

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        sandbox: Any,
        path: str,
        old_string: str,
        new_string: str,
        **_: Any,
    ) -> str | ToolResult:
        try:
            content = await sandbox.read_text(path)
        except FileNotFoundError:
            return ToolResult(
                content=f"ERROR: edit_file failed: file not found: {path}",
                is_error=True,
            )
        except (PathEscapeError, Exception) as exc:
            return ToolResult(content=f"ERROR: edit_file failed: {exc}", is_error=True)

        count = content.count(old_string)
        if count == 0:
            return ToolResult(
                content=f"ERROR: edit_file failed: old_string matched 0 times in {path!r}",
                is_error=True,
            )
        if count > 1:
            return ToolResult(
                content=(
                    f"ERROR: edit_file failed: old_string matched {count} times in "
                    f"{path!r}; must match exactly once"
                ),
                is_error=True,
            )

        updated = content.replace(old_string, new_string, 1)
        try:
            await sandbox.write_text(path, updated)
        except Exception as exc:
            return ToolResult(content=f"ERROR: edit_file failed: {exc}", is_error=True)
        return f"Edited {path}"
