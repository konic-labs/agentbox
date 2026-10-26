"""File system tools (list/read/write/edit)."""

from __future__ import annotations

from typing import Any

from agentbox.errors import PathEscapeError
from agentbox.tools.base import BaseTool, ToolResult

READ_MAX_BYTES = 1024 * 1024  # never pull more than 1 MiB of a file into memory
READ_CHAR_LIMIT = 200_000  # kept for backwards compatibility (executor truncates further)


def _error(tool: str, msg: str) -> ToolResult:
    return ToolResult(content=f"ERROR: {tool} failed: {msg}", is_error=True)


class ListFilesTool(BaseTool):
    name = "list_files"
    description = (
        "List files and directories under a workspace path (directories end with '/'). "
        "Recursive listings skip .git, node_modules, __pycache__ and virtualenvs."
    )
    read_only = True

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory relative to the workspace",
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
        except PathEscapeError as exc:
            return _error(self.name, str(exc))
        except FileNotFoundError as exc:
            return _error(self.name, str(exc) or f"no such directory: {path}")
        except Exception as exc:
            return _error(self.name, str(exc))
        if not entries:
            return "(empty)"
        return "\n".join(entries)


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read a text file under the workspace. Use offset/limit (1-based line numbers) "
        "to page through large files."
    )
    read_only = True

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path under the workspace"},
                "offset": {
                    "type": "integer",
                    "description": "First line to return (1-based)",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return",
                },
                "line_numbers": {
                    "type": "boolean",
                    "description": "Prefix each line with its number",
                    "default": False,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        sandbox: Any,
        path: str,
        offset: int = 1,
        limit: int | None = None,
        line_numbers: bool = False,
        **_: Any,
    ) -> str | ToolResult:
        try:
            if hasattr(sandbox, "read_bytes"):
                data = await sandbox.read_bytes(path, max_bytes=READ_MAX_BYTES + 1)
            else:  # duck-typed sandboxes that only implement read_text
                data = (await sandbox.read_text(path)).encode("utf-8")
        except FileNotFoundError:
            return _error(self.name, f"file not found: {path}")
        except IsADirectoryError:
            return _error(self.name, f"{path} is a directory; use list_files")
        except Exception as exc:
            return _error(self.name, str(exc))

        if b"\x00" in data[:8192]:
            return _error(
                self.name,
                f"{path} looks like a binary file; inspect it with run_command (e.g. xxd | head)",
            )
        clipped = len(data) > READ_MAX_BYTES
        text = data[:READ_MAX_BYTES].decode("utf-8", errors="replace")
        try:
            offset = max(1, int(offset or 1))
            limit = None if limit is None else max(1, int(limit))
        except (TypeError, ValueError):
            return _error(self.name, "offset and limit must be integers")

        lines = text.splitlines(keepends=True)
        paged = offset != 1 or limit is not None
        end = len(lines) if limit is None else min(len(lines), offset - 1 + limit)
        selected = lines[offset - 1 : end]
        if line_numbers:
            width = len(str(end or 1))
            body = "".join(f"{offset + i:>{width}}\t{ln}" for i, ln in enumerate(selected))
        else:
            body = "".join(selected)
        notes = []
        if paged:
            notes.append(f"[lines {offset}-{offset - 1 + len(selected)} of {len(lines)}]")
        if clipped:
            notes.append(f"[file larger than {READ_MAX_BYTES} bytes; showing the first part]")
        if notes:
            return body + ("\n" if body and not body.endswith("\n") else "") + " ".join(notes)
        return body


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Create or overwrite a file under the workspace (parent directories are created)."

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

    async def execute(self, sandbox: Any, path: str, content: str, **_: Any) -> str | ToolResult:
        if not isinstance(content, str):
            return _error(self.name, "content must be a string")
        try:
            await sandbox.write_text(path, content)
        except Exception as exc:
            return _error(self.name, str(exc))
        n_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return f"Wrote {len(content)} chars ({n_lines} lines) to {path}"


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Replace old_string with new_string in a workspace file. old_string must match "
        "exactly once unless replace_all is true."
    )

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
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
        replace_all: bool = False,
        **_: Any,
    ) -> str | ToolResult:
        if not old_string:
            return _error(self.name, "old_string must not be empty (use write_file to create)")
        if old_string == new_string:
            return _error(self.name, "old_string and new_string are identical")
        try:
            content = await sandbox.read_text(path)
        except FileNotFoundError:
            return _error(self.name, f"file not found: {path}")
        except Exception as exc:
            return _error(self.name, str(exc))

        count = content.count(old_string)
        if count == 0:
            return _error(self.name, f"old_string matched 0 times in {path!r}")
        if count > 1 and not replace_all:
            return _error(
                self.name,
                f"old_string matched {count} times in {path!r}; include more context "
                "so it matches exactly once, or set replace_all=true",
            )
        updated = content.replace(old_string, new_string, -1 if replace_all else 1)
        try:
            await sandbox.write_text(path, updated)
        except Exception as exc:
            return _error(self.name, str(exc))
        return f"Edited {path} ({count if replace_all else 1} replacement(s))"
