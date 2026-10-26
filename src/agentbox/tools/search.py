"""search_files: grep the workspace without a shell (argv only)."""

from __future__ import annotations

from typing import Any

from agentbox.sandbox.paths import resolve_workspace_path
from agentbox.tools.base import BaseTool, ToolResult

_EXCLUDE_DIRS = (".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache")
_MAX_LINE_CHARS = 300


class SearchFilesTool(BaseTool):
    name = "search_files"
    description = (
        "Search file contents under the workspace with an extended regular expression. "
        "Returns matching lines as path:line:text."
    )
    read_only = True

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Extended regex (grep -E)"},
                "path": {"type": "string", "default": ".", "description": "Directory or file"},
                "glob": {"type": "string", "description": "Only files matching, e.g. '*.py'"},
                "ignore_case": {"type": "boolean", "default": False},
                "max_results": {"type": "integer", "default": 100},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        sandbox: Any,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignore_case: bool = False,
        max_results: int = 100,
        **_: Any,
    ) -> str | ToolResult:
        if not pattern:
            return ToolResult(content="ERROR: search_files failed: empty pattern", is_error=True)
        try:
            target = resolve_workspace_path(sandbox.workspace_dir, path)
        except Exception as exc:
            return ToolResult(content=f"ERROR: search_files failed: {exc}", is_error=True)
        try:
            limit = max(1, min(int(max_results), 1000))
        except (TypeError, ValueError):
            limit = 100
        argv = ["grep", "-rnIE"]
        if ignore_case:
            argv.append("-i")
        argv += [f"--exclude-dir={d}" for d in _EXCLUDE_DIRS]
        if glob:
            argv.append(f"--include={glob}")
        argv += ["-e", pattern, "--", target]
        result = await sandbox.exec(argv, timeout_s=30.0)
        if result.exit_code == 1:
            return "(no matches)"
        if result.exit_code != 0:
            msg = (result.stderr or "grep failed").strip()
            return ToolResult(content=f"ERROR: search_files failed: {msg}", is_error=True)
        prefix = sandbox.workspace_dir.rstrip("/") + "/"
        lines: list[str] = []
        for ln in result.stdout.splitlines():
            if ln.startswith(prefix):
                ln = ln[len(prefix) :]
            if len(ln) > _MAX_LINE_CHARS:
                ln = ln[:_MAX_LINE_CHARS] + "…"
            lines.append(ln)
            if len(lines) >= limit:
                break
        total = len(result.stdout.splitlines())
        more = "" if total <= limit else f"\n... ({total - limit} more matches; refine the pattern)"
        return "\n".join(lines) + more
