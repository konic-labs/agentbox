"""@tool decorator for defining tools from (async or sync) functions."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import TypeAdapter

from agentbox.tools.base import BaseTool, ToolContext, ToolResult, get_tool_context

_RESERVED = {"sandbox", "ctx"}


def _docstring_args(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {arg: description})."""
    if not doc:
        return "", {}
    text = inspect.cleandoc(doc)
    parts = re.split(r"^\s*(Args|Arguments|Parameters):\s*$", text, maxsplit=1, flags=re.M)
    summary = parts[0].strip()
    args: dict[str, str] = {}
    if len(parts) == 3:
        current: str | None = None
        for line in parts[2].splitlines():
            if re.match(r"^\s*(Returns|Raises|Yields|Examples?):\s*$", line):
                break
            m = re.match(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
            if m:
                current = m.group(1)
                args[current] = m.group(2).strip()
            elif current and line.strip():
                args[current] += " " + line.strip()
    return summary, args


def _json_schema_for(annotation: Any) -> dict[str, Any]:
    try:
        schema = TypeAdapter(annotation).json_schema()
    except Exception:
        return {"type": "string"}
    schema.pop("title", None)
    return schema


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    sig = inspect.signature(fn)
    _, arg_docs = _docstring_args(fn.__doc__)
    properties: dict[str, Any] = {}
    required: list[str] = []
    defs: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if name in _RESERVED or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        prop = _json_schema_for(hints.get(name, str))
        defs.update(prop.pop("$defs", {}))
        if name in arg_docs and arg_docs[name]:
            prop["description"] = arg_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            try:
                TypeAdapter(type(param.default)).dump_json(param.default)
                prop.setdefault("default", param.default)
            except Exception:
                pass
        properties[name] = prop

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    if defs:
        schema["$defs"] = defs
    return schema


def tool(
    name: str | None = None,
    description: str | None = None,
    *,
    parameters: dict[str, Any] | None = None,
    read_only: bool = False,
    terminal: bool = False,
) -> Callable[[Callable[..., Any]], BaseTool]:
    """Wrap a function as a :class:`BaseTool`.

    The function receives the sandbox as its first argument and may declare a
    ``ctx: ToolContext`` parameter::

        @tool(description="Count lines")
        async def count_lines(sandbox, path: str) -> str: ...

    Sync functions run in a worker thread. Parameter descriptions are taken from
    a Google-style ``Args:`` docstring section.
    """

    def decorator(fn: Callable[..., Any]) -> BaseTool:
        tool_name = name or fn.__name__
        summary, _ = _docstring_args(fn.__doc__)
        tool_desc = description or summary or tool_name
        schema = parameters or _schema_from_signature(fn)
        wants_ctx = "ctx" in inspect.signature(fn).parameters
        is_async = inspect.iscoroutinefunction(fn)

        class FunctionTool(BaseTool):
            def parameters(self) -> dict[str, Any]:
                return schema

            async def execute(self, sandbox: Any, **kwargs: Any) -> str | ToolResult:
                if wants_ctx:
                    kwargs["ctx"] = get_tool_context() or ToolContext(sandbox=sandbox)
                if is_async:
                    result = await fn(sandbox, **kwargs)
                else:
                    result = await asyncio.to_thread(fn, sandbox, **kwargs)
                if isinstance(result, ToolResult) or isinstance(result, str):
                    return result
                return str(result)

        FunctionTool.name = tool_name
        FunctionTool.description = tool_desc
        FunctionTool.read_only = read_only
        FunctionTool.terminal = terminal
        FunctionTool.__name__ = f"Tool_{tool_name}"
        instance = FunctionTool()
        instance._fn = fn  # type: ignore[attr-defined]
        return instance

    return decorator
