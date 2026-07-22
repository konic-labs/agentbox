"""@tool decorator for defining tools from async functions."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints

from agentbox.tools.base import BaseTool, ToolResult


_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "sandbox":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        ann = hints.get(name, str)
        origin = get_origin(ann)
        if origin is type(None):
            json_type = "string"
        elif origin is not None:
            # Optional[X] / X | None
            args = [a for a in get_args(ann) if a is not type(None)]
            json_type = _TYPE_MAP.get(args[0], "string") if args else "string"
        else:
            json_type = _TYPE_MAP.get(ann, "string")

        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def tool(
    name: str | None = None,
    description: str | None = None,
    *,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], BaseTool]:
    """Wrap an async function as a BaseTool.

    Signature must be::

        async def fn(sandbox, <param>: <type>, ...) -> str
    """

    def decorator(fn: Callable[..., Any]) -> BaseTool:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or tool_name).strip()
        schema = parameters or _schema_from_signature(fn)

        class FunctionTool(BaseTool):
            def parameters(self) -> dict[str, Any]:
                return schema

            async def execute(self, sandbox: Any, **kwargs: Any) -> str | ToolResult:
                return await fn(sandbox, **kwargs)

        FunctionTool.name = tool_name
        FunctionTool.description = tool_desc
        FunctionTool.__name__ = f"Tool_{tool_name}"
        instance = FunctionTool()
        # preserve original for debugging
        instance._fn = fn  # type: ignore[attr-defined]
        return instance

    return decorator
