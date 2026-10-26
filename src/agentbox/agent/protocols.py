"""Text tool-calling protocols for servers without native tool parsing.

``hermes`` is the format Qwen2.5/Qwen3, Hermes and many open models are trained
on: tool schemas inside ``<tools>`` in the system prompt, calls emitted as
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` in the content.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from agentbox.trajectory.schema import FunctionCall, ToolCall

HERMES_TOOLS_PREAMBLE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_OPEN_RE = re.compile(r"<tool_call>\s*(.*)$", re.S)


def render_hermes_tools(schemas: list[dict[str, Any]]) -> str:
    """Render OpenAI tool schemas as a Hermes/Qwen system-prompt section."""
    lines = "\n".join(json.dumps(s, ensure_ascii=False) for s in schemas)
    return HERMES_TOOLS_PREAMBLE.format(tools=lines)


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _to_call(blob: str) -> ToolCall:
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        # Surface the malformed call to the agent instead of dropping it.
        return ToolCall(id=_new_id(), function=FunctionCall(name="invalid_tool_call", arguments=blob))
    if not isinstance(data, dict):
        return ToolCall(id=_new_id(), function=FunctionCall(name="invalid_tool_call", arguments=blob))
    name = str(data.get("name") or data.get("function") or "invalid_tool_call")
    args = data.get("arguments", data.get("parameters", {}))
    arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return ToolCall(id=_new_id(), function=FunctionCall(name=name, arguments=arguments))


def parse_hermes_tool_calls(content: str | None) -> tuple[str | None, list[ToolCall]]:
    """Extract ``<tool_call>`` blocks; returns ``(remaining_text, calls)``.

    A trailing unterminated ``<tool_call>`` (generation cut off) is parsed too so
    the agent receives an error observation rather than a silent stop.
    """
    if not content or "<tool_call>" not in content:
        return content, []
    calls = [_to_call(m.group(1)) for m in _CALL_RE.finditer(content)]
    rest = _CALL_RE.sub("", content)
    tail = _OPEN_RE.search(rest)
    if tail:
        calls.append(_to_call(tail.group(1)))
        rest = rest[: tail.start()]
    rest = rest.strip()
    return (rest or None), calls


def hermes_tool_response(content: str) -> str:
    return f"<tool_response>\n{content}\n</tool_response>"
