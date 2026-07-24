"""Probe OpenAI-compatible endpoints for chat and tool-calling support."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentbox.config import ModelConfig
from agentbox.errors import ModelError
from agentbox.model.openai_compat import OpenAICompatClient

ProbeKind = Literal["chat", "tools"]


class ProbeResult(BaseModel):
    ok: bool
    kind: ProbeKind
    model: str
    base_url: str | None = None
    message: str = ""
    hint: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


_TOOL_FLAG_HINT = (
    "vLLM: restart with --enable-auto-tool-choice and "
    "--tool-call-parser <parser> (e.g. hermes, qwen3_coder). "
    "Ollama: use a model with tool support. "
    "Other servers: enable OpenAI tools / function calling."
)

_MINIMAL_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def classify_tool_error(msg: str) -> str | None:
    """Return a short hint if the error looks like missing tool-calling config."""
    lower = msg.lower()
    if "enable-auto-tool-choice" in lower or "tool-call-parser" in lower:
        return _TOOL_FLAG_HINT
    if "tool choice" in lower and ("require" in lower or "not support" in lower):
        return _TOOL_FLAG_HINT
    if "does not appear to support tool calling" in lower:
        return _TOOL_FLAG_HINT
    if "tool" in lower and ("400" in lower or "bad request" in lower):
        return _TOOL_FLAG_HINT
    return None


async def probe_chat(config: ModelConfig) -> ProbeResult:
    """Send a minimal chat completion (no tools)."""
    client = OpenAICompatClient(config)
    try:
        resp = await client.complete(
            [{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=min(config.max_tokens or 32, 32),
            temperature=0,
        )
        content = (resp.content or "").strip()
        return ProbeResult(
            ok=True,
            kind="chat",
            model=config.model,
            base_url=config.base_url,
            message=f"chat ok (content={content[:40]!r})",
            details={"finish_reason": resp.finish_reason, "usage": resp.usage},
        )
    except ModelError as exc:
        return ProbeResult(
            ok=False,
            kind="chat",
            model=config.model,
            base_url=config.base_url,
            message=str(exc),
            hint="Check base_url, model id, and that the server is up.",
        )
    except Exception as exc:
        return ProbeResult(
            ok=False,
            kind="chat",
            model=config.model,
            base_url=config.base_url,
            message=str(exc),
            hint="Check base_url, model id, and that the server is up.",
        )


async def probe_tool_calling(config: ModelConfig) -> ProbeResult:
    """Send a minimal tools request with tool_choice=auto."""
    client = OpenAICompatClient(config)
    try:
        resp = await client.complete(
            [{"role": "user", "content": "Use the add tool to compute 2+2."}],
            tools=_MINIMAL_TOOL,
            tool_choice="auto",
            max_tokens=min(config.max_tokens or 128, 128),
            temperature=0,
        )
        n_tools = len(resp.tool_calls or [])
        # Server accepting tools is enough; model need not actually call the tool
        return ProbeResult(
            ok=True,
            kind="tools",
            model=config.model,
            base_url=config.base_url,
            message=(
                f"tool calling accepted (finish_reason={resp.finish_reason!r}, "
                f"tool_calls={n_tools})"
            ),
            details={
                "finish_reason": resp.finish_reason,
                "tool_call_names": [tc.function.name for tc in (resp.tool_calls or [])],
                "usage": resp.usage,
            },
        )
    except ModelError as exc:
        msg = str(exc)
        return ProbeResult(
            ok=False,
            kind="tools",
            model=config.model,
            base_url=config.base_url,
            message=msg,
            hint=_tools_fail_hint(msg),
        )
    except Exception as exc:
        msg = str(exc)
        return ProbeResult(
            ok=False,
            kind="tools",
            model=config.model,
            base_url=config.base_url,
            message=msg,
            hint=_tools_fail_hint(msg),
        )


def _tools_fail_hint(msg: str) -> str:
    classified = classify_tool_error(msg)
    if classified:
        return classified
    lower = msg.lower()
    if "connection" in lower or "connect" in lower or "refused" in lower:
        return "Check base_url, model id, and that the server is up."
    return _TOOL_FLAG_HINT


async def probe_endpoint(
    config: ModelConfig,
    *,
    require_tools: bool = True,
) -> list[ProbeResult]:
    """Run chat + optional tools probes; return list of results."""
    results = [await probe_chat(config)]
    if require_tools:
        results.append(await probe_tool_calling(config))
    return results


def format_probe_results(results: list[ProbeResult]) -> str:
    lines: list[str] = []
    for r in results:
        status = "ok" if r.ok else "FAIL"
        lines.append(f"  [{status}] {r.kind}: {r.message}")
        if r.hint and not r.ok:
            # collapse whitespace in multi-line hints
            hint = re.sub(r"\s+", " ", r.hint).strip()
            lines.append(f"         hint: {hint}")
    return "\n".join(lines)
