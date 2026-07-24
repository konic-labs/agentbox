"""Unit tests for model endpoint probes (no live network)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentbox.config import ModelConfig
from agentbox.errors import ModelError
from agentbox.model.base import ModelResponse
from agentbox.model.probe import (
    classify_tool_error,
    format_probe_results,
    probe_chat,
    probe_tool_calling,
)
from agentbox.trajectory.schema import FunctionCall, ToolCall


def test_classify_tool_error_vllm() -> None:
    msg = (
        "Error code: 400 - \"auto\" tool choice requires "
        "--enable-auto-tool-choice and --tool-call-parser to be set"
    )
    hint = classify_tool_error(msg)
    assert hint is not None
    assert "enable-auto-tool-choice" in hint


def test_classify_tool_error_generic() -> None:
    assert classify_tool_error("connection refused") is None


@pytest.mark.asyncio
async def test_probe_chat_ok() -> None:
    cfg = ModelConfig(model="m", base_url="http://x", api_key="k")
    mock_resp = ModelResponse(content="pong", tool_calls=[], finish_reason="stop")
    with patch(
        "agentbox.model.probe.OpenAICompatClient.complete",
        new=AsyncMock(return_value=mock_resp),
    ):
        result = await probe_chat(cfg)
    assert result.ok
    assert result.kind == "chat"


@pytest.mark.asyncio
async def test_probe_tools_ok() -> None:
    cfg = ModelConfig(model="m", base_url="http://x", api_key="k")
    mock_resp = ModelResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id="t1",
                function=FunctionCall(name="add", arguments='{"a":2,"b":2}'),
            )
        ],
        finish_reason="tool_calls",
    )
    with patch(
        "agentbox.model.probe.OpenAICompatClient.complete",
        new=AsyncMock(return_value=mock_resp),
    ):
        result = await probe_tool_calling(cfg)
    assert result.ok
    assert result.kind == "tools"
    assert result.details.get("tool_call_names") == ["add"]


@pytest.mark.asyncio
async def test_probe_tools_fail_hint() -> None:
    cfg = ModelConfig(model="m", base_url="http://x", api_key="k")
    err = ModelError(
        "Model/server does not appear to support tool calling. Details: "
        "Error code: 400 - \"auto\" tool choice requires --enable-auto-tool-choice"
    )
    with patch(
        "agentbox.model.probe.OpenAICompatClient.complete",
        new=AsyncMock(side_effect=err),
    ):
        result = await probe_tool_calling(cfg)
    assert not result.ok
    assert result.hint is not None
    assert "tool-call-parser" in result.hint


def test_format_probe_results() -> None:
    from agentbox.model.probe import ProbeResult

    text = format_probe_results(
        [
            ProbeResult(ok=True, kind="chat", model="m", message="ok"),
            ProbeResult(
                ok=False, kind="tools", model="m", message="fail", hint="do X"
            ),
        ]
    )
    assert "[ok] chat" in text
    assert "[FAIL] tools" in text
    assert "hint: do X" in text
