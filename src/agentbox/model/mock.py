"""Deterministic mock model for tests."""

from __future__ import annotations

from typing import Any

from agentbox.model.base import ModelResponse


class MockModelClient:
    """Scripted OpenAI-style responses for unit tests."""

    def __init__(self, script: list[ModelResponse], *, model: str = "mock") -> None:
        self.model = model
        self._script = list(script)
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                **kwargs,
            }
        )
        if self._idx >= len(self._script):
            return ModelResponse(
                content="(mock exhausted script)",
                tool_calls=[],
                finish_reason="stop",
            )
        resp = self._script[self._idx]
        self._idx += 1
        return resp
