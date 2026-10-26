"""Deterministic mock model for tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from agentbox.model.base import ModelResponse

_EXHAUSTED = ModelResponse(content="(mock exhausted script)", tool_calls=[], finish_reason="stop")


class MockModelClient:
    """Scripted OpenAI-style responses for tests and hermetic pipelines.

    ``mode="conversation"`` (default) picks the response by the number of
    assistant turns already in the request, so one shared mock behaves
    identically for any number of concurrent rollouts (GH #48).
    ``mode="cursor"`` advances a single global cursor (legacy behaviour).
    """

    def __init__(
        self,
        script: list[ModelResponse],
        *,
        model: str = "mock",
        mode: Literal["conversation", "cursor"] = "conversation",
        record_calls: bool = True,
    ) -> None:
        self.model = model
        self.mode = mode
        self._script = list(script)
        self._idx = 0
        self.record_calls = record_calls
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        **kwargs: Any,
    ) -> ModelResponse:
        if self.record_calls:
            self.calls.append(
                {
                    "messages": list(messages),
                    "tools": tools,
                    "tool_choice": tool_choice,
                    **kwargs,
                }
            )
        if self.mode == "conversation":
            idx = sum(1 for m in messages if m.get("role") == "assistant")
        else:
            idx = self._idx
            self._idx += 1
        if idx >= len(self._script):
            return _EXHAUSTED
        return self._script[idx].model_copy(deep=True)

    async def aclose(self) -> None:
        return None


class FunctionModelClient:
    """Wrap a Python policy ``(messages, tools) -> ModelResponse`` (sync or async)."""

    def __init__(
        self,
        fn: Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], Any],
        *,
        model: str = "function",
    ) -> None:
        self.fn = fn
        self.model = model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ModelResponse:
        result = self.fn(messages, tools)
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, ModelResponse):
            return result
        if isinstance(result, str):
            return ModelResponse(content=result, finish_reason="stop")
        raise TypeError(f"policy must return ModelResponse or str, got {type(result)!r}")

    async def aclose(self) -> None:
        return None
