"""Replay recorded assistant turns (trajectory replay, deterministic tests)."""

from __future__ import annotations

from typing import Any

from agentbox.model.base import ModelResponse


class ReplayModelClient:
    """Return the assistant turns of a recorded trajectory, in order.

    The turn is chosen by the number of assistant messages already in the
    request, so replay is deterministic and safe to share across rollouts.
    """

    def __init__(self, source: Any, *, model: str | None = None) -> None:
        messages = getattr(source, "messages", source)
        self._turns = [
            m
            for m in messages
            if (m.role.value if hasattr(m.role, "value") else m.role) == "assistant"
        ]
        self.model = model or getattr(source, "model", None) or "replay"

    def __len__(self) -> int:
        return len(self._turns)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,  # noqa: ARG002
        tool_choice: str | dict[str, Any] | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ModelResponse:
        idx = sum(1 for m in messages if m.get("role") == "assistant")
        if idx >= len(self._turns):
            return ModelResponse(content="(replay exhausted)", finish_reason="stop")
        turn = self._turns[idx]
        return ModelResponse(
            content=turn.content,
            tool_calls=[tc.model_copy() for tc in turn.tool_calls or []],
            finish_reason=turn.finish_reason or ("tool_calls" if turn.tool_calls else "stop"),
            reasoning_content=turn.reasoning_content,
        )

    async def aclose(self) -> None:
        return None
