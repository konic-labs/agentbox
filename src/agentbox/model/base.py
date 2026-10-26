"""Model client protocol and response types."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentbox.trajectory.schema import ToolCall


class ModelResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    reasoning_content: str | None = None
    usage: dict[str, int] | None = None
    raw_choice: dict[str, Any] | None = None
    # Token-level data (only when the server returns it, see ModelConfig.capture_*)
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    completion_logprobs: list[float] | None = None


@runtime_checkable
class ModelClient(Protocol):
    model: str

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        **kwargs: Any,
    ) -> ModelResponse: ...


async def aclose_model(client: Any) -> None:
    """Close a model client if it supports ``aclose`` (no-op otherwise)."""
    close = getattr(client, "aclose", None)
    if close is not None:
        await close()
