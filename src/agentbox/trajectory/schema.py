"""Native trajectory and message schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentbox.types import FinalStatus, MessageRole


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string (OpenAI style)


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: MessageRole | str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None
    raw: dict[str, Any] | None = None

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to an OpenAI Chat Completions message dict."""
        role = self.role.value if isinstance(self.role, MessageRole) else self.role
        msg: dict[str, Any] = {"role": role}
        if self.content is not None:
            msg["content"] = self.content
        elif role == "assistant" and self.tool_calls:
            msg["content"] = None
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            msg["name"] = self.name
        if self.reasoning_content is not None:
            msg["reasoning_content"] = self.reasoning_content
        return msg


class ToolCallRecord(BaseModel):
    step: int
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool = False
    duration_s: float = 0.0


class TrajectoryMetrics(BaseModel):
    steps: int = 0
    tool_calls: int = 0
    model_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    duration_s: float = 0.0
    sandbox_create_s: float | None = None
    verify_s: float | None = None
    seed_s: float | None = None


class Trajectory(BaseModel):
    task_id: str
    run_id: str
    messages: list[Message]
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
    reward: float = 0.0
    final_status: FinalStatus
    metrics: TrajectoryMetrics = Field(default_factory=TrajectoryMetrics)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    model: str | None = None
    tool_mode: str | None = None
    tools: list[dict[str, Any]] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_art_dict(self) -> dict[str, Any]:
        from agentbox.trajectory.formats.art import to_art_dict

        return to_art_dict(self)

    def to_art(self) -> Any:
        from agentbox.trajectory.formats.art import to_art

        return to_art(self)
