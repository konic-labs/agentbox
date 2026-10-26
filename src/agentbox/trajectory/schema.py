"""Native trajectory and message schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agentbox.trajectory.tokens import TokenTrace, TurnTokens
from agentbox.types import FailureKind, FinalStatus, MessageRole


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
    # Per-turn bookkeeping (never sent to the model)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    latency_s: float | None = None
    tokens: TurnTokens | None = None

    def to_openai_dict(self, *, include_reasoning: bool = False) -> dict[str, Any]:
        """Convert to an OpenAI Chat Completions message dict.

        Reasoning is omitted by default: most chat templates drop it and strict
        servers reject the unknown field (GH #59).
        """
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
        if include_reasoning and self.reasoning_content is not None:
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
    truncated: bool = False


class TrajectoryMetrics(BaseModel):
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    model_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    duration_s: float = 0.0
    sandbox_create_s: float | None = None
    verify_s: float | None = None
    seed_s: float | None = None
    model_latency_s: float | None = None
    tool_latency_s: float | None = None
    attempts: int = 1  # rollout attempts including infra retries


class VerifyInfo(BaseModel):
    """Structured result of the task verifier (ground truth for reward)."""

    success: bool
    reward: float
    exit_code: int | None = None
    command: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    stdout: str = ""  # truncated head+tail
    stderr: str = ""
    tests_passed: int | None = None
    tests_failed: int | None = None
    tests_total: int | None = None
    tampered: list[str] = Field(default_factory=list)


_V1_VERIFY_KEYS = {
    "verify_success": "success",
    "verify_exit_code": "exit_code",
    "verify_command": "command",
    "verify_duration_s": "duration_s",
    "verify_stdout": "stdout",
    "verify_stderr": "stderr",
}

TRAJECTORY_SCHEMA_VERSION = 2


class Trajectory(BaseModel):
    """One rollout: conversation, tool records, reward and provenance.

    ``run_id`` identifies the batch/experiment, ``rollout_id`` the trajectory
    itself (unique; used for file and container names), ``group_id`` +
    ``sample_index`` the GRPO group membership.
    """

    schema_version: int = TRAJECTORY_SCHEMA_VERSION
    task_id: str
    run_id: str
    rollout_id: str = ""
    group_id: str | None = None
    sample_index: int = 0
    messages: list[Message]
    tool_call_records: list[ToolCallRecord] = Field(default_factory=list)
    reward: float = 0.0
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    final_status: FinalStatus
    stop_reason: str | None = None
    failure_kind: FailureKind = FailureKind.NONE
    verify: VerifyInfo | None = None
    metrics: TrajectoryMetrics = Field(default_factory=TrajectoryMetrics)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    model: str | None = None
    tool_mode: str | None = None
    tools: list[dict[str, Any]] | None = None
    task_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_v1(cls, data: Any) -> Any:
        """Accept v1 files (verify/stop/failure info lived in ``metadata``)."""
        if not isinstance(data, dict) or data.get("schema_version", 1) >= 2:
            return data
        data = dict(data)
        meta = dict(data.get("metadata") or {})
        if not data.get("rollout_id"):
            data["rollout_id"] = data.get("run_id", "")
        if "stop_reason" not in data and "stop_reason" in meta:
            data["stop_reason"] = meta.pop("stop_reason")
        if "failure_kind" not in data and "failure_kind" in meta:
            fk = meta.pop("failure_kind")
            data["failure_kind"] = fk if fk in {f.value for f in FailureKind} else "internal"
        if "verify" not in data and any(k in meta for k in _V1_VERIFY_KEYS):
            verify = {new: meta.pop(old) for old, new in _V1_VERIFY_KEYS.items() if old in meta}
            if verify.get("success") is None:
                verify["success"] = verify.get("exit_code") == 0
            verify.setdefault("reward", data.get("reward", 0.0))
            verify["stdout"] = verify.get("stdout") or ""
            verify["stderr"] = verify.get("stderr") or ""
            verify["command"] = verify.get("command") or ""
            data["verify"] = verify
        data["metadata"] = meta
        data["schema_version"] = TRAJECTORY_SCHEMA_VERSION
        return data

    @property
    def is_trainable(self) -> bool:
        """False for infrastructure failures (sandbox, seed, model API, ...)."""
        return not FailureKind(self.failure_kind).is_infra

    @property
    def success(self) -> bool:
        return self.final_status == FinalStatus.SUCCESS

    def token_trace(self) -> TokenTrace | None:
        """Exact training sequences when token ids were captured (else ``None``)."""
        from agentbox.trajectory.tokens import trace_from_messages

        return trace_from_messages(self.messages)

    def to_json(self, *, indent: int | None = None) -> str:
        return self.model_dump_json(indent=indent, exclude_none=True)

    def save(self, path: str | Path, *, indent: int | None = None) -> None:
        """Write atomically (compact JSON by default)."""
        from agentbox.utils.io import atomic_write_text

        atomic_write_text(path, self.to_json(indent=indent))

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_art_dict(self) -> dict[str, Any]:
        """Deprecated: use :func:`agentbox.integrations.art.to_art_dict`."""
        import warnings

        from agentbox.integrations.art import to_art_dict

        warnings.warn(
            "Trajectory.to_art_dict() is deprecated; use agentbox.integrations.art.to_art_dict",
            DeprecationWarning,
            stacklevel=2,
        )
        return to_art_dict(self)

    def to_art(self) -> Any:
        """Deprecated: use :func:`agentbox.integrations.art.to_art_trajectory`."""
        import warnings

        from agentbox.integrations.art import to_art_trajectory

        warnings.warn(
            "Trajectory.to_art() is deprecated; use agentbox.integrations.art.to_art_trajectory",
            DeprecationWarning,
            stacklevel=2,
        )
        return to_art_trajectory(self)
