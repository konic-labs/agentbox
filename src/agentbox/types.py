"""Shared enums and type aliases."""

from __future__ import annotations

from enum import StrEnum


class ToolMode(StrEnum):
    """How tools are selected for an agent rollout."""

    STRUCTURED = "structured"  # file + shell builtins (stable set, benchmark default)
    EXTENDED = "extended"  # structured + search_files, apply_patch, submit
    SHELL = "shell"  # only run_command (+ optional custom merge)
    CUSTOM = "custom"  # no builtins unless listed; use custom_tools / name list


class FinalStatus(StrEnum):
    """Terminal status of a rollout."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    MAX_STEPS = "max_steps"


class StopReason(StrEnum):
    """Why the agent loop stopped."""

    FINAL_ANSWER = "final_answer"  # assistant replied without tool calls
    SUBMITTED = "submitted"  # a terminal tool (submit) was called
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    CONTEXT_LENGTH = "context_length"  # conversation exceeded the model context
    ERROR = "error"


class FailureKind(StrEnum):
    """Who is responsible for a non-successful rollout.

    Only ``none`` and ``policy`` outcomes are valid RL training signal; the other
    kinds are infrastructure noise and must not be scored as reward 0 (GH #40).
    """

    NONE = "none"
    POLICY = "policy"
    SANDBOX = "sandbox"
    SEED = "seed"
    SETUP_CHECK = "setup_check"
    MODEL_API = "model_api"
    VERIFIER = "verifier"
    INTERNAL = "internal"

    @property
    def is_infra(self) -> bool:
        return self not in (FailureKind.NONE, FailureKind.POLICY)


class VerifierType(StrEnum):
    """How success is measured after the agent finishes."""

    PYTEST = "pytest"
    COMMAND = "command"
    SHELL = "shell"  # Alias for command (used by teacher models)
    TEST = "test"  # Alias for pytest (used by teacher models)
    CUSTOM = "custom"


class MessageRole(StrEnum):
    """OpenAI chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
