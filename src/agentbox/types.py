"""Shared enums and type aliases."""

from __future__ import annotations

from enum import Enum


class ToolMode(str, Enum):
    """How tools are selected for an agent rollout."""

    STRUCTURED = "structured"  # all 6 builtins (+ optional custom merge)
    SHELL = "shell"  # only run_command (+ optional custom merge)
    CUSTOM = "custom"  # no builtins unless listed; use custom_tools / name list


class FinalStatus(str, Enum):
    """Terminal status of a rollout."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    MAX_STEPS = "max_steps"


class VerifierType(str, Enum):
    """How success is measured after the agent finishes."""

    PYTEST = "pytest"
    COMMAND = "command"
    SHELL = "shell"  # Alias for command (used by teacher models)
    TEST = "test"  # Alias for pytest (used by teacher models)
    CUSTOM = "custom"


class MessageRole(str, Enum):
    """OpenAI chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
