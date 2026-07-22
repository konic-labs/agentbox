"""AgentBox exception hierarchy."""

from __future__ import annotations


class AgentBoxError(Exception):
    """Base error for all AgentBox failures."""


class DockerNotAvailableError(AgentBoxError):
    """Docker daemon is not reachable."""


class ImageNotFoundError(AgentBoxError):
    """Required container image is missing and could not be pulled."""


class SandboxError(AgentBoxError):
    """Sandbox / container operation failed."""


class SandboxExecError(SandboxError):
    """Command execution inside a sandbox failed unexpectedly."""


class PathEscapeError(SandboxError):
    """A path attempted to escape the workspace jail."""


class ToolError(AgentBoxError):
    """Internal tool error (usually converted to tool-result text)."""


class ModelError(AgentBoxError):
    """LLM / model client error."""


class TaskValidationError(AgentBoxError):
    """Task definition is invalid."""


class TaskSeedError(AgentBoxError):
    """Failed to seed a task into a sandbox (files or setup_commands)."""


class VerifierError(AgentBoxError):
    """Verifier could not be run."""


class RolloutTimeoutError(AgentBoxError):
    """Episode or step timeout exceeded."""


class TaskGenerationError(AgentBoxError):
    """Automated task generation failed after retries."""


class BenchmarkError(AgentBoxError):
    """Benchmark suite or run failed."""


class SuiteIntegrityError(BenchmarkError):
    """Suite content hash does not match frozen snapshot."""
