"""Docker sandbox lifecycle and I/O."""

from agentbox.sandbox.backend import DockerBackend, OutputCapture, get_backend
from agentbox.sandbox.manager import SandboxManager
from agentbox.sandbox.prune import aprune_agentbox_containers, prune_agentbox_containers
from agentbox.sandbox.snapshot import commit_sandbox
from agentbox.sandbox.types import ExecResult, ImageProfile, Sandbox

__all__ = [
    "DockerBackend",
    "ExecResult",
    "ImageProfile",
    "OutputCapture",
    "Sandbox",
    "SandboxManager",
    "aprune_agentbox_containers",
    "commit_sandbox",
    "get_backend",
    "prune_agentbox_containers",
]
