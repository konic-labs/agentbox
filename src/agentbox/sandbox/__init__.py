"""Docker sandbox lifecycle and I/O."""

from agentbox.sandbox.manager import SandboxManager
from agentbox.sandbox.prune import aprune_agentbox_containers, prune_agentbox_containers
from agentbox.sandbox.snapshot import commit_sandbox
from agentbox.sandbox.types import ExecResult, Sandbox

__all__ = [
    "SandboxManager",
    "Sandbox",
    "ExecResult",
    "prune_agentbox_containers",
    "aprune_agentbox_containers",
    "commit_sandbox",
]
