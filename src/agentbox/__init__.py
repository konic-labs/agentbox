"""AgentBox — local agentic virtual environment for trajectory building."""

from agentbox.agent.agent import Agent
from agentbox.config import AgentConfig, ModelConfig, RolloutConfig, SandboxConfig
from agentbox.metrics import AggregateMetrics, aggregate_trajectories
from agentbox.runner.parallel import ParallelResult, ParallelRunner
from agentbox.runner.rollout import Rollout
from agentbox.sandbox.manager import SandboxManager
from agentbox.tasks.schema import Task, VerifierSpec
from agentbox.tools.base import BaseTool
from agentbox.tools.decorator import tool
from agentbox.trajectory.schema import Trajectory
from agentbox.version import __version__

__all__ = [
    "__version__",
    "Agent",
    "AgentConfig",
    "AggregateMetrics",
    "BaseTool",
    "ModelConfig",
    "ParallelResult",
    "ParallelRunner",
    "Rollout",
    "RolloutConfig",
    "SandboxConfig",
    "SandboxManager",
    "Task",
    "Trajectory",
    "VerifierSpec",
    "aggregate_trajectories",
    "tool",
]
