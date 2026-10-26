"""AgentBox — Docker sandboxes, tools, rewards and trajectories for RL post-training.

Trainer-agnostic: drive rollouts yourself (:class:`Rollout`,
:class:`ParallelRunner`), let a trainer drive a step-wise :class:`AgentBoxEnv`, or
plug into TRL / verl / OpenRLHF / SkyRL / ART via :mod:`agentbox.integrations`.
"""

from agentbox.agent.agent import Agent
from agentbox.agent.loop import AgentHooks
from agentbox.benchmark import (
    BenchmarkReport,
    BenchmarkRunConfig,
    BenchmarkRunner,
    BenchmarkSuite,
    ModelUnderTest,
    compare_reports,
    create_suite_from_tasks,
    load_suite,
)
from agentbox.config import (
    AgentConfig,
    ModelConfig,
    ResourceLimits,
    RolloutConfig,
    SandboxConfig,
    SecurityConfig,
)
from agentbox.env import AgentBoxEnv, SyncAgentBoxEnv
from agentbox.export import ExportFilter, export
from agentbox.metrics import AggregateMetrics, MetricsAccumulator, aggregate_trajectories
from agentbox.rewards import RewardSpec, register_reward
from agentbox.runner.parallel import ParallelResult, ParallelRunner, RunSummary
from agentbox.runner.rollout import Rollout
from agentbox.sandbox.manager import SandboxManager
from agentbox.tasks.loader import iter_tasks, load_task
from agentbox.tasks.schema import EnvironmentSpec, Task, VerifierSpec
from agentbox.tools.base import BaseTool
from agentbox.tools.decorator import tool
from agentbox.trajectory.io import JsonlSink, iter_trajectories
from agentbox.trajectory.schema import Trajectory
from agentbox.types import FailureKind, FinalStatus, StopReason, ToolMode
from agentbox.version import __version__

__all__ = [
    "Agent",
    "AgentBoxEnv",
    "AgentConfig",
    "AgentHooks",
    "AggregateMetrics",
    "BaseTool",
    "BenchmarkReport",
    "BenchmarkRunConfig",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "EnvironmentSpec",
    "ExportFilter",
    "FailureKind",
    "FinalStatus",
    "JsonlSink",
    "MetricsAccumulator",
    "ModelConfig",
    "ModelUnderTest",
    "ParallelResult",
    "ParallelRunner",
    "ResourceLimits",
    "RewardSpec",
    "Rollout",
    "RolloutConfig",
    "RunSummary",
    "SandboxConfig",
    "SandboxManager",
    "SecurityConfig",
    "StopReason",
    "SyncAgentBoxEnv",
    "Task",
    "ToolMode",
    "Trajectory",
    "VerifierSpec",
    "__version__",
    "aggregate_trajectories",
    "compare_reports",
    "create_suite_from_tasks",
    "export",
    "iter_tasks",
    "iter_trajectories",
    "load_suite",
    "load_task",
    "register_reward",
    "tool",
]
