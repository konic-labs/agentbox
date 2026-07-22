"""Runtime configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentbox.types import ToolMode


class ResourceLimits(BaseModel):
    """Docker resource constraints for a sandbox."""

    cpu_count: float | None = 1.0
    memory_mb: int | None = 512
    pids_limit: int | None = 256
    network_disabled: bool = True
    memswap_mb: int | None = None


class SandboxConfig(BaseModel):
    """Container lifecycle and environment settings."""

    image: str = "python:3.12-slim-bookworm"
    workspace_dir: str = "/workspace"
    working_dir: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    limits: ResourceLimits = Field(default_factory=ResourceLimits)
    auto_pull: bool = True
    auto_remove: bool = True
    keep_on_failure: bool = False
    labels: dict[str, str] = Field(default_factory=dict)
    command: list[str] = Field(default_factory=lambda: ["sleep", "infinity"])
    ensure_pytest: bool = True


class ModelConfig(BaseModel):
    """OpenAI Chat Completions client config (any compatible server)."""

    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = 4096
    top_p: float | None = None
    timeout_s: float = 120.0
    tool_choice: str | dict[str, Any] | None = "auto"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    require_tool_calling: bool = True


class AgentConfig(BaseModel):
    """Agent loop and tool selection settings."""

    tools: ToolMode | list[str] = ToolMode.STRUCTURED
    custom_tools: list[Any] = Field(default_factory=list)
    builtins: bool = True
    system_prompt: str | None = None
    max_steps: int = 40
    step_timeout_s: float = 120.0
    episode_timeout_s: float = 900.0
    parallel_tool_calls: bool = True
    drop_tools: list[str] = Field(default_factory=list)
    drop_tools_prob: float = 0.0
    include_thinking: bool = True
    # Optional process reward shaping applied after verifier
    step_penalty: float = 0.0
    timeout_penalty: float = 0.0
    max_steps_penalty: float = 0.0


class RolloutConfig(BaseModel):
    """Full configuration for a single rollout or parallel batch."""

    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    model: ModelConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    seed: int | None = None
    save_dir: Path | None = None
    trajectory_format: Literal["json", "jsonl", "art_dict"] = "json"
    run_id: str | None = None
