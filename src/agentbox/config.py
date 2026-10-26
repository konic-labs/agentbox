"""Runtime configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentbox.types import ToolMode

# Capabilities kept after ``cap_drop=["ALL"]``: enough for package managers
# (apt drops to _apt, pip writes as root) and ordinary process control.
DEFAULT_CAP_ADD: tuple[str, ...] = (
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "FSETID",
    "KILL",
    "SETGID",
    "SETUID",
    "NET_BIND_SERVICE",
)


class ResourceLimits(BaseModel):
    """Docker resource constraints for a sandbox."""

    cpu_count: float | None = 1.0
    memory_mb: int | None = 512
    pids_limit: int | None = 256
    network_disabled: bool = True
    memswap_mb: int | None = None
    shm_size_mb: int | None = None
    nofile: int | None = None
    read_only_rootfs: bool = False
    tmpfs: dict[str, str] = Field(
        default_factory=dict,
        description="Mount path -> tmpfs options, e.g. {'/tmp': 'size=256m'}",
    )


class SecurityConfig(BaseModel):
    """Container hardening applied to every sandbox."""

    cap_drop: list[str] = Field(default_factory=lambda: ["ALL"])
    cap_add: list[str] = Field(default_factory=lambda: list(DEFAULT_CAP_ADD))
    no_new_privileges: bool = True
    init: bool = True
    user: str | None = None
    runtime: str | None = Field(default=None, description="OCI runtime, e.g. 'runsc' (gVisor)")


class SandboxConfig(BaseModel):
    """Container lifecycle and environment settings."""

    image: str = "python:3.12-slim-bookworm"
    workspace_dir: str = "/workspace"
    working_dir: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    limits: ResourceLimits = Field(default_factory=ResourceLimits)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    auto_pull: bool = True
    auto_remove: bool = Field(
        default=True,
        description="Remove sandboxes after each rollout (False keeps every sandbox).",
    )
    keep_on_failure: bool = Field(
        default=False,
        description="Keep the sandbox when a rollout does not succeed (for debugging).",
    )
    labels: dict[str, str] = Field(default_factory=dict)
    command: list[str] = Field(default_factory=lambda: ["sleep", "infinity"])
    ensure_pytest: bool = Field(
        default=True,
        description=(
            "Guarantee pytest is importable. Missing pytest is installed once per image "
            "into a cached derived image (not per container)."
        ),
    )
    shell: list[str] | None = Field(
        default=None,
        description="argv prefix for string commands; default: bash -c if present, else sh -c",
    )
    max_output_bytes: int = Field(
        default=256 * 1024,
        description="Bytes of stdout/stderr kept per stream (head + tail) for each command.",
    )
    kill_output_bytes: int | None = Field(
        default=64 * 1024 * 1024,
        description="Kill a command once it has produced this many bytes of output.",
    )
    prepare_network: bool = Field(
        default=True,
        description="Allow network while building cached images (pytest / setup snapshots).",
    )
    cache_setup: bool = Field(
        default=True,
        description="Snapshot task setup_commands into cached images reused by later rollouts.",
    )
    cache_max_images: int = Field(default=64, description="LRU bound on cached setup images.")


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
