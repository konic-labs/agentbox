"""Runtime configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentbox.rewards import RewardSpec
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
    warm_pool: int = Field(
        default=0,
        ge=0,
        description=(
            "Pre-started containers kept ready per sandbox shape (image, limits, env); "
            "each is used by one rollout only. 0 disables the pool."
        ),
    )
    warm_pool_ttl_s: float = Field(
        default=300.0, gt=0, description="Idle pooled containers are removed after this long."
    )


class ModelConfig(BaseModel):
    """OpenAI Chat Completions client config (any compatible server)."""

    model: str
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = Field(
        default=None, description="Read the API key from this environment variable."
    )
    temperature: float | None = 0.7
    max_tokens: int | None = 4096
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    top_p: float | None = None
    seed: int | None = None
    stop: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    top_k: int | None = Field(default=None, description="vLLM/SGLang sampling (extra_body)")
    min_p: float | None = Field(default=None, description="vLLM/SGLang sampling (extra_body)")
    repetition_penalty: float | None = Field(default=None, description="vLLM (extra_body)")
    timeout_s: float = 120.0
    max_retries: int = Field(default=4, description="Retries for 429/5xx/timeouts/connection.")
    retry_backoff_s: float = 1.0
    retry_max_backoff_s: float = 30.0
    adaptive_max_tokens: bool = Field(
        default=True,
        description="When prompt+max_tokens exceeds the context, retry with the room left.",
    )
    min_completion_tokens: int = 128
    capture_tokens: bool = Field(
        default=False,
        description="Request token ids (vLLM return_token_ids) for exact RL training data.",
    )
    capture_logprobs: bool = Field(default=False, description="Request sampler logprobs.")
    top_logprobs: int | None = None
    record_raw: bool = Field(default=False, description="Keep the raw choice dict per turn.")
    strip_think: bool = Field(
        default=False, description="Move <think>...</think> from content into reasoning."
    )
    tool_choice: str | dict[str, Any] | None = "auto"
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Forwarded as the OpenAI parallel_tool_calls request field when set.",
    )
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    require_tool_calling: bool = Field(
        default=True,
        description="Fail fast when the server rejects tool definitions (native protocol).",
    )


class AgentConfig(BaseModel):
    """Agent loop and tool selection settings."""

    tools: ToolMode | list[str] = ToolMode.STRUCTURED
    custom_tools: list[Any] = Field(default_factory=list)
    builtins: bool = True
    system_prompt: str | None = None
    max_steps: int = 40
    step_timeout_s: float = 120.0
    episode_timeout_s: float = 900.0
    parallel_tool_calls: bool = Field(
        default=True,
        description=(
            "Run consecutive read-only tool calls of one turn concurrently. Calls that "
            "write or execute always run sequentially in emission order."
        ),
    )
    tool_timeout_s: float = Field(default=120.0, description="Upper bound for any tool call.")
    max_tool_output_chars: int = Field(
        default=16_000, description="Tool results shown to the model are truncated head+tail."
    )
    tool_protocol: Literal["native", "hermes"] = Field(
        default="native",
        description=(
            "native: OpenAI tool_calls. hermes: tools rendered into the system prompt and "
            "<tool_call> blocks parsed from content (servers without tool parsing)."
        ),
    )
    tool_response_role: Literal["tool", "user"] = Field(
        default="tool",
        description="hermes protocol only: send tool results as role=tool or wrapped user turns.",
    )
    drop_tools: list[str] = Field(default_factory=list)
    drop_tools_prob: float = 0.0
    include_thinking: bool = Field(
        default=True, description="Record model reasoning in the trajectory."
    )
    send_reasoning: bool = Field(
        default=False, description="Echo recorded reasoning back to the server on later turns."
    )
    # Optional process reward shaping applied after verifier
    step_penalty: float = 0.0
    timeout_penalty: float = 0.0
    max_steps_penalty: float = 0.0


class RolloutConfig(BaseModel):
    """Full configuration for a single rollout or parallel batch."""

    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    model: ModelConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    rewards: list[RewardSpec] | None = Field(
        default=None,
        description="Weighted reward components; default: verifier + AgentConfig penalties",
    )
    seed: int | None = None
    save_dir: Path | None = None
    trajectory_format: Literal["json", "jsonl", "art_dict"] = "json"
    run_id: str | None = None
