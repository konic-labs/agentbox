"""Runtime configuration models."""

from __future__ import annotations

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

    cpu_count: float | None = Field(default=1.0, description="CPUs (fractional; None: no limit).")
    memory_mb: int | None = Field(default=512, description="Memory limit in MiB (None: no limit).")
    pids_limit: int | None = Field(
        default=256, description="Maximum processes in the container (fork-bomb guard)."
    )
    network_disabled: bool = Field(
        default=True, description="No network inside sandboxes (network_mode=none)."
    )
    memswap_mb: int | None = Field(
        default=None, description="Memory + swap limit in MiB; default memory_mb (no swap)."
    )
    shm_size_mb: int | None = Field(default=None, description="/dev/shm size in MiB.")
    nofile: int | None = Field(default=None, description="Open-file ulimit (soft = hard).")
    read_only_rootfs: bool = Field(
        default=False, description="Mount the root filesystem read-only (pair with tmpfs)."
    )
    tmpfs: dict[str, str] = Field(
        default_factory=dict,
        description="Mount path -> tmpfs options, e.g. {'/tmp': 'size=256m'}",
    )


class SecurityConfig(BaseModel):
    """Container hardening applied to every sandbox."""

    cap_drop: list[str] = Field(
        default_factory=lambda: ["ALL"], description="Linux capabilities dropped."
    )
    cap_add: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CAP_ADD),
        description="Capabilities added back after cap_drop (package managers, kill).",
    )
    no_new_privileges: bool = Field(
        default=True, description="Block privilege escalation through setuid binaries."
    )
    init: bool = Field(
        default=True, description="Run an init process as PID 1 (reaps zombies, forwards signals)."
    )
    user: str | None = Field(
        default=None, description="Run commands as this user, e.g. '1000:1000' (default: image's)."
    )
    runtime: str | None = Field(default=None, description="OCI runtime, e.g. 'runsc' (gVisor)")


class SandboxConfig(BaseModel):
    """Container lifecycle and environment settings."""

    image: str = Field(
        default="python:3.12-slim-bookworm", description="Image sandboxes start from."
    )
    workspace_dir: str = Field(
        default="/workspace",
        description="Directory holding the task files; tool paths are relative to it.",
    )
    working_dir: str = Field(default="/workspace", description="Default cwd for commands.")
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables set in every sandbox."
    )
    limits: ResourceLimits = Field(
        default_factory=ResourceLimits, description="CPU, memory, process and network limits."
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig, description="Capabilities, privileges, user, runtime."
    )
    auto_pull: bool = Field(default=True, description="Pull the image when it is missing.")
    auto_remove: bool = Field(
        default=True,
        description="Remove sandboxes after each rollout (False keeps every sandbox).",
    )
    keep_on_failure: bool = Field(
        default=False,
        description="Keep the sandbox when a rollout does not succeed (for debugging).",
    )
    labels: dict[str, str] = Field(
        default_factory=dict, description="Extra Docker labels on every sandbox."
    )
    command: list[str] = Field(
        default_factory=lambda: ["sleep", "infinity"],
        description="Container main process; it only has to keep the container alive.",
    )
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

    model: str = Field(description="Model id sent to the server.")
    base_url: str | None = Field(
        default=None, description="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1."
    )
    api_key: str | None = Field(
        default=None, description="API key (prefer api_key_env; keys never reach trajectories)."
    )
    api_key_env: str | None = Field(
        default=None, description="Read the API key from this environment variable."
    )
    temperature: float | None = Field(default=0.7, description="None: server default.")
    max_tokens: int | None = Field(default=4096, description="Completion token budget per turn.")
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = Field(
        default="max_tokens",
        description="Request field name (OpenAI reasoning models: max_completion_tokens).",
    )
    top_p: float | None = Field(default=None, description="Nucleus sampling.")
    seed: int | None = Field(
        default=None, description="Sampling seed (per-sample seeds derive from RolloutConfig.seed)."
    )
    stop: list[str] | None = Field(default=None, description="Stop sequences.")
    presence_penalty: float | None = Field(default=None, description="OpenAI presence penalty.")
    frequency_penalty: float | None = Field(default=None, description="OpenAI frequency penalty.")
    top_k: int | None = Field(default=None, description="vLLM/SGLang sampling (extra_body)")
    min_p: float | None = Field(default=None, description="vLLM/SGLang sampling (extra_body)")
    repetition_penalty: float | None = Field(default=None, description="vLLM (extra_body)")
    timeout_s: float = Field(default=120.0, description="Per-request timeout.")
    max_retries: int = Field(default=4, description="Retries for 429/5xx/timeouts/connection.")
    retry_backoff_s: float = Field(
        default=1.0, description="Exponential backoff base (full jitter; honours Retry-After)."
    )
    retry_max_backoff_s: float = Field(default=30.0, description="Cap on a single backoff.")
    adaptive_max_tokens: bool = Field(
        default=True,
        description="When prompt+max_tokens exceeds the context, retry with the room left.",
    )
    min_completion_tokens: int = Field(
        default=128,
        description="Below this much room the context counts as exhausted (policy failure).",
    )
    capture_tokens: bool = Field(
        default=False,
        description="Request token ids (vLLM return_token_ids) for exact RL training data.",
    )
    capture_logprobs: bool = Field(default=False, description="Request sampler logprobs.")
    top_logprobs: int | None = Field(
        default=None, description="Alternatives per token when capturing logprobs."
    )
    record_raw: bool = Field(default=False, description="Keep the raw choice dict per turn.")
    strip_think: bool = Field(
        default=False, description="Move <think>...</think> from content into reasoning."
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default="auto", description="OpenAI tool_choice (None: omit the field)."
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Forwarded as the OpenAI parallel_tool_calls request field when set.",
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict, description="Extra HTTP headers on every request."
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description="Merged into the request body, e.g. {'chat_template_kwargs': {...}}.",
    )
    require_tool_calling: bool = Field(
        default=True,
        description="Fail fast when the server rejects tool definitions (native protocol).",
    )


class AgentConfig(BaseModel):
    """Agent loop and tool selection settings."""

    tools: ToolMode | list[str] = Field(
        default=ToolMode.STRUCTURED,
        description="Tool mode (structured|extended|shell|custom) or a list of tool names.",
    )
    custom_tools: list[Any] = Field(
        default_factory=list, description="BaseTool instances or @tool functions to add."
    )
    builtins: bool = Field(default=True, description="Include the builtin tools of the mode.")
    system_prompt: str | None = Field(
        default=None, description="Replace the system prompt ({workspace_dir} is substituted)."
    )
    max_steps: int = Field(default=40, description="Maximum model turns per episode.")
    step_timeout_s: float = Field(
        default=120.0, description="Budget for one turn (model call, then its tool calls)."
    )
    episode_timeout_s: float = Field(
        default=900.0, description="Wall-clock budget for the whole episode."
    )
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
    drop_tools: list[str] = Field(default_factory=list, description="Tool names to remove.")
    drop_tools_prob: float = Field(
        default=0.0,
        description="Drop each tool with this probability per rollout (at least one stays).",
    )
    include_thinking: bool = Field(
        default=True, description="Record model reasoning in the trajectory."
    )
    send_reasoning: bool = Field(
        default=False, description="Echo recorded reasoning back to the server on later turns."
    )
    # Optional process reward shaping applied after verifier
    step_penalty: float = Field(default=0.0, description="Reward subtracted per step.")
    timeout_penalty: float = Field(default=0.0, description="Reward subtracted on timeout.")
    max_steps_penalty: float = Field(
        default=0.0, description="Reward subtracted when max_steps is reached."
    )


class RolloutConfig(BaseModel):
    """Full configuration for a single rollout or parallel batch."""

    sandbox: SandboxConfig = Field(default_factory=SandboxConfig, description="Sandbox settings.")
    model: ModelConfig = Field(description="Policy model endpoint and sampling.")
    agent: AgentConfig = Field(default_factory=AgentConfig, description="Agent loop and tools.")
    rewards: list[RewardSpec] | None = Field(
        default=None,
        description="Weighted reward components; default: verifier + AgentConfig penalties",
    )
    seed: int | None = Field(
        default=None,
        description="Run seed (per-sample sampling seeds and tool dropout derive from it).",
    )
    run_id: str | None = Field(default=None, description="Run id (default: random per runner).")
