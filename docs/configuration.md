# Configuration

All primary knobs are Pydantic v2 models in `agentbox.config`.

## Project file (`agentbox.yaml`)

Optional repo-root config loaded via `agentbox.config_load.load_project_config()`.

```yaml
teacher:
  model: Qwen/…
  base_url: http://localhost:8000/v1
  api_key: EMPTY
  max_tokens: 8192
generate:
  concurrency: 8
  target: 20
  llm_judge: true
  min_score: 0.65
students:
  - id: local-small
    model: qwen3.5:0.8b
    base_url: http://localhost:11434/v1
```

- Env interpolation: `${VAR}`
- Discovery: `AGENTBOX_CONFIG` or walk from CWD for `agentbox.yaml` / `.yml` / `.toml` / `.json`
- Example: `examples/agentbox.yaml`

Used by `agentbox generate` (teacher defaults) and `agentbox bench run` (students when no model flags).

## ResourceLimits

| Field | Default | Meaning |
| --- | --- | --- |
| `cpu_count` | `1.0` | Docker `nano_cpus = cpu_count * 1e9` |
| `memory_mb` | `512` | Memory hard limit |
| `pids_limit` | `256` | Process count cap |
| `network_disabled` | `True` | No container network (`network_mode=none`) |
| `memswap_mb` | `None` | Defaults to `memory_mb` (no extra swap) |

## SandboxConfig

| Field | Default | Meaning |
| --- | --- | --- |
| `image` | `python:3.12-slim-bookworm` | Docker image |
| `workspace_dir` | `/workspace` | Jail root for files |
| `working_dir` | `/workspace` | `docker exec` workdir |
| `env` | `{}` | Extra environment variables |
| `limits` | `ResourceLimits()` | Resource / network policy |
| `auto_pull` | `True` | Pull image if missing |
| `auto_remove` | `True` | Remove container on destroy |
| `keep_on_failure` | `False` | Leave container for debugging |
| `labels` | `{}` | Extra Docker labels |
| `command` | `["sleep", "infinity"]` | Keep container alive for exec |
| `ensure_pytest` | `True` | Try to install pytest if missing |

Presets: `agentbox.sandbox.images.sandbox_config_for_preset("python"|"node"|"go"|"baked")`.

## ModelConfig

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | *(required)* | Model id for the server |
| `base_url` | `None` | OpenAI-compatible endpoint |
| `api_key` | `None` | Optional; env `OPENAI_API_KEY` fallback; local often `"EMPTY"` |
| `temperature` | `0.7` | Sampling temperature |
| `max_tokens` | `4096` | Completion cap |
| `top_p` | `None` | Optional nucleus sampling |
| `timeout_s` | `120.0` | HTTP timeout |
| `tool_choice` | `"auto"` | OpenAI tool_choice |
| `extra_headers` | `{}` | Passed to client |
| `extra_body` | `{}` | Provider-specific body fields |
| `require_tool_calling` | `True` | Prefer clear errors when tools unsupported |

## AgentConfig

| Field | Default | Meaning |
| --- | --- | --- |
| `tools` | `ToolMode.STRUCTURED` | `structured` / `shell` / `custom` / name list |
| `custom_tools` | `[]` | Extra `BaseTool` instances (override by name) |
| `builtins` | `True` | Include builtin set for mode |
| `system_prompt` | `None` | Override default system prompt |
| `max_steps` | `40` | Max tool-using turns |
| `step_timeout_s` | `120.0` | Timeout for model call **or** tool batch |
| `episode_timeout_s` | `900.0` | Whole-loop wall clock |
| `parallel_tool_calls` | `True` | Concurrent tool execution |
| `drop_tools` | `[]` | Always remove these tool names |
| `drop_tools_prob` | `0.0` | Stochastic drop probability per tool |
| `include_thinking` | `True` | Keep reasoning fields if present |
| `step_penalty` | `0.0` | Subtracted × steps from reward |
| `timeout_penalty` | `0.0` | Extra penalty on timeout |
| `max_steps_penalty` | `0.0` | Extra penalty on max steps |

## RolloutConfig

| Field | Default | Meaning |
| --- | --- | --- |
| `sandbox` | `SandboxConfig()` | Container settings |
| `model` | *(required)* | `ModelConfig` |
| `agent` | `AgentConfig()` | Loop / tools |
| `seed` | `None` | RNG seed for tool dropping |
| `save_dir` | `None` | Optional default save location |
| `trajectory_format` | `"json"` | Hint for exporters |
| `run_id` | `None` | Auto UUID if unset |

## GenerateConfig

Located in `agentbox.tasks.generate.generator`:

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | required | Teacher model for generation |
| `base_url` / `api_key` | optional | OpenAI-compatible (vLLM/TGI/cloud) |
| `temperature` | `0.8` | Higher diversity for generation |
| `max_tokens` | `8192` | Large enough for multi-file tasks |
| `validate_in_docker` | `True` | Live seed + verifier QC |
| `validate_with_llm` | `True` | Second API call: DSPy task quality judge |
| `llm_judge_min_score` | `0.65` | Min judge score (0..1) to accept |
| `validator_model` | same as `model` | Optional separate judge model id |
| `validator_base_url` / `validator_api_key` | same as teacher | Optional separate judge endpoint |
| `sandbox` | network on | QC sandbox config |
| `max_retries` | `2` | Regenerate on QC failure |
| `use_dspy` | `True` | Fall back to JSON mode if DSPy missing |
| `expect_fail_on_starter` | `True` | Docker QC rejects tasks that already pass |
| `disable_thinking` | `True` | Auto-disable Qwen/GLM deep-thinking extras |

## Environment variables

| Variable | Use |
| --- | --- |
| `OPENAI_API_KEY` | Default API key for model + CLI |
| `AGENTBOX_LOG_LEVEL` | Logging level (if wired in process) |

## Example composition

```python
from agentbox.config import (
    AgentConfig, ModelConfig, ResourceLimits, RolloutConfig, SandboxConfig,
)
from agentbox.types import ToolMode

config = RolloutConfig(
    model=ModelConfig(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        temperature=0.2,
    ),
    agent=AgentConfig(
        tools=ToolMode.STRUCTURED,
        max_steps=30,
        drop_tools_prob=0.05,
        step_penalty=0.001,
    ),
    sandbox=SandboxConfig(
        image="agentbox/sandbox:latest",
        ensure_pytest=False,
        limits=ResourceLimits(
            cpu_count=1.0,
            memory_mb=1024,
            network_disabled=True,
        ),
    ),
    seed=42,
)
```
