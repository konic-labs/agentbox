# Configuration

Configuration has two layers that share one set of models:

- **`agentbox.yaml`** (the *project file*) is read by every CLI command and by
  `agentbox serve`.
- **Python objects** (`SandboxConfig`, `ModelConfig`, `AgentConfig`,
  `RolloutConfig`, `RewardSpec`) are what the runner, the environment API and
  the trainer integrations accept.

Every field, default and description is listed in the generated
[configuration reference](config-reference.md).

## The project file

```yaml
# agentbox.yaml
default_model: local                 # used when no --profile/--model is given

models:                              # named profiles (ModelConfig fields)
  local:
    model: qwen3-1.7b
    base_url: http://localhost:8000/v1
    api_key: ${VLLM_API_KEY:-EMPTY}
    temperature: 1.0
    capture_tokens: true
    capture_logprobs: true
    extra_body: {chat_template_kwargs: {enable_thinking: false}}
  gpt:
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY      # read at request time; never stored in trajectories

sandbox:                             # SandboxConfig
  image: python:3.12-slim-bookworm
  limits: {cpu_count: 1.0, memory_mb: 1024, pids_limit: 256, network_disabled: true}
  warm_pool: 4
  cache_setup: true

agent:                               # AgentConfig
  tools: structured                  # structured | extended | shell | custom | [names]
  max_steps: 30
  episode_timeout_s: 600
  tool_protocol: native              # or hermes for servers without tool parsing

rewards:                             # weighted components; default: verifier + agent penalties
  - {name: verifier}
  - {name: step_penalty, params: {per_step: 0.01}}

teacher:                             # task generation and LLM judging
  model: glm-5.2
  base_url: https://api.example.com/v1
  api_key_env: TEACHER_API_KEY
students:                            # models `bench run` evaluates without --model
  - {id: qwen-1.7b, model: qwen3-1.7b, base_url: http://localhost:8000/v1}
generate: {concurrency: 8, target: 50, min_score: 0.7}
```

### Where it is found

1. `--config/-C PATH` on the command line.
2. `$AGENTBOX_CONFIG` (an error if it points to a missing file).
3. The first `agentbox.yaml`, `agentbox.yml`, `agentbox.toml` or `agentbox.json`
   in the current directory or any parent directory.

With no file, defaults apply, so the CLI works without configuration.

### Environment interpolation

String values may reference environment variables:

| Syntax | Meaning |
| --- | --- |
| `${NAME}` | The value of `NAME`; error if unset |
| `${NAME:-default}` | The value, or `default` when unset or empty |
| `${NAME:?message}` | The value, or fail with `message` |

Prefer `api_key_env: NAME` for secrets. It is resolved when requests are made,
and keys never appear in trajectories, reports, `config show` output or command
lines.

### Profiles

`--profile/-P NAME` (or `$AGENTBOX_PROFILE`) selects a profile from `models`.
`students` entries become profiles named by their `id`, and `teacher` becomes
`teacher`. Commands that take a model also accept `--model`/`--base-url`/
`--api-key-env` directly, which override the profile.

### Checking a file

```bash
agentbox config show              # resolved config, keys redacted (--json for machines)
agentbox config validate          # schema errors exit 2; unknown keys are listed
agentbox config validate --strict # ... and unknown keys fail too (CI)
agentbox config path              # which file is in use
```

## Python objects

```python
from agentbox import AgentConfig, ModelConfig, ResourceLimits, RolloutConfig, SandboxConfig

config = RolloutConfig(
    model=ModelConfig(model="qwen3-1.7b", base_url="http://localhost:8000/v1", api_key="EMPTY"),
    sandbox=SandboxConfig(limits=ResourceLimits(memory_mb=1024), warm_pool=4),
    agent=AgentConfig(max_steps=30, tools="extended"),
    seed=7,                       # per-sample sampling seeds and tool dropout derive from it
)
```

`Rollout.run` and `ParallelRunner` take `config=` or the individual `model=`,
`sandbox=` and `agent=` arguments (explicit arguments win). `ModelConfig` can
be replaced by any `ModelClient` instance, such as a TITO client, a mock or
your own.

### Precedence

For a single rollout, settings resolve in this order (later wins):

1. Defaults of the config models.
2. The project file (CLI only).
3. CLI flags (`--max-steps`, `--image`, `--network`, `--warm-pool`, `--reward`, ...).
4. Per-task overrides: `Task.environment` (image, env vars, network, CPU,
   memory, working directory, `ensure_pytest`), `Task.max_steps`,
   `Task.episode_timeout_s` and `Task.allowed_tools`.

## Environment variables

| Variable | Effect |
| --- | --- |
| `AGENTBOX_CONFIG` | Project file path |
| `AGENTBOX_PROFILE` | Default `--profile` |
| `AGENTBOX_DOCKER_IO_THREADS` | Exec/archive thread pool size (default 128) |
| `AGENTBOX_DOCKER_CTL_THREADS` | Create/remove/inspect thread pool size (default 16) |
| `AGENTBOX_REAP_ORPHANS` | `0` disables removing dead processes' sandboxes at startup |
| `AGENTBOX_TASKS` | Task source used by the verl/OpenRLHF/SkyRL adapters to resolve `task_id`s |
| `AGENTBOX_VERL_CONFIG` / `AGENTBOX_OPENRLHF_CONFIG` / `AGENTBOX_SKYRL_CONFIG` | YAML/JSON with `sandbox`, `agent`, `rewards` sections for those adapters |
| `AGENTBOX_SERVER_TOKEN` | Bearer token required by `agentbox serve` |
| `DOCKER_HOST` | Docker endpoint (standard docker-py behaviour) |
