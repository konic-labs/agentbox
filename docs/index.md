# AgentBox — Technical Documentation

Maintainer- and user-focused reference for **AgentBox**: a fully local framework
for running LLM agents inside isolated Docker sandboxes and collecting multi-turn
trajectories for offline datasets and online RL (ART / GRPO-style pipelines).

This tree follows **Separation of Concerns (SoC)**: one topic per document.
Cross-links replace duplication. The package is the source of truth for APIs;
these docs describe contracts, data flow, and operational practice.

## Runtime shape

```txt
CLI / Python API
  -> Task (schema + optional generator)
  -> SandboxManager.create (Docker container)
  -> TaskSeeder (starter_files + setup_commands)
  -> AgentLoop + ToolExecutor (OpenAI tools)
  -> Verifier (pytest / command)
  -> Trajectory (JSON / ART dict / JSONL)
  -> optional ParallelRunner / metrics / prune
```

Control plane (prompts, models, trajectories) stays on the host. Execution plane
(file I/O, shell, tests) always runs **inside** the container.

## Start here

| Goal | Document |
| --- | --- |
| Install, first mock rollout, Docker checks | **[Setup](setup.md)** |
| Module boundaries and invariants | [Architecture](architecture.md) |
| How to run from the terminal | [CLI](cli.md) |
| Single / parallel rollouts | [Runner](runner.md) |
| Config models and defaults | [Configuration](configuration.md) |

## Documentation map

| Document | Concern |
| --- | --- |
| [Setup](setup.md) | Install, Docker, images, first runs, verification |
| [Architecture](architecture.md) | Package layout, control vs execution, data flow |
| [Configuration](configuration.md) | `SandboxConfig`, `ModelConfig`, `AgentConfig`, `RolloutConfig` |
| [Sandbox](sandbox.md) | Container lifecycle, path jail, resources, prune, snapshots |
| [Tools](tools.md) | Builtins, custom tools, registry, executor |
| [Models](models.md) | OpenAI-compatible client, providers, mock client |
| [Agent](agent.md) | Agent facade, loop, prompts, timeouts |
| [Tasks](tasks.md) | Task schema, seeder, verifier, filters, bulk I/O, rewards |
| [Task Generation](generation.md) | Automated generation, DSPy, live QC |
| [Trajectories](trajectories.md) | Message format, recorder, JSON / ART / JSONL export |
| [Runner](runner.md) | `Rollout.run`, `ParallelRunner`, GRPO-style groups |
| [Metrics](metrics.md) | Aggregate success rate, steps, per-task stats |
| [CLI](cli.md) | Typer commands, flags, exit codes |
| [Security](security.md) | Isolation model, network, path jail, residual risk |
| [Benchmarks](benchmarks.md) | Frozen real-rollout suites, multi-model reports, setup checks |
| [Development](development.md) | Tests, packaging, extension checklist |

## Core guarantees

- **Container per rollout**: each episode gets a fresh Docker container by default.
- **Tools execute in-container only**: host filesystem is not the agent workspace.
- **OpenAI wire format only**: Chat Completions + tool calling for model I/O.
- **Provider-agnostic models**: any OpenAI-compatible `base_url` (local or remote).
- **Seed before agent**: files and `setup_commands` complete before the loop starts.
- **Verifier owns reward**: objective exit codes (optionally shaped by step costs).
- **Trajectory export is self-contained**: offline ART-shaped dicts without requiring ART installed.

## Public Python surface

Primary imports from `agentbox`:

```python
from agentbox import (
    Agent,
    Task,
    Rollout,
    ParallelRunner,
    Trajectory,
    SandboxManager,
    BaseTool,
    tool,
    aggregate_trajectories,
)
```

Optional:

```python
from agentbox.tasks.generate import TaskGenerator, GenerateConfig
from agentbox.sandbox import prune_agentbox_containers, commit_sandbox
from agentbox.sandbox.images import sandbox_config_for_preset
```

## Related

- [README](../README.md) — project overview and quick start
- OpenPipe ART — GRPO training consumer of exported trajectories
- Examples under [`examples/`](../examples/)
