# Architecture

AgentBox is organized so **sandbox execution**, **tools**, **model I/O**,
**tasks/verification**, and **trajectory recording** stay independent. Docker
and LLM HTTP stay on opposite sides of a clear boundary.

## Control plane vs execution plane

| Concern | Plane | Location |
| --- | --- | --- |
| Prompts, message history | Control | Host process |
| Model HTTP (Chat Completions) | Control | Host (`openai.AsyncOpenAI`) |
| Tool *schemas* | Control | Host (serializable JSON) |
| Tool *execution* | Execution | Inside container via `SandboxManager` |
| Verifier *command* | Execution | Inside container |
| Trajectory serialization | Control | Host |

**Invariant:** the model never talks to Docker. Only `SandboxManager` /
`ToolExecutor` do.

## Module responsibilities

| Module / package | Responsibility | Notes |
| --- | --- | --- |
| `agentbox.cli` | Typer CLI | No Docker logic beyond calling APIs |
| `agentbox.config` | Pydantic configs | Sandbox / model / agent / rollout |
| `agentbox.types` | Enums | ToolMode, FinalStatus, VerifierType, MessageRole |
| `agentbox.errors` | Exception hierarchy | Infra vs task vs model |
| `agentbox.sandbox` | Container lifecycle, path jail, prune, snapshots | docker SDK + `asyncio.to_thread` |
| `agentbox.tools` | Builtins, custom tools, registry, executor | Always uses sandbox facade |
| `agentbox.model` | OpenAI-compatible + mock clients | No container I/O |
| `agentbox.agent` | `Agent` facade, `AgentLoop`, prompts | Orchestrates model + tools |
| `agentbox.tasks` | Schema, seeder, verifier, filters, bulk, rewards | Generator under `tasks.generate` |
| `agentbox.trajectory` | Messages, recorder, exporters | JSON / ART / JSONL |
| `agentbox.runner` | `Rollout`, `ParallelRunner` | Episode orchestration |
| `agentbox.metrics` | Aggregate stats over trajectories | Offline analytics |

## Package layout

```txt
src/agentbox/
  cli/                 # Typer entrypoint
  sandbox/             # Docker manager, paths, prune, snapshot, images
  tools/               # builtins, BaseTool, @tool, registry, executor
  model/               # OpenAICompatClient, MockModelClient
  agent/               # Agent, AgentLoop, prompts
  tasks/               # schema, seeder, verifier, filter, bulk, rewards
    generate/          # TaskGenerator, DSPy signatures, live QC
  trajectory/          # schema, recorder, formats/{art,jsonl}
  runner/              # rollout, parallel
  metrics/             # aggregate_trajectories
  config.py
  types.py
  errors.py
```

## Single-rollout data flow

```txt
Rollout.run(task, model|agent, sandbox_config)
  -> resolve ModelClient + AgentConfig + SandboxConfig
  -> SandboxManager.create(task_id, run_id)
       labels: agentbox=1, task_id, run_id
  -> TaskSeeder.seed
       write starter_files under /workspace
       run setup_commands sequentially (fail closed)
       healthcheck
  -> build_tool_registry (mode + custom + drop)
  -> AgentLoop.run(system, task.description, ToolContext)
       while steps < max_steps:
         model.complete(messages, tools=schemas)
         if tool_calls: ToolExecutor.execute_many -> append tool messages
         else: break (final answer)
  -> Verifier.verify (pytest|command)
  -> TrajectoryRecorder.finalize(reward, final_status, metrics)
  -> destroy sandbox (unless keep_on_failure)
  -> return Trajectory
```

## Parallel data flow

```txt
ParallelRunner(concurrency=N)
  -> expand tasks × n_per_task (or group_size for GRPO)
  -> asyncio.Semaphore(N)
  -> each job: isolated Rollout.run (own manager/container)
  -> failures become Trajectory(final_status=ERROR), do not cancel siblings
  -> optional rich progress bar
```

## Task generation data flow

```txt
TaskGenerator.generate(difficulty, domain, constraints)
  -> DSPy module OR OpenAI JSON completion
  -> parse_task_from_prediction -> Task (Pydantic)
  -> optional validate_task_live (Docker)
  -> optional validate_task_llm (DSPy structured judge, same teacher endpoint):
       create sandbox -> seed -> verifier smoke
       expect starter to FAIL verifier
  -> return Task | retry up to max_retries
```

## Design invariants

1. **Fresh isolation** — one container per rollout; no host bind-mount of project by default.
2. **Path jail** — all file tools resolve under `workspace_dir` (`/workspace`).
3. **OpenAI tools protocol** — assistant `tool_calls` + `role=tool` results with `tool_call_id`.
4. **Fail soft in the loop** — tool errors become tool-message text, not uncaught exceptions.
5. **Seed fail closed** — if `setup_commands` fail, agent never starts; status `ERROR`.
6. **Verifier authority** — reward comes from exit code (plus optional shaping).

## Dependency direction

```txt
cli -> runner -> agent -> model
              \-> tools -> sandbox
              \-> tasks  -> sandbox
              \-> trajectory
runner / tasks.generate -> sandbox (QC only)
```

Lower layers do not import CLI or runner. `tools` never import `agent`.

## Extension points

| Want to… | Extend |
| --- | --- |
| New tool | `BaseTool` or `@tool`, register via `custom_tools` |
| New model transport | Implement `ModelClient.complete` |
| New verifier type | Extend `Verifier` / `VerifierSpec` carefully |
| New export format | `trajectory/formats/` |
| New image family | `sandbox.images.PRESETS` or `SandboxConfig.image` |

See [Development](development.md) for testing expectations.
