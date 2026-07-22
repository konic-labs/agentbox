# Runner

Runners orchestrate a full episode: sandbox → seed → agent → verify → trajectory.

## Rollout (single)

```python
from agentbox import Rollout, Task, Agent

traj = await Rollout.run(
    task,
    agent=agent,                 # or model= ModelClient | ModelConfig | str
    sandbox=SandboxConfig(...),  # optional
    config=RolloutConfig(...),   # optional
    manager=SandboxManager(...), # optional shared manager
)
```

### Resolution order

| Input | Behavior |
| --- | --- |
| `model` as `ModelClient` | Used directly |
| `model` as `ModelConfig` / `str` | Wrapped in `OpenAICompatClient` |
| `agent` as `Agent` | Supplies model client + `AgentConfig` |
| `config.model` | Fallback model |
| Task `max_steps` / `allowed_tools` | Override agent config |

### Lifecycle guarantees

- Fresh container per call (unless you inject a pre-created manager + reuse — not the default path)  
- Seed failure short-circuits to `ERROR`  
- Verifier runs after the loop when the sandbox is usable (including after max steps)  
- `finally`: destroy unless `keep_on_failure`  

## ParallelRunner

```python
from agentbox import ParallelRunner

runner = ParallelRunner(
    concurrency=16,
    agent=agent,
    sandbox=sandbox_config,
    # or config=RolloutConfig(...)
)

trajs = await runner.run_tasks(tasks, n_per_task=1, progress=True)
groups = await runner.run_groups(tasks, group_size=4)  # GRPO-style
result = await runner.run(tasks)  # ParallelResult with counts
```

### Properties

- `asyncio.Semaphore(concurrency)`  
- One `Rollout.run` per job with **isolated** managers (failure isolation)  
- Exceptions become `Trajectory(final_status=ERROR)`  
- Optional `rich` progress bar; degrades to logs if rich missing  
- `on_trajectory` callback per finished job  

### ParallelResult

```python
result.trajectories
result.succeeded / failed / errors
result.success_rate
```

### Concurrency guidance

| Host RAM | Suggested concurrency | Container memory |
| --- | --- | --- |
| 16 GB | 8–16 | 512 MB |
| 32 GB | 16–32 | 512 MB |
| 64 GB+ | 32–64 | 512 MB–1 GB |

Model server concurrency is a separate bottleneck.

### Mock models in parallel

`MockModelClient` is stateful. Use a **new instance per rollout**, not one shared
mock across concurrent jobs.

## CLI batch

```bash
agentbox run-dir tasks/ -m MODEL --base-url URL -c 16 --n 4 --out trajectories/
```

Writes per-trajectory JSON and `dataset.jsonl`.

## Metrics after a batch

```python
from agentbox import aggregate_trajectories
stats = aggregate_trajectories(trajs)
print(stats.success_rate, stats.mean_steps, stats.by_task)
```

See [Metrics](metrics.md).
