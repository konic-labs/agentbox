# AgentBox

Fully local, open-source framework for running LLM agents in isolated Docker sandboxes and collecting multi-turn trajectories for offline datasets and online RL (ART / GRPO).

AgentBox is the **environment + rollout harness**, not a trainer. Pair it with OpenPipe ART, TRL, Unsloth, or verl for training.

## Features

- **One Docker container per rollout** with CPU/memory/pids limits
- **Structured tools** + pure shell mode + **custom tools** (`BaseTool` / `@tool`)
- **OpenAI-compatible** models (Featherless, OpenRouter, vLLM, Ollama, …)
- **Task seeding**: `starter_files` + `setup_commands` before the agent starts
- **Verifiers** (`pytest` / shell) → rewards (optional step penalties)
- **ParallelRunner** for concurrent rollouts / GRPO groups
- **Trajectory export**: JSON, JSONL, ART-compatible dict
- **CLI**: `doctor`, `run`, `run-dir`, `export`, `build-image`, `prune`
- **Task generation** (optional): frontier model + DSPy + live Docker QC
- **Curriculum filters**, metrics aggregation, container snapshots

## Requirements

- Python 3.11+
- Docker Engine (daemon running)

## Install

```bash
git clone <repo-url> agentbox && cd agentbox
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional extras:

```bash
uv pip install -e ".[rich]"      # progress bars
uv pip install -e ".[art]"       # live art.Trajectory objects
uv pip install -e ".[generate]"  # DSPy task generation
```

## Quick start (mock, no LLM)

```bash
python examples/hello_rollout.py
# Parallel mock rollouts
python examples/parallel_rollouts.py
# ART-shaped groups
python examples/art_integration.py
```

## CLI

```bash
agentbox doctor
agentbox doctor --prune
agentbox build-image
agentbox run examples/tasks/fix_fizzbuzz/task.json --model mock --network
agentbox run task.json -m qwen2.5-coder:7b --base-url http://localhost:11434/v1
agentbox run-dir tasks/ -m MODEL --base-url URL -c 16 --n 4 --out trajectories/
agentbox export traj.json --format art -o traj.art.json
agentbox prune
```

## Python API

```python
import asyncio
from agentbox import Task, Agent, Rollout, ParallelRunner
from agentbox.tools import tool

@tool(description="Count lines in a workspace file")
async def count_lines(sandbox, path: str) -> str:
    r = await sandbox.exec(f"wc -l -- {path}")
    return r.stdout.strip()

async def main():
    task = Task.from_json("examples/tasks/fix_fizzbuzz/task.json")

    agent = Agent(
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        max_steps=20,
        custom_tools=[count_lines],
        # drop_tools_prob=0.1,  # random tool ablation
    )

    traj = await Rollout.run(task=task, agent=agent)
    print(traj.final_status, traj.reward)
    traj.save("trajectories/run.json")
    print(traj.to_art_dict()["reward"])

    # Parallel / GRPO groups
    runner = ParallelRunner(concurrency=16, agent=agent)
    groups = await runner.run_groups([task], group_size=4)
    art_group = [t.to_art_dict() for t in groups[0]]

asyncio.run(main())
```

### Task definition

```json
{
  "task_id": "fix_fizzbuzz_001",
  "description": "Fix fizzbuzz so tests pass.",
  "starter_files": {
    "fizzbuzz.py": "def fizzbuzz(n):\n    return str(n)\n",
    "test_fizzbuzz.py": "..."
  },
  "setup_commands": ["pip install -q pytest"],
  "verifier": {
    "type": "pytest",
    "command": "python -m pytest -q",
    "success_exit_code": 0
  },
  "metadata": { "difficulty": "easy", "tags": ["python"] }
}
```

Flow: create clean container → write files → run `setup_commands` → **then** agent → verifier → trajectory.

### Custom tools

```python
from agentbox.tools import BaseTool, tool

@tool(name="search_repo", description="grep under workspace")
async def search_repo(sandbox, query: str) -> str:
    r = await sandbox.exec(f"grep -RIn -- {query!r} . || true")
    return r.stdout or "(no matches)"
```

Register with `Agent(..., custom_tools=[search_repo])`. Same `name` overrides a builtin.

### Builtin tools

| Tool | Purpose |
|------|---------|
| `list_files` | List workspace paths |
| `read_file` | Read file content |
| `write_file` | Create/overwrite file |
| `edit_file` | Exact one-occurrence replace (`old_string` / `new_string`) |
| `run_command` | Shell in container |
| `run_tests` | Default `python -m pytest -q` |

Shell mode: `Agent(..., tools="shell")`.

### Automated task generation (Phase 3)

```python
from agentbox.tasks.generate import TaskGenerator, GenerateConfig

gen = TaskGenerator(GenerateConfig(
    model="glm-5.2",
    base_url="https://api.featherless.ai/v1",
    api_key="...",
    validate_in_docker=True,  # seed + verifier smoke
    use_dspy=True,            # falls back to OpenAI JSON if DSPy missing
))
task = await gen.generate(difficulty="medium", domain="python")
task.save_json("tasks/generated/task.json")
```

CLI-style example:

```bash
python examples/generate_tasks.py --model glm-5.2 --base-url https://api.featherless.ai/v1 --n 5
```

### Filtering & curriculum

```python
from agentbox.tasks import filter_tasks, sample_curriculum

easy = filter_tasks(tasks, difficulty="easy", tags=["python"])
batch = sample_curriculum(tasks, n=20)  # easy → medium → hard
```

### Metrics

```python
from agentbox import aggregate_trajectories
stats = aggregate_trajectories(trajs)
print(stats.success_rate, stats.mean_steps, stats.by_task)
```

### Snapshots & multi-language images

```python
from agentbox.sandbox import commit_sandbox
from agentbox.sandbox.images import sandbox_config_for_preset

image = await commit_sandbox(sandbox, repository="agentbox/snap")
cfg = sandbox_config_for_preset("node")  # or python / go / baked
```

## Default sandbox image

Default: `python:3.12-slim-bookworm` (glibc, **not** Alpine).

```bash
docker build -t agentbox/sandbox:latest docker/python-sandbox
agentbox build-image
```

## Tests

```bash
pytest tests/unit -q
pytest tests/integration -q -m docker
```

## Architecture (short)

```text
TaskGenerator (optional) → Task → TaskSeeder → AgentLoop + Tools → Verifier → Trajectory
                                      ↑
                               Docker sandbox (one per rollout)
```

## Security notes

- Tools always execute **inside** the container
- Paths jailed under `/workspace`
- Network disabled by default
- Never mount the Docker socket into sandboxes

## Roadmap status

| Phase | Status |
|-------|--------|
| 1 MVP (sandbox, tools, loop, seeder, trajectories) | **Done** |
| 2 Usability (parallel, CLI, dropping, prune) | **Done** |
| 3 Task generation (DSPy + live QC) | **Done** |
| 4 Advanced (snapshots, multi-image, metrics, shaping) | **Done** (core) |

Optional later: web trajectory viewer, richer dashboards, live ART trainer adapters.

## License

Apache-2.0
