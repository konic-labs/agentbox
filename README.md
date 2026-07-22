# AgentBox

Fully local, open-source framework for running LLM agents in isolated Docker sandboxes and collecting multi-turn trajectories for offline datasets and online RL (ART / GRPO).

AgentBox is the **environment + rollout harness**, not a trainer. Pair it with OpenPipe ART, TRL, Unsloth, or verl for training.

## Features (MVP)

- **One Docker container per rollout** with resource limits
- **Structured tools** + pure shell mode
- **Custom tools** via `BaseTool` or `@tool`
- **OpenAI-compatible** models only (v1): Featherless, OpenRouter, vLLM, Ollama, etc.
- **Task seeding**: `starter_files` + `setup_commands` before the agent starts
- **Verifiers** (`pytest` / shell command) for rewards
- **Trajectory export**: JSON + ART-compatible dict

## Requirements

- Python 3.11+
- Docker Engine (daemon running)

## Install

```bash
git clone <repo-url> agentbox
cd agentbox
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional extras:

```bash
uv pip install -e ".[art]"       # live art.Trajectory objects
uv pip install -e ".[generate]"  # DSPy task generation (Phase 3)
```

## Quick start (mock, no LLM)

```bash
python examples/hello_rollout.py
```

This runs a scripted agent that fixes a broken `fizzbuzz` task inside Docker and writes `out/hello_traj.json`.

## Python API

```python
import asyncio
from agentbox import Task, Agent, Rollout
from agentbox.tools import tool

@tool(description="Count lines in a workspace file")
async def count_lines(sandbox, path: str) -> str:
    r = await sandbox.exec(f"wc -l -- {path}")
    return r.stdout.strip()

async def main():
    task = Task.from_json("examples/tasks/fix_fizzbuzz/task.json")

    # Local inference (Ollama example)
    agent = Agent(
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        max_steps=20,
        custom_tools=[count_lines],
    )

    # Or remote OpenAI-compatible provider:
    # agent = Agent(
    #     model="glm-5.2",
    #     base_url="https://api.featherless.ai/v1",
    #     api_key="...",
    # )

    traj = await Rollout.run(task=task, agent=agent)
    print(traj.final_status, traj.reward)
    traj.save("trajectories/run.json")
    print(traj.to_art_dict()["reward"])

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

The container is created clean, files are written, `setup_commands` run, **then** the agent starts.

### Custom tools

```python
from agentbox.tools import BaseTool, tool

@tool(name="search_repo", description="grep under workspace")
async def search_repo(sandbox, query: str) -> str:
    r = await sandbox.exec(f"grep -RIn -- {query!r} . || true")
    return r.stdout or "(no matches)"

class MyTool(BaseTool):
    name = "my_tool"
    description = "..."
    def parameters(self):
        return {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    async def execute(self, sandbox, x: str) -> str:
        return x
```

Register with `Agent(..., custom_tools=[search_repo, MyTool()])`. Same `name` overrides a builtin.

### Builtin tools

| Tool | Purpose |
|------|---------|
| `list_files` | List workspace paths |
| `read_file` | Read file content |
| `write_file` | Create/overwrite file |
| `edit_file` | Exact one-occurrence search/replace (`old_string` / `new_string`) |
| `run_command` | Shell in container |
| `run_tests` | Default `python -m pytest -q` |

Shell mode: `Agent(..., tools="shell")` exposes only `run_command`.

### ART export

```python
art_dict = traj.to_art_dict()
# keys: messages_and_choices, reward, metrics, metadata, tools
```

## Default sandbox image

Default: `python:3.12-slim-bookworm` (glibc, not Alpine). Override with:

```python
from agentbox.config import SandboxConfig
Rollout.run(task=task, agent=agent, sandbox=SandboxConfig(image="my-image:latest"))
```

Optional baked image:

```bash
docker build -t agentbox/sandbox:latest docker/python-sandbox
```

## Tests

```bash
pytest tests/unit -q
pytest tests/integration -q -m docker   # requires Docker
```

## Roadmap

| Phase | Focus |
|-------|--------|
| **1 (done MVP)** | Sandbox, tools, agent loop, seeder, verifier, trajectories |
| **2** | Parallel runner, CLI, tool dropping, resource UX |
| **3** | DSPy task generator + live QC |
| **4** | Snapshots, multi-language images, dashboards |

## Security notes

- Tools always execute **inside** the container
- Paths are jailed under `/workspace`
- Network disabled by default (`ResourceLimits.network_disabled=True`)
- Never mount the Docker socket into sandboxes

## License

Apache-2.0
