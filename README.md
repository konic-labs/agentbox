<div align="center">

# AgentBox

**ART-native rollout engine for real agentic trajectories in Docker sandboxes.**

[Quick Start](#quick-start) ·
[Workflow](#workflow) ·
[Tools](#tools) ·
[Models](#models--providers) ·
[Tasks](#tasks) ·
[Trajectories](#trajectories--art) ·
[CLI](#cli-reference) ·
[Benchmarks](#benchmarks) ·
[Docs](#technical-docs) ·
[Development](#development) ·
[License](#license)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Runtime](https://img.shields.io/badge/Runtime-Docker%20%7C%20AsyncIO-black)
![ART](https://img.shields.io/badge/ART-native-orange)
![Package](https://img.shields.io/badge/Package-agentbox-green)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

</div>

AgentBox is an **ART-native** environment + rollout harness: isolated Docker
sandboxes, structured tools (or pure shell), task seeding, verifiers, and
multi-turn trajectories shaped for [OpenPipe ART](https://github.com/OpenPipe/ART)
(`messages_and_choices`, rewards, groups) — for online GRPO with ART’s inference
backend, offline SFT dumps, and the same real-rollout path for benchmarks.

It is **not** a trainer. ART (or TRL / Unsloth / verl) owns training; AgentBox
owns the virtual env and trajectory collection.

## Highlights

- **ART-native trajectories** — `to_art_dict()` / `to_art()`; GRPO groups via `ParallelRunner`
- **One container per rollout** — clean state, parallel isolation, labeled GC
- **OpenAI tools protocol** — Chat Completions + tool calling (ART-compatible wire format)
- **Provider-agnostic models** — Featherless, OpenRouter, vLLM, Ollama, ART backend client, …
- **Builtins + custom tools** — `BaseTool` / `@tool`, override-by-name
- **Task seeding** — `starter_files` + `setup_commands` before the agent starts
- **Objective rewards** — pytest / command verifiers (hybrid-ready with ART RULER)
- **ParallelRunner** — concurrent rollouts and GRPO-style groups
- **Trajectory export** — JSON, JSONL, ART-native dicts
- **Typer CLI** — `doctor`, `run`, `run-dir`, `bench`, `export`, `build-image`, `prune`
- **Real-rollout benchmarks** — freeze tasks + env; score any OpenAI-compatible model
- **Task generation** — frontier model + optional DSPy + live Docker QC

## Quick Start

**Prerequisites:** Python 3.11+, [Docker](https://docs.docker.com/get-docker/)
daemon running, optional [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/konic-labs/agentbox.git
cd agentbox
uv venv .venv --seed --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# Optional
uv pip install -e ".[rich]"      # progress bars
uv pip install -e ".[art]"       # live art.Trajectory
uv pip install -e ".[generate]"  # DSPy task generation
```

Sanity checks (no external LLM required):

```bash
agentbox version
agentbox doctor
uv run pytest tests/unit -q
python examples/hello_rollout.py
```

### Minimal Python rollout (mock)

```python
import asyncio
from agentbox import Task, Rollout
from agentbox.config import SandboxConfig, ResourceLimits
from agentbox.model import MockModelClient, ModelResponse
from agentbox.trajectory.schema import ToolCall, FunctionCall
import json

async def main():
    task = Task.from_json("examples/tasks/fix_fizzbuzz/task.json")
    fixed = 'def fizzbuzz(n):\n    return "Fizz" if n % 3 == 0 else str(n)\n'
    mock = MockModelClient([
        ModelResponse(
            content=None,
            tool_calls=[ToolCall(
                id="c1",
                function=FunctionCall(
                    name="write_file",
                    arguments=json.dumps({"path": "fizzbuzz.py", "content": fixed}),
                ),
            )],
        ),
        ModelResponse(content="done", tool_calls=[]),
    ])
    traj = await Rollout.run(
        task,
        model=mock,
        sandbox=SandboxConfig(
            limits=ResourceLimits(network_disabled=False),
        ),
    )
    print(traj.final_status, traj.reward)
    traj.save("out/traj.json")

asyncio.run(main())
```

### Real model (Ollama)

```bash
agentbox run examples/tasks/fix_fizzbuzz/task.json \
  --model qwen2.5-coder:7b \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --network \
  --out trajectories/
```

```python
from agentbox import Agent, Task, Rollout

agent = Agent(
    model="qwen2.5-coder:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    max_steps=20,
)
traj = await Rollout.run(task=Task.from_json("examples/tasks/fix_fizzbuzz/task.json"), agent=agent)
```

### Parallel / GRPO-style groups

```python
from agentbox import ParallelRunner, Agent

runner = ParallelRunner(concurrency=16, agent=agent)
trajs = await runner.run_tasks(tasks, n_per_task=1)
groups = await runner.run_groups(tasks, group_size=4)
art = [[t.to_art_dict() for t in g] for g in groups]
```

## Workflow

| Step | What happens | Artifact |
| --- | --- | --- |
| 0. Task | Load or generate Task definition | `task.json` |
| 1. Sandbox | Create labeled Docker container | running container |
| 2. Seed | Write `starter_files`, run `setup_commands` | `/workspace` ready |
| 3. Agent | Multi-turn tools via OpenAI protocol | messages + tool records |
| 4. Verify | pytest / command exit code | reward |
| 5. Record | Finalize trajectory, destroy container | JSON / ART dict |

```txt
task → create sandbox → seed → agent loop → verifier → trajectory → destroy
```

## Tools

| Builtin | Role |
| --- | --- |
| `list_files` | List workspace paths |
| `read_file` | Read file content |
| `write_file` | Create / overwrite |
| `edit_file` | Exact one-occurrence replace (`old_string` / `new_string`) |
| `run_command` | Shell in container |
| `run_tests` | Default `python -m pytest -q` |

Modes: `structured` (all builtins), `shell` (`run_command` only), `custom`.

Custom tools:

```python
from agentbox.tools import tool

@tool(description="Count lines")
async def count_lines(sandbox, path: str) -> str:
    r = await sandbox.exec(f"wc -l -- {path}")
    return r.stdout.strip()

agent = Agent(model="...", base_url="...", custom_tools=[count_lines])
```

Same `name` overrides a builtin. Details: [docs/tools.md](docs/tools.md).

## Models & Providers

OpenAI Chat Completions + tools only. Configure any compatible endpoint:

| Type | Examples | Config |
| --- | --- | --- |
| External | Featherless, OpenRouter, Together, Fireworks | `base_url` + `api_key` |
| Local | vLLM, SGLang, Ollama `/v1` | `base_url`; `api_key` optional |

```python
Agent(model="glm-5.2", base_url="https://api.featherless.ai/v1", api_key="...")
Agent(model="Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1")
```

## Tasks

```json
{
  "task_id": "fix_fizzbuzz_001",
  "description": "Fix fizzbuzz so tests pass.",
  "starter_files": { "fizzbuzz.py": "...", "test_fizzbuzz.py": "..." },
  "setup_commands": ["pip install -q pytest"],
  "verifier": {
    "type": "pytest",
    "command": "python -m pytest -q",
    "success_exit_code": 0
  },
  "metadata": { "difficulty": "easy", "tags": ["python"] }
}
```

Automated generation (optional):

```python
from agentbox.tasks.generate import TaskGenerator, GenerateConfig

gen = TaskGenerator(GenerateConfig(
    model="glm-5.2",
    base_url="https://api.featherless.ai/v1",
    api_key="...",
    validate_in_docker=True,
))
task = await gen.generate(difficulty="easy", domain="python")
```

## Trajectories & ART

AgentBox is built to drop into ART workflows:

| Flow | AgentBox role | ART role |
| --- | --- | --- |
| **Online GRPO** | Docker env, tools, tasks, optional verifier | Inference (vLLM/LoRA) + `backend.train` |
| **Offline SFT** | Bulk rollouts from any API | SFT on exported trajectories |
| **Rewards** | Verifier scores | Optional RULER on groups (hybrid OK) |

```python
traj.save("trajectories/run.json")
art = traj.to_art_dict()   # ART-native: messages_and_choices, reward, metrics, metadata
# uv pip install -e ".[art]"
# live = traj.to_art()     # openpipe-art Trajectory object
```

```bash
agentbox export traj.json --format art -o traj.art.json
python examples/art_integration.py
```

## CLI Reference

```bash
agentbox doctor [--prune]
agentbox build-image [--tag agentbox/sandbox:latest]
agentbox run TASK.json -m MODEL --base-url URL [--network] [--out trajectories/]
agentbox run-dir tasks/ -m MODEL --base-url URL -c 16 --n 4
agentbox bench create DIR --from-tasks tasks/ --suite-id ID --name NAME
agentbox bench freeze DIR && agentbox bench validate DIR --strict
agentbox bench run DIR -m MODEL --base-url URL --model-id LABEL -o bench-results/run1
agentbox bench show bench-results/run1/report.json
agentbox export traj.json --format art -o out.json
agentbox prune
```

Full flag list: [docs/cli.md](docs/cli.md) · [docs/benchmarks.md](docs/benchmarks.md).

## Benchmarks

Same Docker rollouts as training collection, frozen as a suite, scored against
any OpenAI-compatible endpoint (local or external).

```bash
python examples/bench_run.py   # hermetic mock solver (Docker required)

agentbox bench run examples/benchmarks/coding-mini \
  --model-id ollama-qwen \
  --model qwen2.5-coder:7b \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --out bench-results/run1
```

Setup checks run after seed (env health); task verifiers own success/reward.
Details: [docs/benchmarks.md](docs/benchmarks.md).

## Sandbox image

Default: **`python:3.12-slim-bookworm`** (glibc; not Alpine).

```bash
agentbox build-image   # optional baked image with pytest + unix tools
```

Override: `SandboxConfig(image="my-org/env:1.2")` or presets in
`agentbox.sandbox.images`.

## Technical Docs

| Doc | Concern |
| --- | --- |
| [docs/index.md](docs/index.md) | Documentation map |
| [docs/setup.md](docs/setup.md) | Install & first runs |
| [docs/architecture.md](docs/architecture.md) | Module boundaries & data flow |
| [docs/configuration.md](docs/configuration.md) | All config fields |
| [docs/sandbox.md](docs/sandbox.md) | Docker lifecycle, jail, prune |
| [docs/tools.md](docs/tools.md) | Builtins & custom tools |
| [docs/models.md](docs/models.md) | Providers & clients |
| [docs/agent.md](docs/agent.md) | Loop, timeouts, prompts |
| [docs/tasks.md](docs/tasks.md) | Schema, seeder, verifier |
| [docs/generation.md](docs/generation.md) | Automated task generation |
| [docs/trajectories.md](docs/trajectories.md) | Formats & ART export |
| [docs/runner.md](docs/runner.md) | Rollout & parallel |
| [docs/cli.md](docs/cli.md) | CLI reference |
| [docs/benchmarks.md](docs/benchmarks.md) | Real-rollout multi-model suites |
| [docs/security.md](docs/security.md) | Isolation model |
| [docs/development.md](docs/development.md) | Tests & extension |

## Development

```bash
uv pip install -e ".[dev]"
uv run pytest tests/unit -q
uv run pytest tests/integration -q -m docker
uv run agentbox doctor
```

Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, …). See
[docs/development.md](docs/development.md).

## Security (short)

Tools run **only** inside Docker; paths jailed under `/workspace`; network off
by default; never mount the Docker socket into sandboxes. Details:
[docs/security.md](docs/security.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
