# Setup

Install AgentBox, prepare Docker, and run a first successful rollout.

## Prerequisites

| Requirement | Notes |
| --- | --- |
| Python **3.11+** | Developed against 3.12 via `uv` |
| **Docker Engine** | Daemon must be reachable (`docker ping`) |
| Disk / RAM | Containers default to ~512 MB each; concurrency multiplies usage |

Optional:

- Local inference server (Ollama, vLLM, SGLang) **or** remote OpenAI-compatible API
- [`uv`](https://github.com/astral-sh/uv) for fast installs

## Install

```bash
git clone https://github.com/konic-labs/agentbox.git
cd agentbox
uv venv .venv --seed --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Optional extras

```bash
uv pip install -e ".[rich]"      # progress bars for ParallelRunner / CLI
uv pip install -e ".[art]"       # live art.Trajectory via to_art()
uv pip install -e ".[generate]"  # DSPy for structured task generation
```

Core dependencies stay thin: `docker`, `pydantic`, `openai`, `httpx`, `typer`,
`anyio`, `structlog`.

## Verify environment

```bash
agentbox version
agentbox doctor
pytest tests/unit -q
```

`agentbox doctor` checks:

- Python version
- Docker daemon connectivity
- Default image presence (`python:3.12-slim-bookworm`)
- Count of leftover containers labeled `agentbox=1`

```bash
agentbox doctor --prune   # also remove orphan agentbox containers
```

## Default sandbox image

Default runtime image: **`python:3.12-slim-bookworm`** (Debian slim / glibc).

Why not Alpine: musl breaks many Python wheels and developer tooling.

On first use AgentBox will **pull** the image if `auto_pull=True` (default).
If `ensure_pytest=True` and pytest is missing, it may run `pip install pytest`
when network is enabled on the container.

### Optional baked image

Faster cold starts with pytest and Unix tools preinstalled:

```bash
agentbox build-image
# equivalent:
docker build -t agentbox/sandbox:latest docker/python-sandbox
```

Then:

```python
from agentbox.config import SandboxConfig
SandboxConfig(image="agentbox/sandbox:latest", ensure_pytest=False)
```

## First run (mock model, no LLM)

Does not call any external API:

```bash
python examples/hello_rollout.py
```

Expected: Docker container spins up, mock agent rewrites `fizzbuzz.py`, pytest
verifier returns reward `1.0`, trajectory written under `out/`.

Other hermetic demos:

```bash
python examples/parallel_rollouts.py
python examples/art_integration.py
```

## First run (real model)

### Local Ollama

```bash
# terminal A
ollama serve
ollama pull qwen2.5-coder:7b

# terminal B
agentbox run examples/tasks/fix_fizzbuzz/task.json \
  --model qwen2.5-coder:7b \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --network \
  --out trajectories/
```

`--network` enables container networking so `setup_commands` / pip can work when
needed. Default sandboxes have **network disabled** for isolation.

### Remote OpenAI-compatible provider

```bash
export OPENAI_API_KEY=...
agentbox run examples/tasks/fix_fizzbuzz/task.json \
  --model glm-5.2 \
  --base-url https://api.featherless.ai/v1 \
  --network \
  --max-steps 40
```

Python equivalent:

```python
import asyncio
from agentbox import Agent, Task, Rollout

async def main():
    task = Task.from_json("examples/tasks/fix_fizzbuzz/task.json")
    agent = Agent(
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        max_steps=20,
    )
    traj = await Rollout.run(task=task, agent=agent)
    print(traj.final_status, traj.reward)
    traj.save("trajectories/run.json")

asyncio.run(main())
```

## Tests

```bash
pytest tests/unit -q
pytest tests/integration -q -m docker   # requires Docker daemon
```

Integration tests are skipped cleanly if Docker is unavailable when you structure
tests with the project’s availability check (see [Development](development.md)).

## Common issues

| Symptom | Fix |
| --- | --- |
| `DockerNotAvailableError` | Start Docker Desktop / dockerd; re-run `agentbox doctor` |
| Image pull slow | Pre-pull: `docker pull python:3.12-slim-bookworm` |
| `pytest` missing in container | Use baked image, or `network_disabled=False` + `ensure_pytest=True` |
| Model 400 on tools | Endpoint must support OpenAI tool calling |
| Leftover containers | `agentbox prune` or `agentbox doctor --prune` |

## Next

- [Architecture](architecture.md) — how packages split responsibilities  
- [CLI](cli.md) — full command reference  
- [Configuration](configuration.md) — all knobs  
