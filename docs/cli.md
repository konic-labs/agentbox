# CLI

Entry point: `agentbox` (Typer), installed via package scripts.

```bash
agentbox --help
agentbox version
```

## Commands

### `version`

Print package version.

### `doctor`

Check runtime readiness.

```bash
agentbox doctor
agentbox doctor --prune
```

Reports: version, Python, Docker ping, default image presence, labeled container
count. `--prune` removes containers with label `agentbox=1`.

Exit `2` if Docker is unavailable.

### `build-image`

Build the optional baked sandbox image.

```bash
agentbox build-image
agentbox build-image --tag agentbox/sandbox:latest --path docker/python-sandbox
```

### `run`

Single task rollout.

```bash
agentbox run TASK.json|TASK_DIR \
  --model mock|MODEL_ID \
  --base-url URL \
  --api-key KEY \
  --max-steps 40 \
  --tools structured|shell|custom \
  --out trajectories/ \
  --image IMAGE \
  --network
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | `mock` | `mock` uses a no-op final answer |
| `--base-url` | none | OpenAI-compatible endpoint |
| `--api-key` | env `OPENAI_API_KEY` | Optional for local servers |
| `--max-steps` | 40 | Agent steps |
| `--tools` | structured | Tool mode |
| `--out` | `trajectories/` | Output directory |
| `--image` | config default | Override image |
| `--network` | off | Enable container network |

Exit codes: `0` success, `1` failed/non-success, `2` infra/error.

### `run-dir`

Parallel batch over a directory of tasks.

```bash
agentbox run-dir tasks/ \
  --model MODEL \
  --base-url URL \
  --concurrency 16 \
  --n 4 \
  --max-steps 40 \
  --out trajectories/ \
  --network
```

Discovers `*/task.json` directories and top-level `*.json` task files.
Writes `dataset.jsonl` plus per-run JSON.

### `export`

```bash
agentbox export traj.json --format art --out traj.art.json
agentbox export traj.json --format jsonl --out traj.jsonl
```

### `prune`

```bash
agentbox prune
```

Remove all containers labeled `agentbox=1`.

## Environment

| Variable | Effect |
| --- | --- |
| `OPENAI_API_KEY` | Default API key for `run` / `run-dir` |

## Python vs CLI

The CLI covers common ops. Full flexibility (custom tools, generators, metrics)
is via the Python API — see [Architecture](architecture.md) and module docs.
