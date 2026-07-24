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
agentbox doctor --model MODEL --base-url http://localhost:8000/v1
```

Reports: version, Python, Docker ping, default image presence, labeled container
count. Optional model probe checks chat + tool-calling (`tool_choice=auto`).
`--prune` removes containers with label `agentbox=1`.

Exit `2` if Docker is unavailable or model probe fails.

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

### `bench` (subcommand group)

Real-rollout benchmark suites. See [benchmarks.md](benchmarks.md).

```bash
agentbox bench create DIR --from-tasks tasks/ --suite-id ID --name NAME [--network]
agentbox bench freeze DIR
agentbox bench validate DIR [--strict]
agentbox bench run DIR --model-id LABEL -m MODEL --base-url URL [-o OUT]
agentbox bench run DIR --student id=model@url --student id2=model2@url2
agentbox bench run DIR --models-file models.yaml -o OUT --probe --limit 5
agentbox bench run DIR --mock -o OUT          # hermetic pipeline smoke
agentbox bench show OUT/report.json --out REPORT_extended.md
agentbox bench compare a/report.json b/report.json
```

### `generate` (subcommand group)

Automated coding-task generation + QC. See [generation.md](generation.md).

```bash
agentbox generate one -m MODEL --base-url URL -o generated/tasks
agentbox generate batch -m MODEL --base-url URL -n 20 -c 8 --two-stage
agentbox generate validate-llm generated/tasks -m MODEL --base-url URL
agentbox generate validate-docker generated/tasks
```

Teacher defaults can come from `agentbox.yaml` (`teacher` / `generate` sections).

### `traj` (subcommand group)

```bash
agentbox traj show path/to/traj.json
agentbox traj render bench-results/run1 -o traj-dash.html
agentbox traj render bench-results/run1 --model-id qwen-27b -o out.html
```

## Environment

| Variable | Effect |
| --- | --- |
| `OPENAI_API_KEY` | Default API key for `run` / `run-dir` / generate |
| `AGENTBOX_CONFIG` | Path to project config (else walk for `agentbox.yaml`) |

## Python vs CLI

The CLI covers common ops. Full flexibility (custom tools, generators, metrics)
is via the Python API — see [Architecture](architecture.md) and module docs.
