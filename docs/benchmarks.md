# Benchmarks

AgentBox **benchmarks** run the same real Docker rollouts used for trajectory
collection against any OpenAI-compatible model, then produce comparable reports.

This is evaluation only — no training.

## Why this exists

Training and evaluation must share one environment. A frozen **suite** pins:

- Tasks (`Task` definitions)
- Sandbox policy (image, resources, network)
- Agent policy (tools mode, max_steps, timeouts)
- Scoring policy (verifier + optional setup checks)
- Content hash for integrity

You can replay the suite on local servers, remote APIs, or post-train checkpoints
without changing task code.

## Suite layout

```text
my-benchmark/
  suite.json
  tasks/
    fix_fizzbuzz_001/
      task.json
      files/           # optional
    fix_reverse_001/
      task.json
```

Create from a tasks directory:

```bash
agentbox bench create examples/benchmarks/coding-mini \
  --from-tasks examples/tasks \
  --suite-id agentbox-coding-mini \
  --name "Coding Mini" \
  --version 1.0.0 \
  --network
```

Or via Python:

```python
from agentbox import create_suite_from_tasks, load_suite
from pathlib import Path

suite = create_suite_from_tasks(
    "examples/tasks",
    "examples/benchmarks/coding-mini",
    suite_id="agentbox-coding-mini",
    name="Coding Mini",
    freeze=True,
)
suite = load_suite("examples/benchmarks/coding-mini")
assert suite.verify_integrity()
```

## Freeze & integrity

```bash
agentbox bench freeze path/to/suite
agentbox bench validate path/to/suite --strict
```

`content_hash` covers task bodies, starter files, and fairness-critical sandbox/
agent/scoring fields. Tampering invalidates the hash.

## Setup checks vs task verifier

| Stage | Component | Failure meaning |
|-------|-----------|-----------------|
| Seed | `TaskSeeder` | Env materialization failed → `ERROR` |
| Setup checks | `SetupChecker` after seed | Runtime unhealthy → `ERROR` (not task fail) |
| Task verifier | `Verifier` after agent | Wrong solution → `FAILED` / reward 0 |

Setup checks are suite-level commands, e.g. `python -c 'import sys'`.
Do **not** put “tests already pass” checks here when the starter is intentionally broken.

## Running a suite

### CLI (real model)

```bash
agentbox bench run examples/benchmarks/coding-mini \
  --model-id ollama-qwen \
  --model qwen2.5-coder:7b \
  --base-url http://localhost:11434/v1 \
  --api-key ollama \
  --out bench-results/run1 \
  --concurrency 4
```

### Multiple models file

```yaml
# models.yaml
models:
  - model_id: ollama-qwen
    model: qwen2.5-coder:7b
    base_url: http://localhost:11434/v1
    api_key: ollama
  - model_id: remote
    model: glm-5.2
    base_url: https://api.featherless.ai/v1
    api_key: ${OPENAI_API_KEY}
```

```bash
agentbox bench run examples/benchmarks/coding-mini --models-file models.yaml -o bench-results/run1
```

### Python API

```python
from pathlib import Path
from agentbox import (
    load_suite, BenchmarkRunner, BenchmarkRunConfig, ModelUnderTest,
)
from agentbox.config import ModelConfig

suite = load_suite("examples/benchmarks/coding-mini")
report = await BenchmarkRunner(
    BenchmarkRunConfig(
        suite=suite,
        models=[
            ModelUnderTest(
                model_id="ollama-qwen",
                model=ModelConfig(
                    model="qwen2.5-coder:7b",
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                ),
            ),
        ],
        output_dir=Path("bench-results/run1"),
        save_trajectories=True,
    )
).run()
print(report.leaderboard)
```

### Hermetic demo

```bash
python examples/bench_run.py   # mock solver + Docker
```

## Reports

```text
bench-results/run1/
  report.json
  REPORT.md
  suite_snapshot/
  models/<model_id>/
    model_report.json
    trajectories/*.json
```

Leaderboard fields: `success_rate`, `mean_reward`, `mean_steps`, `mean_duration_s`,
`pass_at_k_mean`, `n`.

**Pass@k:** fraction of tasks with ≥1 success among `n_per_task` rollouts.  
API keys are redacted from reports.

```bash
agentbox bench show bench-results/run1/report.json
agentbox bench compare results/a/report.json results/b/report.json
```

## Fairness rules

- Same tools / max_steps / sandbox for every model under test  
- `drop_tools_prob` forced to 0 for suite agents  
- Suite content hash recorded on every report  
- Prefer fixed `temperature` in each `ModelConfig` when comparing  

## Relation to training

| Flow | Role of suite tasks |
|------|---------------------|
| Online RL (ART + AgentBox) | Same tasks as train scenarios |
| SFT bulk collection | Same tasks for trajectory dump |
| Benchmark | Same tasks for score-only multi-model eval |

## Implementation modules

| Module | Path |
|--------|------|
| Schema | `agentbox.benchmark.schema` |
| Hash | `agentbox.benchmark.hash` |
| Loader | `agentbox.benchmark.loader` |
| Setup checks | `agentbox.benchmark.setup_check` |
| Runner | `agentbox.benchmark.runner` |
| Report | `agentbox.benchmark.report` |
| Compare | `agentbox.benchmark.compare` |
| CLI | `agentbox.cli.bench` |

Rollouts always go through `Rollout` / `ParallelRunner` — no second agent stack.
