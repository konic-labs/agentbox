# Benchmarks

A **suite** freezes everything that decides a score (tasks, sandbox, agent
policy, scoring and sample count) behind a content hash. You can then compare
models and checkpoints on exactly the same environment you train in. A
benchmark run uses the same `Rollout` / `ParallelRunner` stack as training
rollouts; there is no second agent implementation.

## Suites

```text
my-suite/
  suite.json            # manifest: id, version, sandbox, agent, scoring, n_per_task, seed, content_hash
  tasks/
    fix_fizzbuzz_001/task.json
    fix_fizzbuzz_001/files/...
```

```bash
agentbox bench create suites/coding-v1 --from-tasks tasks/ \
  --suite-id coding --name "Coding v1" --version 1.0.0 --n 4   # copies tasks and freezes the hash
agentbox bench validate suites/coding-v1 --strict                # structure + integrity (exit 1 on mismatch)
agentbox bench freeze suites/coding-v1                           # re-freeze after an intended edit
```

The content hash (version 2) covers each task except its free-form `metadata`,
plus the fairness-critical configuration: image, workspace, environment,
`ensure_pytest`, CPU, memory, PIDs and network; the agent's tools, prompt,
step and time budgets, tool output budget, protocol and tool dropout; and
`n_per_task`, seed and scoring. Labels, pull policy and new unrelated config
options do not change it. Every report records the hash, and `--strict` runs
refuse a suite whose hash does not match.

## Running

```bash
agentbox bench run suites/coding-v1 --profile qwen-1.7b --profile gpt -o bench/2026-10-27
agentbox bench run suites/coding-v1 --model qwen3-1.7b --base-url http://localhost:8000/v1 --model-id base
agentbox bench run suites/coding-v1 --student ckpt-200=qwen3-1.7b@http://gpu-1:8000/v1 \
                                     --student ckpt-400=qwen3-1.7b@http://gpu-2:8000/v1
agentbox bench run suites/coding-v1 --models-file models.yaml --probe --resume
```

| Option | Effect |
| --- | --- |
| `--profile`, `--student id=MODEL@URL`, `--models-file`, `--model` | Which models to run (several are allowed; without any, every profile in `agentbox.yaml` except the teacher) |
| `-c/--concurrency`, `--n`, `--limit` | Override concurrency, samples per task, number of tasks |
| `--resume` | Skip rollouts already present in the output directory |
| `--fail-fast` | Stop a model's run at its first infrastructure failure |
| `--probe` | Check chat + tool calling for each model before running |
| `--strict` | Refuse on integrity mismatch |
| `--min-success-rate X` | Exit 1 if the best model scores below X (CI) |
| `--mock` | No-op model (pipeline smoke test) |

```yaml
# models.yaml
models:
  - {model_id: base, model: qwen3-1.7b, base_url: http://localhost:8000/v1}
  - {model_id: api, model: gpt-4.1-mini, api_key_env: OPENAI_API_KEY}
```

Output:

```text
bench/2026-10-27/
  report.json  REPORT.md  suite_snapshot/
  models/<model_id>/model_report.json
  models/<model_id>/trajectories/*.json
```

## Reading results

| Metric | Definition |
| --- | --- |
| success rate | Successful rollouts / all rollouts |
| pass@k | Unbiased estimator `1 − C(n−c, k) / C(n, k)` per task (Chen et al., 2021), averaged over tasks; reported for k ∈ {1, 5, 10} when k ≤ n |
| pass@n | Fraction of tasks with at least one success |
| mean reward, steps, duration | Over all rollouts |
| infra failures | Rollouts whose failure was not the model's ([failure kinds](rewards.md#infrastructure-failures-are-not-rewards)); retried, and still counted as failures in the score |

```bash
agentbox bench show bench/2026-10-27            # leaderboard, per-task matrix, top failure clusters
agentbox bench show bench/2026-10-27 --out SUMMARY.md
agentbox traj stats bench/2026-10-27            # trajectory-level statistics of a run
agentbox traj render bench/2026-10-27 --model-id base -o base.html
```

Failure clusters group failed rollouts by failure kind, verifier exit code and
a normalised head of the verifier output. That makes systematic problems
visible, such as a broken setup, a missing dependency or one test every model
fails.

## Comparing runs

```bash
agentbox bench compare bench/main/report.json bench/candidate/report.json --fail-on-regression 0.02
```

For each model present in both reports, `compare` pairs tasks and reports the
change in success rate, a **bootstrap 95% confidence interval** of the mean
per-task delta, wins, losses and ties, an exact **sign-test p-value**, and the
regressed tasks. With `--fail-on-regression X`, it exits 1 when success drops
by more than X, which makes a simple CI gate. It warns when suite ids or
content hashes differ.

## Setup checks

`ScoringConfig.setup_checks` are suite-level commands that run after seeding
and before the agent (for example `python -c "import numpy"`). A failing check
marks the rollout `failure_kind=setup_check`: the environment was broken, not
the model. Do not use them to assert that the starter code passes tests, since
starters are meant to fail.

## Fairness

- Every model runs with the suite's tools, step budget, sandbox and tool
  dropout (forced to 0 for suites).
- Fix sampling parameters (temperature, top_p, max_tokens) per model in
  profiles or `models.yaml`. Reports record each model's configuration with
  keys redacted.
- Use `--n ≥ 4` with pass@k when comparing models that are close.

## Python

```python
from pathlib import Path
from agentbox import BenchmarkRunConfig, BenchmarkRunner, ModelConfig, ModelUnderTest, load_suite

suite = load_suite("suites/coding-v1")
report = await BenchmarkRunner(BenchmarkRunConfig(
    suite=suite,
    models=[ModelUnderTest(model_id="base", model=ModelConfig(model="qwen3-1.7b",
                                                              base_url="http://localhost:8000/v1"))],
    output_dir=Path("bench/run1"),
    resume=True,
)).run()
print(report.leaderboard)
```

`agentbox.compare_reports(a, b)` returns the comparison as data.
