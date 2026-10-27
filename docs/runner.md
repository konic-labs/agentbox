# Runner

## One rollout

```python
from agentbox import Rollout

traj = await Rollout.run(
    task,
    model=ModelConfig(...),          # or a ModelClient, or a model id string
    sandbox=SandboxConfig(...),      # optional
    agent=AgentConfig(...),          # optional
    config=RolloutConfig(...),       # optional bundle (explicit arguments win)
    hooks=None,                      # AgentHooks
    group_id=None, sample_index=0,   # GRPO bookkeeping
)
```

`Rollout.run` never raises for rollout failures. Provisioning errors, model API
errors, verifier crashes and internal errors all come back as a `Trajectory`
with `final_status=error` and the right `failure_kind`, and the sandbox is
always released.

## Many rollouts

```python
from agentbox import JsonlSink, ParallelRunner

runner = ParallelRunner(
    model=ModelConfig(...),
    sandbox=SandboxConfig(warm_pool=4),
    concurrency=32,             # sandboxes in flight
    model_concurrency=64,       # optional separate cap on in-flight model requests
    max_retries=1,              # retries of infrastructure failures per rollout
    retry_backoff_s=2.0,
    run_id="exp-42",
)
```

| Method | Returns | Use |
| --- | --- | --- |
| `stream(tasks, n_per_task=1)` | async iterator of trajectories in **completion order** | the streaming primitive |
| `stream_groups(tasks, group_size=8)` | async iterator of complete groups (lists sorted by `sample_index`) | online GRPO: train on a group as soon as all its samples are done |
| `run_to_sink(tasks, sink, n_per_task=1, resume=False)` | `RunSummary` | data collection: write as you go and aggregate online |
| `run_tasks(tasks, n_per_task=1)` | `list[Trajectory]` in task order | small runs and tests |
| `run_groups(tasks, group_size=4)` | `list[list[Trajectory]]` | small GRPO batches |

```python
summary = await runner.run_to_sink(iter_tasks("tasks/"), JsonlSink("runs/a/trajectories.jsonl"),
                                   n_per_task=8, resume=True, progress=True)
print(summary.metrics.success_rate, summary.rollouts_per_min, summary.skipped)
```

### Memory and backpressure

`tasks` may be any iterable, including a lazy `iter_tasks(...)` over a large
dataset. Jobs are generated on demand, results pass through a queue bounded by
`2 × concurrency`, and `run_to_sink` aggregates metrics incrementally. Memory
therefore does not grow with the number of rollouts: 2 000 rollouts with 50 KB
of output each peak below 25 MB of Python heap.

### Groups and ids

Each rollout has a unique `rollout_id`. Samples of one task in one run share
`group_id = "<task_id>@<run_id[:8]>"` and carry `sample_index` 0..n-1.
`run_id` is the runner's (random unless given), so trajectory files never
overwrite each other across runs.

### Infrastructure retries

A rollout whose `failure_kind` is in `retry_on` (default: `sandbox`, `seed`,
`setup_check`, `model_api`, `verifier`) is retried up to `max_retries` times
with backoff. The final trajectory records `metrics.attempts`. Policy failures
are never retried; they are the training signal.

### Resume

`run_to_sink(..., resume=True)` asks the sink which `(task_id, sample_index)`
pairs are complete (`JsonlSink.completed()` reads its file) and skips them.
On the CLI:

```bash
agentbox run tasks -n 8 --out runs/a            # interrupted...
agentbox run tasks -n 8 --out runs/a --resume   # continues where it stopped
agentbox run tasks -n 16 --out runs/a --resume  # grows the groups to 16 samples
```

`runs/a/run.json` stores a hash of the model, agent, sandbox, reward and seed
configuration. `--resume` refuses to mix configurations unless you pass
`--force`. `n` and operational knobs (`--warm-pool`, `--keep-failed`) are not
part of the hash.

## Sinks

| Sink | Writes |
| --- | --- |
| `JsonlSink(path, append=True)` | One JSON object per line; `.jsonl.gz` compresses. Supports resume (`completed()`) |
| `DirectorySink(dir)` | One file per trajectory (`{task}_{sample}_{rollout12}.json`) |
| `MemorySink()` | A list (tests, small runs) |
| `CallbackSink(fn)` | Calls `fn(traj)` (sync or async), e.g. push to a trainer queue |
| `MultiSink(a, b, ...)` | Fans out |

Read trajectories back lazily with `iter_trajectories(path)`, which accepts a
file, a `.jsonl(.gz)` file, a directory or a benchmark output directory.

## The CLI

`agentbox run` is `run_to_sink` with a JSONL sink and a run manifest:

```bash
agentbox run tasks/ data.jsonl.gz -n 8 -c 32 --model-concurrency 64 \
  --out runs/exp42 --compress --seed 7 --retries 2 --warm-pool 4 --split train
```

It prints a summary (success, mean reward, steps, p90 duration, infra failures,
rollouts per minute, failure kinds) and returns exit code 1 if
`--min-success-rate` is not met.
