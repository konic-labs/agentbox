# Trajectories

Every rollout, whether produced by the CLI, `Rollout`, `ParallelRunner`,
`AgentBoxEnv` or a trainer adapter, is recorded as one self-contained
`Trajectory` (schema version 2).

## Schema

| Field | Content |
| --- | --- |
| `task_id`, `task_hash` | The task and a hash of its exact revision |
| `run_id`, `rollout_id` | Experiment id and unique rollout id |
| `group_id`, `sample_index` | GRPO group membership (`<task_id>@<run>`, 0..n-1) |
| `messages` | System, user, assistant (tool calls, reasoning, `finish_reason`, `usage`, `latency_s`, `tokens`) and tool messages |
| `tool_call_records` | Per call: name, arguments, result preview, `is_error`, `duration_s`, `truncated`, step |
| `reward`, `reward_breakdown` | Total reward and its components ([rewards](rewards.md)) |
| `final_status` | `success`, `failed`, `timeout`, `max_steps`, `error` |
| `stop_reason` | `final_answer`, `submitted`, `max_steps`, `timeout`, `context_length`, `error` |
| `failure_kind` | `none`, `policy`, or an infrastructure kind ([rewards](rewards.md#infrastructure-failures-are-not-rewards)) |
| `verify` | Verifier result: `success`, `reward`, `exit_code`, `command`, `stdout`/`stderr` (head + tail), `tests_passed/failed/total`, `tampered`, `timed_out`, `duration_s` |
| `metrics` | `steps`, `tool_calls`, `tool_errors`, `model_calls`, token usage, `duration_s`, `sandbox_create_s`, `seed_s`, `verify_s`, `model_latency_s`, `tool_latency_s`, `attempts`, `salvaged_tool_calls` |
| `model`, `tool_mode`, `tools` | The policy id and the tool schemas it saw |
| `metadata` | Task metadata, `split`, `group_key`, `submitted_answer`, `kept_sandbox`, ... |
| `error`, `created_at`, `finished_at` | |

Helpers: `traj.success`, `traj.is_trainable`, `traj.token_trace()`,
`traj.to_json()`, `traj.save(path)` (atomic) and `Trajectory.load(path)`.
Version-1 files are upgraded on load.

## Storage

```python
from agentbox import JsonlSink, iter_trajectories
from agentbox.trajectory.io import write_jsonl

for t in iter_trajectories("runs/a"):              # file, .jsonl(.gz), dir, or bench run dir
    ...
write_jsonl(trajectories, "subset.jsonl.gz")
```

JSONL (optionally gzip) is the default store. It is appendable, streamable and
resumable. One-file-per-trajectory directories (`DirectorySink`) suit
benchmark reports. Readers are lazy, so files larger than memory are fine.

## Tooling

```bash
agentbox traj stats runs/a                       # metrics, tool usage, failure kinds, GRPO group stats
agentbox traj show runs/a/trajectories.jsonl     # one rollout in detail
agentbox traj validate runs/a                    # schema + conversation validity (dangling calls, unknown ids, order)
agentbox traj filter runs/a -o hard.jsonl --max-reward 0 --trainable-only
agentbox traj filter runs/a -o uses_patch.jsonl --tool apply_patch --grep "Traceback"
agentbox traj diff a.json b.json                 # outcome, tool sequence, first divergence
agentbox traj replay t.json --task tasks/x       # re-execute the recorded tool calls, compare the reward
agentbox traj render runs/a -o dash.html         # self-contained HTML dashboard
```

`traj stats` includes group statistics for GRPO data: the number of groups,
the fractions that are all-solved, all-failed and mixed, the zero-variance
fraction, and the mean in-group reward std. Zero-variance groups carry no
GRPO signal.

`traj replay` is a determinism and reward-stability check. It runs the same
actions in a fresh sandbox and reports whether the verifier agrees.

## Metrics

```python
from agentbox import MetricsAccumulator, aggregate_trajectories

acc = MetricsAccumulator()
for t in iter_trajectories("runs/a"):
    acc.add(t)                      # O(1) memory per trajectory
m = acc.result()
print(m.success_rate, m.mean_reward, m.reward_std, m.p90_duration_s, m.infra_failures,
      m.by_failure_kind, m.tool_error_rate, m.salvaged_tool_calls)
```

## Exporters

```bash
agentbox traj export runs/a -f FORMAT -o out.jsonl [--successful-only] [--min-reward X] [--include-infra]
```

```python
from agentbox import ExportFilter, export

stats = export(iter_trajectories("runs/a"), "grpo", "grpo.jsonl",
               filter=ExportFilter(min_reward=None), drop_zero_variance=True, advantage="rloo")
```

| Format | One row per | Content | Typical consumer |
| --- | --- | --- | --- |
| `chat` | trajectory | `{"messages", "tools", ...}` OpenAI chat format | SFT (TRL `SFTTrainer`, axolotl) |
| `sharegpt` | trajectory | `{"conversations": [{"from", "value"}], "tools"}` | ShareGPT-style SFT tooling |
| `dpo` | group | `{"prompt", "chosen", "rejected"}`: best vs worst sample, if the margin is at least `--min-margin` | TRL `DPOTrainer` |
| `grpo` | group | `rewards`, `advantages` (`--advantage grpo\|drgrpo\|rloo`) and samples; `--drop-zero-variance` | offline GRPO and analysis |
| `tokens` | token sequence (`--per-turn`: per turn) | `prompt_ids`, `response_ids`, `response_mask`, `response_logprobs`, `exact` | custom trainers, verl/OpenRLHF-style pipelines |
| `hf` | trajectory | Parquet via `datasets`; `--push-to-hub REPO` | the HF Hub |
| `art` | trajectory | ART `Trajectory` dicts; Choices with token ids when captured | OpenPipe ART ([guide](integrations/art.md)) |

Every exporter drops infrastructure failures unless `--include-infra` is given.
Register your own with `agentbox.export.register_exporter("name")`.
