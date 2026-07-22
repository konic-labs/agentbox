# Trajectories

Trajectories are the primary artifact of AgentBox: full multi-turn histories
with tools, metrics, reward, and export helpers.

## Native schema

```python
class Trajectory:
    task_id: str
    run_id: str
    messages: list[Message]
    tool_call_records: list[ToolCallRecord]
    reward: float
    final_status: FinalStatus   # success|failed|timeout|error|max_steps
    metrics: TrajectoryMetrics
    metadata: dict
    error: str | None
    model: str | None
    tool_mode: str | None
    tools: list[dict] | None    # OpenAI tool schemas used
    created_at / finished_at
```

### Message

OpenAI-compatible roles: `system`, `user`, `assistant`, `tool`.

Assistant tool calls use nested `ToolCall` / `FunctionCall` with JSON-string
`arguments`. Tool messages carry `tool_call_id` and `content`.

`Message.to_openai_dict()` produces Chat Completions message objects.

### TrajectoryMetrics

Includes `steps`, `tool_calls`, `model_calls`, token counts (if available),
`duration_s`, `sandbox_create_s`, `seed_s`, `verify_s`.

## Recording

`TrajectoryRecorder` is used by `Rollout` to accumulate messages, tool records,
and metrics, then `finalize(reward, final_status, ...)`.

## Persistence

```python
traj.save("trajectories/run.json")
traj = Trajectory.load("trajectories/run.json")
```

JSON is Pydantic `model_dump_json`.

### JSONL datasets

```python
from agentbox.trajectory.formats import export_jsonl
export_jsonl(trajs, "dataset.jsonl")
```

One Trajectory JSON object per line.

## ART export

Offline dict matching OpenPipe ART’s conceptual shape:

```python
art = traj.to_art_dict()
# {
#   "messages_and_choices": [... OpenAI message dicts ...],
#   "tools": [...],
#   "reward": 1.0,
#   "metrics": {"duration", "steps", "tool_calls", "correct", ...},
#   "metadata": {"task_id", "run_id", "final_status", ...}
# }
```

Rules:

- Message dicts only (portable; no live `Choice` objects required)  
- `metrics["correct"]` is `1.0` iff `final_status == success`  
- Tool turns preserve OpenAI `tool_calls` / `tool` roles  

Live object (optional extra):

```bash
uv pip install -e ".[art]"
```

```python
art_traj = traj.to_art()  # art.Trajectory
```

### GRPO groups

Grouping is the caller’s job (or `ParallelRunner.run_groups`):

```python
groups = await runner.run_groups(tasks, group_size=4)
art_groups = [[t.to_art_dict() for t in g] for g in groups]
```

Example: `examples/art_integration.py`.

## Final status mapping (rollout)

| Situation | Typical `final_status` |
| --- | --- |
| Verifier success | `success` |
| Verifier fail, normal stop | `failed` |
| Episode/step timeout | `timeout` |
| Max steps then fail verify | `max_steps` |
| Seed / infra failure | `error` |

Reward is still taken from the verifier (and optional shaping) when verification
runs.

## CLI export

```bash
agentbox export traj.json --format art -o traj.art.json
agentbox export traj.json --format jsonl -o traj.jsonl
```
