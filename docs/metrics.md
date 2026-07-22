# Metrics

Lightweight offline aggregation over completed trajectories.

## AggregateMetrics

```python
from agentbox import aggregate_trajectories

stats = aggregate_trajectories(trajs)
```

| Field | Meaning |
| --- | --- |
| `n` | Trajectory count |
| `success_rate` | Fraction with `final_status == success` |
| `mean_reward` | Average reward |
| `mean_steps` | Average agent steps |
| `mean_tool_calls` | Average tool calls |
| `mean_duration_s` | Average wall duration |
| `by_status` | Counts per `FinalStatus` value |
| `by_task` | Per-`task_id` n / success_rate / mean_reward / mean_steps |

## Per-trajectory metrics

Filled during rollout in `Trajectory.metrics`:

- Loop: steps, tool_calls, model_calls, tokens  
- Infra: sandbox_create_s, seed_s, verify_s, duration_s  

ART export maps a subset into float/bool `metrics` including `correct`.

## What AgentBox does not include

- Live training dashboards (W&B etc.)  
- Built-in web UI for trajectory browsing  

Those remain external; export JSON/JSONL/ART and analyze with your stack.
