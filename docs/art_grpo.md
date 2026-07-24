# ART / GRPO with AgentBox

AgentBox collects **real-rollout trajectories** in Docker. [OpenPipe ART](https://github.com/OpenPipe/ART)
consumes them for GRPO (and related) training. AgentBox is not a trainer.

## Offline path (simplest)

```text
1. Define / generate Tasks
2. ParallelRunner or bench run → Trajectory JSON files
3. traj.to_art_dict() per trajectory
4. Group by task_id (group size G for GRPO)
5. Feed groups into ART training loop
```

### Collect a group for one task

```python
from agentbox import ParallelRunner, Task
from agentbox.config import ModelConfig, RolloutConfig, SandboxConfig, AgentConfig

task = Task.from_json("tasks/my_task/task.json")
config = RolloutConfig(
    model=ModelConfig(model="qwen…", base_url="http://localhost:8000/v1", api_key="EMPTY"),
    agent=AgentConfig(max_steps=25),
    sandbox=SandboxConfig(ensure_pytest=True),
)
runner = ParallelRunner(concurrency=4, config=config)
# G rollouts of the same task
trajs = await runner.run_tasks([task], n_per_task=4)
group = [t.to_art_dict() for t in trajs]
# rewards: [t["reward"] for t in group]
```

### Portable ART shape

```python
art = traj.to_art_dict()
assert "messages_and_choices" in art
assert "reward" in art
assert art["metrics"]["correct"] in (0.0, 1.0)
```

Install live ART objects (optional):

```bash
pip install agentbox[art]
traj.to_art()  # art.Trajectory
```

See `examples/art_integration.py` and `examples/art_grpo_offline.py`.

## Online path (sketch)

```text
for step in train:
  sample task batch
  AgentBox ParallelRunner → G trajectories / task  (same tools protocol as ART)
  compute group-relative advantages from rewards (pytest verify)
  ART policy update (GRPO)
```

Wire the model `base_url` to ART’s inference server when training online so
rollouts and updates share weights.

## Hybrid rewards

Primary reward should remain **programmatic verify** (`FinalStatus.SUCCESS`).

Optional secondary signals (not default):

- LLM trajectory judge (soft)
- ART RULER-style judges

Combine outside AgentBox or via custom `shaped_reward` hooks; keep verify as
the hard gate for “correct.”

## CLI export

```bash
agentbox export traj.json --format art -o art.json
agentbox run-dir tasks/ -m mock -c 4 -o trajectories/
```
