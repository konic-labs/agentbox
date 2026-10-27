# ART (OpenPipe)

ART's training loop consumes `TrajectoryGroup`s, each a list of
`art.Trajectory` objects whose `messages_and_choices` mix prompt messages with
the policy's `Choice`s. AgentBox rolls out against ART's OpenAI-compatible
inference server and converts the results.

```bash
pip install "agentbox-rl[art]"
```

## Online loop

```python
import art
from agentbox import ModelConfig, ParallelRunner, iter_tasks
from agentbox.integrations.art import to_art_groups

backend = art.LocalBackend()
model = art.TrainableModel(name="agent-001", project="agentbox", base_model="Qwen/Qwen3-1.7B")
await model.register(backend)

policy = ModelConfig(
    model=model.inference_model_name or model.name,
    base_url=model.inference_base_url,
    api_key=model.inference_api_key,
    capture_tokens=True,                           # vLLM token ids + logprobs -> trainable Choices
    capture_logprobs=True,
)
runner = ParallelRunner(model=policy, concurrency=32)
tasks = list(iter_tasks("tasks/", split="train"))

for step in range(await model.get_step(), 100):
    batch = tasks[(step * 8) % len(tasks):][:8]
    trajectories = await runner.run_tasks(batch, n_per_task=8)
    groups = to_art_groups(trajectories)           # one group per task; infra failures dropped
    await backend.train(model, groups, learning_rate=1e-6)
```

Written against `openpipe-art` 0.5.x (`backend.train(model, groups, ...)`).
ART's `LocalBackend` trains only on Choices with logprobs unless
`allow_training_without_logprobs=True`, which is why `capture_logprobs`
matters here.

## Conversion

| AgentBox | ART |
| --- | --- |
| assistant turn with captured token ids, logprobs and a reconstructable prompt | `Choice` with `logprobs.content` (`token_id:<id>` tokens) plus vLLM `prompt_token_ids` / `token_ids`, so ART trains on exactly the sampled tokens |
| other messages | message dicts |
| `reward` | `reward` |
| verifier success | `metrics["correct"]` (from the verifier, not the loop status) |
| `metrics` | `duration`, `steps`, `tool_calls`, `tool_errors`, `model_calls`, `total_tokens`, `reward/<component>` |
| ids, status, failure kind, task metadata | `metadata`, scalarised to JSON strings for releases that only accept scalars |
| `group_id` | one `TrajectoryGroup` per group; `trainable_only=True` drops infrastructure failures |

`to_art_dict(traj)` returns the same structure as plain JSON (no ART install
needed), and `agentbox traj export -f art` writes it for offline pipelines.
Pass `choices=False` to emit plain messages only.

Validated against ART's own `tokenize_trajectory_groups` with Qwen3-1.7B on
vLLM: every captured assistant token was trained with its sampled logprob.
TITO trajectories formed one ART sequence each, and chat-completions
trajectories one sequence per turn, because prefix breaks force ART to split
them ([why](../models.md#prefix-breaks-and-tito)).

## Migrating from `Trajectory.to_art*`

`Trajectory.to_art_dict()` and `Trajectory.to_art()` still work but are
deprecated. Use `agentbox.integrations.art.to_art_dict` /
`to_art_trajectory` / `to_art_groups`.
