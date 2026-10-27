# OpenRLHF

OpenRLHF's agent mode calls a user-provided *agent function* for every prompt
and expects token-level results: the observation tokens, which ranges are
actions, and the rollout logprobs. `agentbox.integrations.openrlhf` is such a
module. It runs each episode **token-in/token-out** on OpenRLHF's own vLLM
engine and tokenizer, so the returned tokens and logprobs are exactly what was
sampled.

## Dataset

Each prompt's `label` selects the task: either a task id resolved against
`$AGENTBOX_TASKS`, or an inline task JSON string. The prompt text itself is
not used for the episode; AgentBox renders the system and user prompt from
the task with the model's chat template.

```python
import json
from agentbox import iter_tasks

with open("data/train.jsonl", "w") as fh:
    for t in iter_tasks("tasks/", split="train"):
        fh.write(json.dumps({"prompt": t.prompt_text(), "label": t.task_id}) + "\n")
```

## Train

```bash
export AGENTBOX_TASKS=$PWD/tasks
export AGENTBOX_OPENRLHF_CONFIG=$PWD/agentbox-openrlhf.yaml   # sandbox / agent / rewards sections
python -m openrlhf.cli.train_ppo_ray \
  --train.agent_func_path $(python -c "import agentbox.integrations.openrlhf as m; print(m.__file__)") \
  ...                                                        # your usual OpenRLHF arguments
```

The module provides both forms OpenRLHF has used: an `AgentExecutor` class
(`AgentExecutorBase` subclass, built lazily) and the `run_episode(...)`
coroutine behind it.

## What the episode returns

| Key | Value |
| --- | --- |
| `observation_tokens` | Prompt plus every generated and tool-result token, as one sequence |
| `action_ranges` | `(start, end)` spans of model-generated tokens |
| `rollout_log_probs` | Sampler logprobs aligned with the tokens (0 for non-action tokens) |
| `reward` / `scores` | The AgentBox reward |
| `truncated` | The context was exhausted |
| `extra_logs` | `agentbox_success`, `agentbox_infra_failure`, `agentbox_steps`, `agentbox_tool_errors`, `agentbox_sequences` |

Infrastructure failures that happen before the first model call (sandbox or
seeding) are retried up to three times. Later infrastructure failures (a
verifier crash) are reported through `extra_logs.agentbox_infra_failure`, so
you can monitor or filter them.
