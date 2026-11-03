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
export AGENTBOX_OPENRLHF_CONFIG=$PWD/agentbox-openrlhf.yaml   # sandbox / agent / rewards / remote
python -m openrlhf.cli.train_ppo_ray \
  --train.agent_func_path $(python -c "import agentbox.integrations.openrlhf as m; print(m.__file__)") \
  ...                                                        # your usual OpenRLHF arguments
```

The module provides both forms OpenRLHF has used: an `AgentExecutor` class
(`AgentExecutorBase` subclass, built lazily) and the `run_episode(...)`
coroutine behind it. It accepts the `images` argument OpenRLHF 0.11 passes;
episodes are text-only.

### On several GPUs

[`examples/openrlhf/run.sh`](../../examples/openrlhf/run.sh) runs GRPO on
Qwen3-0.6B with one vLLM engine per GPU and the actor colocated with them
(ZeRO-2, sleep mode), 8 tasks × 8 samples per step:

```bash
python examples/verl/make_coding_tasks.py data/tasks
EPISODES=3 examples/openrlhf/run.sh        # 27 tasks x 3 passes = 10 steps
```

Every engine actor is one AgentBox process with its own warm pool, and
`sandbox.host_max_sandboxes` in
[`agentbox-openrlhf.yaml`](../../examples/openrlhf/agentbox-openrlhf.yaml)
bounds them together. Sandboxes die with the engine actors Ray SIGKILLs at the
end, and the script prunes after training as a fallback. It takes only full
rollout batches
(`--data.max_samples`): OpenRLHF trains on a last, partial batch too, and on two
ranks that batch hung ZeRO-2's gradient all-reduce until NCCL's 10-minute
timeout.

Measured with OpenRLHF 0.11.2 on 2× RTX PRO 6000 (96 GB), 10 steps with
token-level importance-sampling correction
(`--algo.advantage.is_correction_level token`):

| | |
| --- | --- |
| Time per step | 20.3 s (12.1 s generation) |
| `vllm_kl`: sampled vs recomputed logprobs, both engines | 0.0005–0.0013 per step |
| Token sequences per episode (`agentbox_sequences`) | 1.0 |
| Infrastructure failures | 0 |
| Sandboxes in use, peak | 64, plus up to 9 idle in the engines' warm pools |
| Containers left after the run | 0 |

The episodes' logprobs match what the actor recomputes, on both engines. The
reward moved between 0.45 and 0.70 with the task mix of each batch. Ten steps
check the pipeline, not learning: the verl recipe's reward also rose only
after its first ten steps.

### Remote sandboxes

With a `remote` section in the settings, AgentBox's agent loop still runs on
the engine, token-in/token-out, and only the tool calls and the verifier run in
an `agentbox serve` session (`agentbox.env.remote.remote_rollout`). The GPU
nodes need no Docker. See [server](../server.md#trainer-adapters).

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
verifier crash, a lost sandbox server) are reported through
`extra_logs.agentbox_infra_failure`, so you can monitor or filter them.
