# verl

`AgentBoxAgentLoop` subclasses verl's `ToolAgentLoop`. verl keeps full
control of generation and tokenisation (continuous token sequences, no
re-tokenisation drift). AgentBox provisions a sandbox per trajectory, executes
every tool call in it, and sets the trajectory's `reward_score` from the
verifier and reward components.

## 1. Dataset

```python
from agentbox import iter_tasks
from agentbox.integrations.verl import write_verl_parquet

write_verl_parquet(iter_tasks("tasks/", split="train"), "data/train.parquet")
write_verl_parquet(iter_tasks("tasks/", split="eval"), "data/val.parquet")
```

Each row routes to the `agentbox` agent loop (`agent_name`), carries the
system and user prompt, and embeds the task as JSON in
`extra_info.agentbox_task`. For large datasets pass `embed_task=False` to keep
only `extra_info.task_id`, and point `$AGENTBOX_TASKS` at the task source so
workers can resolve ids.

## 2. Register the agent loop

```yaml
# agent_loops.yaml
- name: agentbox
  _target_: agentbox.integrations.verl.AgentBoxAgentLoop
```

## 3. Sandbox, agent and reward settings

Workers read `$AGENTBOX_VERL_CONFIG` (YAML or JSON) with the same sections as
`agentbox.yaml`:

```yaml
# agentbox-verl.yaml
sandbox:
  limits: {cpu_count: 1.0, memory_mb: 1024}
  warm_pool: 8
agent:
  tools: structured
rewards:
  - {name: verifier}
  - {name: step_penalty, params: {per_step: 0.005}}
```

## 4. Train

```bash
export AGENTBOX_VERL_CONFIG=$PWD/agentbox-verl.yaml
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=data/train.parquet data.val_files=data/val.parquet \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=$PWD/agent_loops.yaml \
  ...                                          # the rest of your usual verl config
```

`multi_turn.format`, `multi_turn.max_assistant_turns` and
`agent.agent_loop_config_path` are read by verl's agent-loop code. The
async/multi-turn switches have moved between verl releases, so check the
multi-turn example of your version. Tool schemas come from AgentBox (the
agent's tool mode), not from verl's `tool_config_path`.

## Behaviour

- `reward_score` is the AgentBox reward. `extra_fields["agentbox"]` holds
  `success`, `status`, `failure_kind`, `reward_breakdown` and `infra_failure`.
- verl has no notion of an unscorable sample. For infrastructure failures (the
  sandbox could not be provisioned, or the verifier crashed), the adapter zeroes
  the sample's `response_mask`, so its tokens contribute no gradient, and flags
  it in `extra_fields`. A provisioning failure does not crash the agent loop:
  tool calls return an error, and the batch continues.
- Tool calls run in AgentBox's sandbox with AgentBox's timeouts, output caps and
  path jail. verl's `max_tool_response_length` truncation still applies to what
  goes back into the token sequence.
