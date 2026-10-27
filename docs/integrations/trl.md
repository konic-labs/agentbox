# TRL

TRL's `GRPOTrainer` supports multi-turn tool use through `environment_factory`.
For every rollout it creates (or reuses) an environment instance, calls
`reset(**row)` with the dataset row, exposes the instance's public methods as
tools, and calls `get_reward()` when the rollout ends. AgentBox builds that
class for you, backed by a real sandbox per episode.

```bash
pip install "agentbox-rl[trl]"      # trl>=1.0, transformers, datasets, accelerate
```

## Minimal script

```python
from trl import GRPOConfig, GRPOTrainer

from agentbox import AgentConfig, ResourceLimits, RewardSpec, SandboxConfig, iter_tasks
from agentbox.integrations.trl import make_environment_factory, tasks_to_dataset

tasks = list(iter_tasks("tasks/"))
Env = make_environment_factory(
    tasks,
    sandbox=SandboxConfig(limits=ResourceLimits(memory_mb=1024), warm_pool=8),
    agent=AgentConfig(max_steps=8),
    tools=["list_files", "read_file", "write_file", "run_tests"],
    rewards=[RewardSpec(name="verifier"), RewardSpec(name="step_penalty", params={"per_step": 0.01})],
)
trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=tasks_to_dataset(tasks),
    environment_factory=Env,
    args=GRPOConfig(
        output_dir="grpo-agentbox",
        num_generations=8,
        per_device_train_batch_size=8,
        max_completion_length=1024,
        max_tool_calling_iterations=8,
        learning_rate=1e-6,
        chat_template_kwargs={"enable_thinking": False},
        bf16=True,
    ),
)
trainer.train()
```

`agentbox init --trainer trl` writes this script, and
`agentbox train trl tasks/ --model Qwen/Qwen3-0.6B --steps 100 --group-size 8`
runs the same setup from the CLI (`--use-vllm`, `--lora`, `--tools`,
`--max-tool-calls`, ...), using the sandbox and agent sections of
`agentbox.yaml`.

## How it maps

| TRL | AgentBox |
| --- | --- |
| dataset row (`prompt`, `task_id`) | `tasks_to_dataset` / `tasks_to_rows`: conversational prompt (system + user) plus `task_id` |
| `reset(**row)` | provisions and seeds a fresh sandbox for `task_id`; returns `None` when the row has a prompt (TRL appends the return value to it), or the task prompt for environment-owned data |
| tool methods | `list_files`, `read_file`, `write_file`, `edit_file`, `run_command`, `run_tests`, `search_files`, `apply_patch`, `submit`, each with a Google-style docstring so TRL derives exact JSON schemas |
| `get_reward()` | runs the verifier and reward components, releases the sandbox, and keeps the trajectory on `env.last_trajectory` |

Choose tools with `tools=[...]`, or omit it to use the agent's tool mode. A
dataset without `task_id` rows works too: `reset()` then cycles through the
tasks, which suits `max_steps`-bounded training on environment-owned data.

## Notes

- TRL tokenises the conversation itself, which is what TRL's loss expects.
  AgentBox's token capture is not involved.
- Each environment instance holds at most one live sandbox. TRL keeps a pool of
  instances, one per concurrent rollout, so `per_device_train_batch_size`
  bounds the number of concurrent sandboxes per process.
- **Infrastructure failures are unscorable, not 0.** When provisioning fails
  (after one retry) or the verifier crashes, `get_reward()` returns `None`.
  TRL then leaves the completion out of the group baseline and gives it zero
  advantage, so a Docker hiccup never becomes a fake negative. Training keeps
  running, tools tell the model the environment is unavailable, and TRL logs a
  warning for the sample.
- Validated with TRL 1.16 on Qwen3-0.6B (L4 GPU). GRPO steps trained with
  non-zero reward variance, and each rendered prompt contains the task exactly
  once.
