# SkyRL

SkyRL environments (`skyrl_gym`) are text-in/text-out. `AgentBoxTextEnv`
renders the tools into the system prompt (Hermes format), parses
`<tool_call>` blocks out of each model action, runs them in the sandbox, and
returns the tool results as the next observation. When the episode ends it
returns the AgentBox reward.

```bash
pip install "skyrl[skyrl-train]==0.4.0" "vllm-router==0.1.14"   # later routers dropped a field skyrl 0.4.0 reads
```

```python
from agentbox.integrations.skyrl import register

register()          # registers "agentbox" with skyrl_gym (entry point agentbox.integrations.skyrl:AgentBoxTextEnv)
```

SkyRL builds its environments inside the training process, a Ray task, and
the registry is per process. Register there, not only in the launcher:
[`examples/skyrl/main_agentbox.py`](../../examples/skyrl/main_agentbox.py) is
SkyRL's entrypoint with that one change.

## Dataset

SkyRL hands every column other than `prompt` and `env_class` to the
environment, so the task goes in a top-level `agentbox_task` column (task
JSON), or a `task_id` column with `$AGENTBOX_TASKS` set. The environment
renders the real system prompt, with the tools, when the episode starts; the
`prompt` column only serves SkyRL's length filter.

```python
from agentbox import iter_tasks

rows = [
    {"prompt": [{"role": "user", "content": t.prompt_text()}],
     "env_class": "agentbox",
     "agentbox_task": t.model_dump_json(exclude_defaults=True)}
    for t in iter_tasks("tasks/", split="train")
]
```

[`examples/skyrl/prepare_data.py`](../../examples/skyrl/prepare_data.py) writes
these rows to Parquet. Rows that nest the fields under an `extras` column work
too.

Settings come from `$AGENTBOX_SKYRL_CONFIG` (a path, or the YAML/JSON text,
with `sandbox`, `agent`, `rewards` and `remote` sections). Keep
`agent.max_steps` equal to SkyRL's `generator.max_turns`.

## Train on several GPUs

```bash
python examples/verl/make_coding_tasks.py data/tasks
python examples/skyrl/prepare_data.py --out data/skyrl
EPOCHS=3 examples/skyrl/run.sh        # one vLLM engine per GPU, FSDP policy colocated
```

[`run.sh`](../../examples/skyrl/run.sh) runs GRPO on Qwen3-0.6B with 8 tasks ×
8 samples per step and removes the sandboxes Ray's SIGKILL left behind when it
ends. SkyRL steps the environments on its env threads
(`environment.skyrl_gym.max_env_workers`); the script gives every trajectory
of a step its own.

Measured with skyrl 0.4.0 on 2× RTX PRO 6000 (96 GB), 9 steps (three epochs
over 27 tasks):

| | |
| --- | --- |
| Time | 4 minutes for the 9 steps, with a checkpoint at each epoch's end |
| Reward per step | 0.36, 0.62, 0.68, 0.67, 0.48, 0.70, 0.55, 0.53, 0.80 |
| pass@8 per batch | 0.625 at step 1, then 0.75–1.0 |
| Sandboxes in use, peak | 64, plus 4 idle in the warm pool |
| Containers left after the run | 0 |

Every episode's tool calls were parsed from the model's text on the
AgentBox side. Nine steps check the pipeline, not learning.

## Episode

| Call | Behaviour |
| --- | --- |
| `init(prompt)` | Provisions and seeds the sandbox (retried on transient failures); returns the system + user messages and `{"task_id", "tools"}` |
| `step(action)` | Parses and runs tool calls; returns the tool results as user observations (`<tool_response>` blocks) |
| final `step` | When the model answers without tool calls, submits, or reaches `agent.max_steps`: runs the verifier, closes the sandbox, returns `reward`, `done=True` and metadata (`success`, `status`, `reward_breakdown`, `infra_failure`) |

Tool responses go back as user turns, which is what Qwen and Hermes chat
templates expect in text-only loops. With a `remote` settings section the
episode runs in an `agentbox serve` session ([server](../server.md)); if the
server is lost mid-episode, the episode ends with reward 0 and
`metadata.infra_failure`.
