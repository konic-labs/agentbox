# SkyRL

SkyRL environments (`skyrl_gym`) are text-in/text-out. `AgentBoxTextEnv`
renders the tools into the system prompt (Hermes format), parses
`<tool_call>` blocks out of each model action, runs them in the sandbox, and
returns the tool results as the next observation. When the episode ends it
returns the AgentBox reward.

```bash
pip install skyrl-gym
```

```python
from agentbox.integrations.skyrl import register

register()          # registers "agentbox" with skyrl_gym (entry point agentbox.integrations.skyrl:AgentBoxTextEnv)
```

## Dataset

Rows carry the task in `extras["agentbox_task"]` (task JSON), or
`extras["task_id"]` with `$AGENTBOX_TASKS` set. Settings come from
`$AGENTBOX_SKYRL_CONFIG` (YAML/JSON with `sandbox`, `agent` and `rewards`
sections).

```python
import json
from agentbox import iter_tasks

rows = [
    {"prompt": [{"role": "user", "content": t.prompt_text()}],
     "env_class": "agentbox",
     "extras": {"agentbox_task": t.model_dump_json()}}
    for t in iter_tasks("tasks/", split="train")
]
```

## Episode

| Call | Behaviour |
| --- | --- |
| `init(prompt)` | Provisions and seeds the sandbox (retried on transient failures); returns the system + user messages and `{"task_id", "tools"}` |
| `step(action)` | Parses and runs tool calls; returns the tool results as user observations (`<tool_response>` blocks) |
| final `step` | When the model answers without tool calls, submits, or reaches `agent.max_steps`: runs the verifier, closes the sandbox, returns `reward`, `done=True` and metadata (`success`, `status`, `reward_breakdown`) |

Tool responses go back as user turns, which is what Qwen and Hermes chat
templates expect in text-only loops.
