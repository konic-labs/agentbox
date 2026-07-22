# Agent

The agent subsystem turns a task description + tools + model into a multi-turn
conversation with tool calls.

## Components

| Piece | Role |
| --- | --- |
| `Agent` | Ergonomic config facade (`model`, tools, max_steps, …) |
| `AgentLoop` | Step loop: complete → tools → append → repeat |
| `prompts` | Default system prompts for structured / shell modes |

## Agent facade

```python
from agentbox import Agent

agent = Agent(
    model="glm-5.2",
    base_url="https://api.featherless.ai/v1",
    api_key="...",
    tools="structured",          # or "shell" / "custom" / ["read_file", "run_command"]
    custom_tools=[...],
    builtins=True,
    max_steps=40,
    temperature=0.7,
)
# agent.model_client  # ModelClient
# agent.config        # AgentConfig
```

`Rollout.run(task=task, agent=agent)` uses both.

## Loop algorithm

```txt
messages = [system, user]
step = 0
while step < max_steps:
  if episode_deadline exceeded: TIMEOUT
  response = model.complete(messages, tools)   # step_timeout_s
  append assistant message
  if no tool_calls: return final_answer
  tool_messages = execute_many(tool_calls)     # step_timeout_s
  append tool messages; record ToolCallRecords
  step += 1
return MAX_STEPS
```

### Stop reasons

| `stop_reason` | Meaning |
| --- | --- |
| `final_answer` | Model replied without tools |
| `max_steps` | Hit step budget |
| `timeout` | Episode or step timeout |
| `error` | Uncaught failure in loop |

Note: `final_status` on the loop is provisional. **Final reward/status** come
from the verifier after the loop (see [Tasks](tasks.md), [Runner](runner.md)).

## Timeouts

| Level | Config | Scope |
| --- | --- | --- |
| Model / tool batch | `AgentConfig.step_timeout_s` | Single complete or execute_many |
| Episode | `AgentConfig.episode_timeout_s` | Checked between steps via monotonic deadline |
| Model HTTP | `ModelConfig.timeout_s` | Client-level |

## Parallel tool calls

If the model returns multiple `tool_calls`, they run concurrently when
`parallel_tool_calls=True`. Results keep **call order** when appended.

## System prompts

Default structured prompt (summary): coding agent in Linux container, workspace
`/workspace`, prefer small edits, finish with a final message without tools.

Shell mode: only `run_command` available; use Unix tools.

Overrides:

- `AgentConfig.system_prompt` — full override (supports `{workspace_dir}`)
- `Task.system_prompt_extra` — appended to the rendered base prompt

## Tool dropping

For ablation / generalization:

- `drop_tools=["run_tests"]` — always remove  
- `drop_tools_prob=0.1` — drop each remaining tool with probability  
- Seeded via `RolloutConfig.seed`  

Available names are recorded on the trajectory metadata as `tools_available`.

## Process reward shaping

After verification, optional penalties (see `AgentConfig`):

- `step_penalty * steps`
- `timeout_penalty` if timed out  
- `max_steps_penalty` if max steps hit  

Binary verifier reward remains the base signal.
