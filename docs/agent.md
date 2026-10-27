# Agent loop

`AgentLoop` drives one episode: it sends the conversation to the model,
executes the tool calls it returns, appends the results, and repeats until a
stop condition. `Rollout`, `ParallelRunner` and the CLI use it directly, and
`AgentBoxEnv` exposes the same tool execution step by step for trainers that
generate themselves.

## Stop conditions

| `stop_reason` | When | `final_status` without a verifier pass |
| --- | --- | --- |
| `final_answer` | The model answered without tool calls | `failed` |
| `submitted` | A terminal tool (`submit`) was called | `failed` |
| `max_steps` | `max_steps` model turns were used | `max_steps` |
| `timeout` | The episode deadline (`episode_timeout_s`) passed | `timeout` |
| `context_length` | The conversation no longer fits the model context, or a generation was cut off (`finish_reason=length`) | `failed` |
| `error` | The model endpoint failed after retries, or an internal error occurred | `error` |

The **verifier decides success**. A rollout that hit `max_steps` after fixing
the code is still a `success`. Stop reasons caused by the policy have
`failure_kind=policy` (valid training signal). Endpoint failures are
`model_api` (infrastructure; retried and excluded from training data). See
[rewards](rewards.md#infrastructure-failures-are-not-rewards).

## Deadlines and budgets

- `step_timeout_s` bounds one turn (the model call and then its tool calls),
  never beyond the episode deadline.
- `episode_timeout_s` bounds the whole episode. Tool timeouts requested by the
  model (`run_command(timeout_s=...)`) are clamped to `tool_timeout_s` and to
  the time left.
- `max_steps` counts model turns. `Task.max_steps` and `Task.episode_timeout_s`
  override them per task.

When a step times out mid-tool-call, the pending calls still get tool messages
("timed out"), so the recorded conversation is always valid for training and
replay.

## Tool execution

Tool calls from one assistant turn run **in emission order**. A model that
writes a file and then runs the tests in one turn gets the tests of the new
file. With `parallel_tool_calls: true` (default), **consecutive read-only**
calls (`read_file`, `list_files`, `search_files`, custom `read_only=True`
tools) run concurrently, and everything else runs one at a time. After a
terminal call (`submit`), the remaining calls of the turn are answered with
"Skipped" and the episode ends.

## Tool-calling protocols

| `tool_protocol` | Wire format | Use when |
| --- | --- | --- |
| `native` (default) | OpenAI `tools` / `tool_calls` | The server parses tool calls (vLLM `--enable-auto-tool-choice --tool-call-parser ...`, SGLang, OpenAI, most hosted APIs) |
| `hermes` | Tool schemas rendered into the system prompt inside `<tools>`; calls parsed from `<tool_call>{...}</tool_call>` blocks in the content | The server has no tool parser, or you want AgentBox's lenient parser |

With `hermes`, assistant turns are replayed as raw text, and tool results go
back as `role=tool` or, with `tool_response_role: user`, as user turns wrapped
in `<tool_response>` (what Qwen and Hermes chat templates expect).

**Salvaging native calls.** When a server's tool parser rejects a call (vLLM's
hermes parser on slightly invalid JSON, for example), the call comes back as
plain text. With `salvage_tool_calls: true` (default), AgentBox parses the
`<tool_call>` blocks itself, strips them from the content and executes them. A
block that cannot be parsed becomes an `invalid_tool_call` whose error message
tells the model the expected format. Salvaged calls are counted in
`metrics.salvaged_tool_calls`, and `agentbox run` warns when it happens; a
non-zero count usually means the server's parser and the model disagree, so
consider TITO or `hermes`.

## Context length

- `ModelConfig.adaptive_max_tokens` (default on): when `prompt + max_tokens`
  exceeds the server's context, the request is retried with the room left.
- When fewer than `min_completion_tokens` remain, or the server reports a
  context overflow, the episode stops with `stop_reason=context_length`. That
  is a **policy** failure, not an error.

## Reasoning models

`<think>` content and the `reasoning` / `reasoning_content` fields are
recorded on the assistant message (`include_thinking`, default on).
`strip_think` moves inline `<think>...</think>` out of the content.
`send_reasoning` controls whether recorded reasoning is sent back to the server
on later turns; most chat templates drop it, and strict servers reject it.

## Prompts

The system prompt depends on the tool mode and names the workspace directory.
`AgentConfig.system_prompt` replaces it (`{workspace_dir}` is substituted, and
other braces are left alone). `Task.system_prompt_extra` appends task-specific
instructions. The user prompt is `Task.prompt` (a template with
`{description}` and `{files}`) or `Task.description`.

## Hooks

`AgentHooks` observes or steers an episode without forking the loop:

```python
from agentbox import AgentHooks, ParallelRunner
from agentbox.trajectory.schema import Message
from agentbox.types import MessageRole


class Nudge(AgentHooks):
    async def before_model(self, step, messages):      # step is 0-based; messages is a copy
        if step == 10:
            # returned messages are appended, sent to the model and recorded
            return Message(role=MessageRole.USER, content="Run the tests before you finish.")
        return None

    async def after_model(self, step, message):
        ...                               # every assistant message

    async def after_tools(self, step, outcomes):
        ...                               # ToolOutcome list for the turn


runner = ParallelRunner(model=model, hooks=Nudge())
```

`Rollout.run(..., hooks=...)` accepts the same object.

## Determinism

`RolloutConfig.seed` derives a per-sample seed from `(seed, task_id,
sample_index)`. It is sent as the sampling seed and drives tool dropout, so
reruns of a run are as reproducible as the server allows. A shared
`MockModelClient` plays the same script for every rollout, which makes tests
deterministic under concurrency.
