# Rewards

A rollout's reward is a **weighted sum of components**. Each component sees the
task, the verifier result and the agent loop result, and the per-component
values are stored in `Trajectory.reward_breakdown`, so you can always see why a
sample got its reward.

```yaml
# agentbox.yaml (or RolloutConfig.rewards, or --reward on the CLI)
rewards:
  - {name: verifier}                                    # ground truth from the task's verifier
  - {name: step_penalty, params: {per_step: 0.01}}
  - {name: valid_tool_call_ratio, weight: 0.1}          # format reward
  - {name: my_pkg.rewards:style_bonus, weight: 0.2}     # any importable callable
```

```bash
agentbox run tasks --reward verifier --reward "step_penalty:1.0,per_step=0.01"
```

The CLI form is `name[:weight][,key=value]...`. Without any reward
configuration the reward is `verifier` plus the legacy `AgentConfig` penalties
(`step_penalty`, `timeout_penalty`, `max_steps_penalty`), when set.

## Built-in components

| Name | Value | Params |
| --- | --- | --- |
| `verifier` | The verifier's reward (binary, fraction, stdout or custom, see [tasks](tasks.md#reward-modes)) | none |
| `success` | `value` if the verifier succeeded, else 0 | `value=1.0` |
| `tests_fraction` | passed / total tests (0 without counts) | none |
| `step_penalty` | `-per_step × steps` | `per_step=0.01` |
| `timeout_penalty` | `-penalty` if the episode timed out | `penalty=0.5` |
| `max_steps_penalty` | `-penalty` if `max_steps` was reached | `penalty=0.5` |
| `tool_error_penalty` | `-per_error × errors`, optionally capped | `per_error=0.05`, `cap=None` |
| `valid_tool_call_ratio` | Share of tool calls that were well-formed and named a real tool (1.0 with no calls) | none |
| `length_penalty` | `-per_token × max(0, completion_tokens − budget)` | `per_token=0.0001`, `budget=0` |
| `submitted` | `value` if the agent called `submit` | `value=1.0` |
| `tamper_penalty` | `-penalty` if the verifier detected tampering | `penalty=1.0` |
| `llm_judge` | A 0–1 rubric score from any OpenAI-compatible model | `model` (ModelConfig dict, required), `rubric`, `max_transcript_chars=6000` |

Using the same component twice is allowed. Later occurrences are stored as
`name#2`, `name#3`, and so on.

### LLM judges

```yaml
rewards:
  - {name: verifier, weight: 0.8}
  - name: llm_judge
    weight: 0.2
    params:
      model: {model: gpt-4.1-mini, api_key_env: OPENAI_API_KEY}
      rubric: "1 if the fix is minimal and idiomatic, 0 if it special-cases the tests."
```

The judge sees the task description, the verifier outcome and a middle-truncated
transcript. It answers with `{"score": x, "reason": ...}`, and unparseable
answers score 0. One HTTP client per endpoint is shared by all rollouts on an
event loop.

## Custom components

Register a factory under a name:

```python
from agentbox import register_reward
from agentbox.rewards import RewardContext


@register_reward("short_answer_bonus")
def short_answer_bonus(max_chars: int = 200):
    def fn(ctx: RewardContext) -> float:
        last = ctx.loop.messages[-1].content if ctx.loop else ""
        return 1.0 if last and len(last) <= max_chars else 0.0
    return fn
```

Or skip registration and name any importable function as
`package.module:function` (or `package.module.function`). It is called as
`fn(ctx, **params)` and may be `async`.

`RewardContext` has `task`, `verify` (a `VerifyResult` with `success`,
`reward`, `exit_code`, `stdout`, `tests_passed/failed/total`, `tampered`,
`details`) and `loop` (messages, tool-call records, `steps`, `stop_reason`,
usage), plus `ctx.steps` and `ctx.stop_reason` shortcuts.

## Infrastructure failures are not rewards

When a rollout fails for a reason the policy did not cause, scoring it 0 would
teach the model that a correct strategy is bad. Every trajectory has a
`failure_kind`:

| `failure_kind` | Meaning | Trainable |
| --- | --- | --- |
| `none` | Success | yes |
| `policy` | The agent failed: wrong answer, max steps, timeout, context exhausted, malformed calls | yes |
| `sandbox` | The container could not be created or died | no |
| `seed` | Starter files or `setup_commands` failed | no |
| `setup_check` | A benchmark setup check failed | no |
| `model_api` | The model endpoint failed after retries | no |
| `verifier` | The verifier crashed (not the same as the tests failing) | no |
| `internal` | A bug in AgentBox or a custom tool | no |

`Trajectory.is_trainable` is true for `none` and `policy`. Runners retry
infrastructure failures (`--retries`, `ParallelRunner(max_retries=...)`), every
exporter drops them by default, and `agentbox traj stats` reports them
separately.

For group-relative methods (GRPO, RLOO, ...), see
[training: RL utilities](training.md#rl-utilities) for advantages and
zero-variance filtering.
