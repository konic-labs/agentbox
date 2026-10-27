# Training

AgentBox does not train models. It gives an RL framework what agentic training
needs outside the trainer: an isolated environment per episode, tools, a
trustworthy reward, and exact records of what was sampled.

## Choosing an integration

| Your setup | Use | How generation works |
| --- | --- | --- |
| TRL `GRPOTrainer` (single node, quickest start) | [TRL](integrations/trl.md) | TRL generates and calls AgentBox tools as environment methods; AgentBox scores the sandbox |
| verl (multi-node, FSDP/Megatron, async rollouts) | [verl](integrations/verl.md) | verl's `ToolAgentLoop` owns tokens; AgentBox runs tools and returns the reward |
| OpenRLHF (Ray, vLLM engines) | [OpenRLHF](integrations/openrlhf.md) | AgentBox runs the episode token-in/token-out on OpenRLHF's own vLLM engine |
| SkyRL (`skyrl_gym` text environments) | [SkyRL](integrations/skyrl.md) | Text in, text out; AgentBox parses `<tool_call>` blocks |
| OpenPipe ART | [ART](integrations/art.md) | AgentBox rolls out against ART's server; trajectories become `TrajectoryGroup`s |
| Your own trainer | `AgentBoxEnv` (below), `ParallelRunner.stream_groups`, `agentbox traj export -f tokens` | You decide |

All of them share the same tasks, sandboxes, tools, rewards, failure
attribution and trajectory schema, so you can switch trainers without touching
tasks or reward code, and benchmark checkpoints with `agentbox bench` on the
same environment you trained in.

## The environment API

`AgentBoxEnv` exposes one sandboxed episode step by step. The trainer
generates; AgentBox executes and scores.

```python
from agentbox import AgentBoxEnv, AgentConfig, SandboxConfig

env = AgentBoxEnv(sandbox=SandboxConfig(warm_pool=4), agent=AgentConfig(tools="extended"))

obs = await env.reset(task)          # provisions + seeds a fresh sandbox
obs.messages                          # [system, user] in OpenAI format
obs.tools                             # OpenAI tool schemas

while not env.done:
    action = await policy(obs.messages, obs.tools)
    # action: assistant text, an OpenAI assistant message dict, or a list of ToolCalls
    step = await env.step(action)     # runs the tool calls, returns the tool messages
    obs.messages.extend(step.messages)
    if step.done:                     # final answer, submit, max_steps or deadline
        break

result = await env.evaluate()         # verifier + reward components
result.reward, result.success, result.breakdown, result.failure_kind
traj = env.trajectory()               # the full Trajectory (same schema as Rollout)
await env.close()                     # or reuse: the next reset() starts a fresh sandbox
```

- `env.call_tool(name, arguments)` runs a single tool outside the step
  protocol. Trainers that dispatch tools themselves use it, as the TRL and
  verl adapters do.
- `SyncAgentBoxEnv` is a blocking facade with the same methods for synchronous
  trainers. It runs a shared background event loop.
- `RemoteEnv` (`agentbox.env.remote`) has the same surface over HTTP, so the
  trainer can run on GPU nodes while sandboxes live on Docker hosts
  ([server](server.md)).

## Exact token data

Policy-gradient losses need the log-probabilities of the tokens that were
actually sampled. Re-tokenising the text afterwards silently changes tokens,
for example around tool-call JSON and `<think>` blocks.

- **Capture.** With `capture_tokens` and `capture_logprobs` on a vLLM endpoint,
  every assistant turn stores its prompt and completion token ids and sampler
  logprobs ([models](models.md#token-capture)).
- **Single sequences.** With **TITO**, each trajectory is one prefix-consistent
  token sequence with a response mask (model tokens = 1, tool-result tokens = 0)
  ([models](models.md#prefix-breaks-and-tito)).
- **Export.** `agentbox traj export -f tokens` writes
  `prompt_ids`/`response_ids`/`response_mask`/`response_logprobs` rows and flags
  whether each trajectory was `exact` (no prefix breaks).

The verl and OpenRLHF adapters work at the token level natively. The TRL
adapter lets TRL tokenise, which is what TRL expects.

## RL utilities

```python
from agentbox.rl import group_advantages, group_stats, is_zero_variance, pass_at_k

group_advantages([1.0, 0.0, 1.0, 0.0], method="grpo")    # (r - mean) / (std + eps)
group_advantages(rewards, method="drgrpo")               # r - mean (Dr. GRPO)
group_advantages(rewards, method="rloo")                 # r - mean of the other samples
is_zero_variance([1.0, 1.0, 1.0])                        # True: no GRPO signal
group_stats(groups)    # frac_all_solved / frac_all_failed / frac_mixed / frac_zero_variance / mean_group_reward_std
pass_at_k(n=8, c=3, k=1)                                 # unbiased estimator (Chen et al., 2021)
```

Two practical consequences for GRPO-style training:

- **Drop zero-variance groups.** A group whose samples all got the same reward
  has zero advantage everywhere. `traj stats` reports their fraction, and
  `traj export -f grpo --drop-zero-variance` removes them.
- **Train on the learnable band.** Probe tasks with a few samples, then keep
  the ones the policy sometimes solves:

  ```bash
  agentbox run tasks/ -n 4 --out runs/probe
  agentbox task select tasks/ --runs runs/probe --min-pass 0.01 --max-pass 0.99 -o learnable.jsonl
  ```

  The same selection is available in Python as
  `agentbox.tasks.filter.pass_rates` / `filter_by_pass_rate`, and
  `sample_curriculum` orders tasks by their `difficulty` metadata.
- **Never score infrastructure as policy.** A sandbox or endpoint failure scored
  as reward 0 inside a group creates a fake negative advantage. AgentBox labels
  those rollouts with an infrastructure `failure_kind`, retries them, and keeps
  them out of exports and adapters' rewards.

## Throughput checklist

- Warm the caches before training: `agentbox cache warm tasks/` (setup
  snapshots) and `warm_pool: K` in the sandbox config.
- Size `concurrency` to your Docker host. Each sandbox holds a CPU and memory
  limit; see [performance](performance.md).
- Use `model_concurrency` to cap in-flight requests to the inference server
  separately from sandboxes.
- Stream (`stream_groups`, `run_to_sink`) instead of collecting lists.
