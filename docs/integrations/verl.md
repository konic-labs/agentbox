# verl

`AgentBoxAgentLoop` subclasses verl's `ToolAgentLoop`. verl keeps full
control of generation and tokenisation (continuous token sequences, no
re-tokenisation drift). AgentBox provisions a sandbox per trajectory, executes
every tool call in it, and sets the trajectory's `reward_score` from the
verifier and reward components.

## Single GPU: a tested recipe

Everything in [`examples/verl/`](../../examples/verl) was run end to end on one
L4 (24 GB) with 4 vCPUs and 15 GB of RAM, using verl 0.9.1, vLLM 0.24 and
torch 2.11 on CUDA 13. The run is GRPO with a full fine-tune of Qwen3-0.6B,
thinking on, over 36 small coding tasks:

```bash
pip install "verl[vllm]==0.9.1" agentbox-rl
python examples/verl/make_coding_tasks.py data/tasks        # 27 train + 9 held-out eval tasks
python examples/verl/prepare_data.py --out data/verl        # verl parquet rows -> agent "agentbox"
STEPS=30 examples/verl/run_grpo_single_gpu.sh               # ~2.5 min per step
```

There are three task families: create a file, fix a bug, and implement a
function from a stub. Every task has protected tests, planted-file detection
and `pytest_fraction` partial credit. Each step trains on 8 tasks × 8 samples,
about 100K tokens.

### Results

<img src="../assets/readme/training.svg" width="100%" alt="Reward while training, in two 30-step GRPO runs of Qwen3-0.6B with verl on one L4 GPU. The 4-step mean of the training reward rises from about 0.57 to 0.86, and validation on 9 held-out tasks from 0.70 to 0.89. Held-out success is 48.6% for the base model and 77.8% after 30 steps; the first run reached 83.3%.">

The recipe was run twice for 30 steps. Run 2 used the script exactly as
shipped, plus an extra checkpoint at step 15. The reward is the AgentBox
verifier's. Training and validation numbers are verl's own metrics.
Validation covers the 9 held-out tasks with greedy decoding.

| | Run 1 | Run 2 |
| --- | --- | --- |
| Training reward, steps 1–5 → 25–29 (64 samples per step) | 0.56 → 0.82 | 0.57 → 0.86 |
| Validation reward, before training → last validation | 0.59 → 0.85 (step 25) | 0.70 → 0.89 (step 30) |
| Response length in tokens, steps 1–5 → 25–29 | 1005 → 876 | 1021 → 805 |
| Host RAM in use (`perf/cpu_memory_used_gb`) | 12.0 → 12.9 GB | 10.3 → 10.5 GB |
| Time per step | 160 s | 150 s |

Run 1 has no step-30 validation, because it ran out of host RAM after its
final checkpoint save
([#103](https://github.com/konic-labs/agentbox/issues/103)). Run 2 has the
fix. Its saves at steps 15 and 30 each gave their memory back, and it exited
cleanly. The two runs scored 0.59 and 0.70 before training, so read a single
validation point on 9 tasks as ±0.1. The evaluation below has 72 episodes per
model.

The step-30 checkpoints and the base model were evaluated with AgentBox
itself: the held-out tasks as a frozen suite, temperature 0.6, thinking on.

| | Base | Run 1, step 30 | Run 2, step 30 |
| --- | --- | --- | --- |
| Held-out suite: success (9 tasks × 8) | 48.6% | **83.3%** | **77.8%** |
| … create / fix / implement | 32/32 · 0/24 · 3/16 | 32/32 · 13/24 · 15/16 | 32/32 · 11/24 · 13/16 |
| … mean reward | 0.634 | 0.866 | 0.810 |
| Train split: success (27 tasks × 4) | 48.1% | **87.0%** | **85.2%** |
| … create / fix / implement | 44/48 · 3/36 · 5/24 | 44/48 · 29/36 · 21/24 | 44/48 · 26/36 · 22/24 |

`bench compare` pairs runs by task. On the held-out suite, both runs improved
the same 4 tasks and regressed none: +34.7 points for run 1 (95% CI 9.7 to
62.5), and +29.2 for run 2 (6.9 to 54.2). The suite is small and comes from
the training families, so read this as a check that the pipeline learns, not
as a benchmark score.

The trajectories show what changed. The base model often ran the tests and
stopped, or answered in prose without calling a tool (6 of 72 held-out
episodes). Both trained models write the fix: 1.28 and 1.31 `write_file` calls
per episode, up from 0.50. They almost always act (1 and 0 episodes without a
tool call), and they think about a quarter less. Run 1 also stopped running
the tests first (0.19 `run_tests` calls per episode, down from 0.65), while
run 2 kept more of them (0.46).

To repeat the evaluation:

```bash
vllm serve checkpoints/agentbox/global_step_30/actor/huggingface --tokenizer Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b-grpo --enable-auto-tool-choice --tool-call-parser hermes
agentbox bench create suites/eval --from-tasks data/tasks/eval.jsonl --suite-id mini-eval \
  --name "mini eval" --n 8 --tools list_files,read_file,write_file,run_tests --max-steps 6
agentbox bench run suites/eval --models-file models-base.yaml -o bench/base
agentbox bench run suites/eval --models-file models-grpo.yaml -o bench/grpo
agentbox bench compare bench/base/report.json bench/grpo/report.json
```

```yaml
# models-grpo.yaml; models-base.yaml names the base model and its server
models:
  - model_id: policy
    model: qwen3-0.6b-grpo
    base_url: http://localhost:8000/v1
    temperature: 0.6
    top_p: 0.95
    max_tokens: 3072
    extra_body: {chat_template_kwargs: {enable_thinking: true}}
```

Give both models the same `model_id` so `bench compare` pairs them
([benchmarks](../benchmarks.md)). Start one vLLM server at a time
on a shared GPU. vLLM sizes its KV cache from the memory it sees free at
start-up, so a server that starts while another is still allocating gets a
negative budget.

### Settings that matter on a small host

| Problem | Setting |
| --- | --- |
| verl's v1 trainer starts TransferQueue storage units (8 × ~0.4 GB of host RAM) | `trainer.use_v1=False`, the legacy trainer, which verl 0.9.1 still ships but marks as deprecated |
| Every Ray actor costs host RAM; 15 GB is tight | `actor_rollout_ref.rollout.agent.num_workers=1`, `reward.num_workers=1`, `+ray_kwargs.ray_init.include_dashboard=False`, `+ray_kwargs.ray_init.object_store_memory=1500000000`, `RAY_memory_usage_threshold=0.97` |
| A checkpoint save builds full fp32 state dicts on the CPU. glibc keeps the freed heap (~2.5 GB) in the worker, and the validation that follows runs out of host RAM | `MALLOC_MMAP_THRESHOLD_=1048576`. It also lowered host RAM in use from 12.0–12.9 GB to 10.3–10.5 GB, with no slowdown |
| Ray schedules one logical CPU per actor | `ray_kwargs.ray_init.num_cpus=16` on a 4-vCPU box |
| No flash-attn wheel for torch 2.11 / CUDA 13 | `actor_rollout_ref.model.use_remove_padding=False`, `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` |
| Without padding removal, fp32 logits over a 152K vocabulary exhaust the GPU | `actor_rollout_ref.model.use_fused_kernels=True`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `ppo_max_token_len_per_gpu=8192` |
| Padded rows cost compute, so size the budgets to the data | `data.max_prompt_length=1024`, `data.max_response_length=3072` (observed: prompts ~640 tokens, responses up to ~2.4K) |
| verl's default tool response budget (256 tokens) cuts pytest output | `actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024` |
| The rollout defaults to tensor parallel 2 | `actor_rollout_ref.rollout.tensor_model_parallel_size=1` |
| Metrics are block-buffered under Ray | `PYTHONUNBUFFERED=1` |

verl stops its workers with SIGKILL at the end of a run, so their exit handlers
never run. Their sandboxes die with them all the same, and the script runs
`agentbox sandbox prune` as a fallback
([sandbox lifecycle](../sandbox.md#lifecycle-and-cleanup)).

## Multi-GPU: a tested recipe

[`run_grpo_multi_gpu.sh`](../../examples/verl/run_grpo_multi_gpu.sh) runs the
same training on every GPU of a node, or on several nodes. Its algorithm
settings are those of the single-GPU recipe: 8 tasks × 8 samples per step,
learning rate 2e-6, six assistant turns, thinking on. It turns on padding
removal and drops the small-host workarounds.

```bash
pip install "verl[vllm]==0.9.1" agentbox-rl          # and flash-attn for your torch and GPU
python examples/verl/make_coding_tasks.py data/tasks
python examples/verl/prepare_data.py --out data/verl
STEPS=30 examples/verl/run_grpo_multi_gpu.sh         # one vLLM replica per GPU
TP=2 examples/verl/run_grpo_multi_gpu.sh             # replicas span two GPUs
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPUS` | every GPU | GPUs per node |
| `NNODES` | 1 | nodes of a Ray cluster started beforehand with `ray start` |
| `TP` | 1 | tensor parallel size of each vLLM replica |
| `WORKERS` | 2 × GPUs × nodes | agent-loop workers, one AgentBox process each |
| `AGENTBOX_VERL_CONFIG` | `agentbox-verl.yaml` | AgentBox settings; [`agentbox-verl-remote.yaml`](../../examples/verl/agentbox-verl-remote.yaml) for remote sandboxes |

It was run on one node with 2× RTX PRO 6000 Blackwell (96 GB each, PCIe), 48
vCPUs and 500 GB of RAM, using verl 0.9.1, vLLM 0.24, torch 2.11 on CUDA 13
and flash-attn 2.8.3 built from source for sm_120.

### Results on two GPUs

30 steps, one vLLM replica per GPU, four agent-loop workers:

| | One L4: run 1 / run 2 | Two RTX PRO 6000 |
| --- | --- | --- |
| Training reward, steps 1–5 → 25–29 | 0.56 → 0.82 / 0.57 → 0.86 | 0.57 → 0.88 |
| Validation reward, before training → last validation | 0.59 → 0.85 / 0.70 → 0.89 | 0.70 → 0.76 (0.89 and 0.93 at steps 20 and 25) |
| Response length in tokens, steps 1–5 → 25–29 | 1005 → 876 / 1021 → 805 | 1028 → 595 |
| Time per step | 160 s / 150 s | 9.8 s |
| Sandboxes in use at once, peak | | 64, plus up to 20 idle in warm pools |
| Host RAM in use, peak | | 41 GB |

The training reward follows the single-GPU curves: its 4-step means were
0.60, 0.57, 0.65, 0.76, 0.85, 0.79 and 0.85. Validation on the 9 held-out tasks
with greedy decoding moved between 0.67 and 0.93 from one evaluation to the
next, so read a single point as ±0.1, as on one GPU. The answers shortened
more than on one GPU. The 30 steps and seven validations took 8.9 minutes.

### Throughput

Steps 1–5 of each setup, 64 trajectories per step:

| Setup | Step | Generation | Model update |
| --- | --- | --- | --- |
| 1 GPU | 17.4 s | 11.1 s | 4.6 s |
| 2 GPUs, a replica each (`TP=1`) | 13.2 s | 9.3 s | 2.5 s |
| 2 GPUs, one replica (`TP=2`) | 14.1 s | 10.2 s | 2.4 s |
| 2 GPUs, remote sandboxes | 13.3 s | 9.5 s | 2.5 s |

The model update halves with the second GPU, but generation barely changes:
64 multi-turn episodes do not fill two GPUs, and a step waits for its slowest
trajectories. At four times the batch, 32 tasks × 8 samples per step, the
second GPU pays off (three steps each):

| Setup | Step | Generation | Model update | Sandboxes in use, peak |
| --- | --- | --- | --- | --- |
| 1 GPU | 51.8 s | 27.7 s | 18.3 s | 256 |
| 2 GPUs | 32.8 s | 19.6 s | 9.4 s | 256 |
| 2 GPUs, `host_max_sandboxes: 64` | 35.3 s | 22.4 s | 9.3 s | 64 |

Two GPUs ran 7.8 trajectories per second against 4.9 on one, 1.6 times as
many. With the cap at 64, the 256 trajectories of a step never had more than
64 sandboxes in use, plus 32 idle in the four workers' warm pools, and a step
took 8% longer. No run left a container behind.

### Remote sandboxes

With `AGENTBOX_VERL_CONFIG=examples/verl/agentbox-verl-remote.yaml` and
`agentbox serve` on the same node, five steps took 13.3 s each, against
13.2 s with local sandboxes. A tool call took 0.124 s on average, against
0.118 s locally over the same steps. The server created 344 sessions and
refused or failed none. Across a network, each tool call and each episode's
start and end add a round trip.

### Settings that matter on several GPUs

| Problem | Setting |
| --- | --- |
| On a cluster started with `ray start`, workers do not inherit the launching shell's environment | the script passes `AGENTBOX_VERL_CONFIG` and `AGENTBOX_TASKS` as `++ray_kwargs.ray_init.runtime_env.env_vars.*`; the config variable may hold one-line JSON when the nodes share no filesystem |
| vLLM's custom all-reduce failed with `CUDA error ... custom_all_reduce.cuh: 'invalid argument'` at `TP=2` on our PCIe-connected pair | the script sets `+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True` when `TP` > 1, and NCCL does the reductions |
| Ray SIGKILLs the agent-loop workers: the 30-step run left 20 idle warm-pool containers | sandboxes now die with their worker (`die_with_owner`): a run whose launcher had no prune left none. The script still runs `agentbox sandbox prune` after training (`--ray-address auto` on several nodes) for older or unbound sandboxes |
| A large batch starts every sandbox at once | `sandbox.host_max_sandboxes` ([sandbox](../sandbox.md#host-wide-cap)) |
| No flash-attn wheel exists for torch 2.11 / CUDA 13 / sm_120 | `FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=44 pip wheel --no-build-isolation --no-cache-dir flash-attn==2.8.3` with `ninja` on `PATH`: about 4 minutes on 48 vCPUs, while without ninja it compiles one file at a time. `--no-cache-dir` keeps pip from reusing a wheel built against another torch |

## Your own run

### 1. Dataset

```python
from agentbox import AgentConfig, iter_tasks
from agentbox.integrations.verl import write_verl_parquet

agent = AgentConfig(tools=["list_files", "read_file", "write_file", "run_tests"])
write_verl_parquet(iter_tasks("tasks/", split="train"), "data/train.parquet", agent=agent)
write_verl_parquet(iter_tasks("tasks/", split="eval"), "data/val.parquet", agent=agent)
```

Each row routes to the `agentbox` agent loop (`agent_name`). It carries the
user prompt and a system prompt rendered for the agent's tools, and embeds the
task as JSON in `extra_info.agentbox_task`. For large datasets, pass
`embed_task=False` to keep only `extra_info.task_id`, and point
`$AGENTBOX_TASKS` at the task source so workers can resolve the ids.

### 2. Register the agent loop

```yaml
# agent_loops.yaml
- name: agentbox
  _target_: agentbox.integrations.verl.AgentBoxAgentLoop
```

### 3. Sandbox, agent and reward settings

Workers read `$AGENTBOX_VERL_CONFIG`: a path to a YAML or JSON file with the
same sections as `agentbox.yaml`
([example](../../examples/verl/agentbox-verl.yaml)), or that text itself. A
`remote` section runs the sandboxes on an `agentbox serve` host
([server](../server.md#trainer-adapters)). Use the same tools as the dataset's
system prompt. Tool schemas come from AgentBox, not from verl's
`tool_config_path`.

### 4. Train

Start from
[`run_grpo_single_gpu.sh`](../../examples/verl/run_grpo_single_gpu.sh) on one
small GPU and from
[`run_grpo_multi_gpu.sh`](../../examples/verl/run_grpo_multi_gpu.sh)
otherwise. `multi_turn.format`, `multi_turn.max_assistant_turns` and
`agent.agent_loop_config_path` are read by verl's agent-loop code, and these
switches have moved between verl releases. Check the multi-turn example of
your version.

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
- verl parses tool calls with its own strict Hermes parser. It drops malformed
  calls, and the turn ends. AgentBox's tool-call salvage applies only to its
  own agent loop.
